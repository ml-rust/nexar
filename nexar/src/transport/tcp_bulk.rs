use crate::error::{NexarError, Result};
use crate::transport::BulkTransport;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};

type TaggedReceiverMap = HashMap<u64, Arc<Mutex<mpsc::Receiver<Vec<u8>>>>>;

/// Shared state between the recv loop and the transport.
///
/// When a tagged frame arrives before `recv_bulk_tagged` has been called for
/// that tag, the data is buffered in `pending`. When a receiver registers,
/// any pending data is flushed into the new channel.
struct RecvState {
    senders: HashMap<u64, mpsc::Sender<Vec<u8>>>,
    pending: HashMap<u64, Vec<Vec<u8>>>,
}

/// Bulk transport over raw TCP (no TLS).
///
/// Carries a `[tag: u64 LE][len: u64 LE][payload]` framing so that tagged
/// collectives can bypass QUIC's AES-256-GCM overhead for large tensor data.
/// Tag 0 is used for untagged `send_bulk`/`recv_bulk`.
pub struct TcpBulkTransport {
    writer: Mutex<tokio::io::WriteHalf<TcpStream>>,
    /// Default (tag=0) receive channel.
    untagged_rx: Mutex<mpsc::Receiver<Vec<u8>>>,
    /// Shared state with the recv loop (senders + pending buffer).
    state: Arc<Mutex<RecvState>>,
    /// Per-tag receivers, each independently lockable so concurrent tags don't block.
    tagged_rx: Mutex<TaggedReceiverMap>,
    /// Background recv task handle.
    _recv_handle: tokio::task::JoinHandle<()>,
}

/// Trait for bulk transports that support tagged sends/receives.
///
/// TCP can carry tags in its wire format, unlike RDMA which has no tag concept.
/// This lets all collectives use bulk acceleration, not just untagged ones.
pub trait TaggedBulkTransport: BulkTransport {
    /// Send raw bytes with a u64 tag via the bulk transport.
    fn send_bulk_tagged<'a>(&'a self, tag: u64, data: &'a [u8]) -> BoxFuture<'a, Result<()>>;

    /// Receive raw bytes for a specific tag via the bulk transport.
    ///
    /// `expected_size` is advisory (may be used for buffer pre-allocation hints).
    fn recv_bulk_tagged<'a>(
        &'a self,
        tag: u64,
        expected_size: usize,
    ) -> BoxFuture<'a, Result<Vec<u8>>>;
}

impl TcpBulkTransport {
    /// Create a `TcpBulkTransport` from an already-connected `TcpStream`.
    pub fn from_stream(stream: TcpStream) -> Self {
        let (reader, writer) = tokio::io::split(stream);

        let (untagged_tx, untagged_rx) = mpsc::channel(64);
        let state = Arc::new(Mutex::new(RecvState {
            senders: HashMap::new(),
            pending: HashMap::new(),
        }));

        let recv_state = Arc::clone(&state);
        let recv_handle = tokio::spawn(async move {
            recv_loop(reader, untagged_tx, recv_state).await;
        });

        Self {
            writer: Mutex::new(writer),
            untagged_rx: Mutex::new(untagged_rx),
            state,
            tagged_rx: Mutex::new(HashMap::new()),
            _recv_handle: recv_handle,
        }
    }

    /// Write a tagged frame: `[tag: u64 LE][len: u64 LE][payload]`.
    async fn write_frame(&self, tag: u64, data: &[u8]) -> Result<()> {
        let mut writer = self.writer.lock().await;
        writer
            .write_all(&tag.to_le_bytes())
            .await
            .map_err(|e| NexarError::transport(format!("tcp bulk write tag: {e}")))?;
        writer
            .write_all(&(data.len() as u64).to_le_bytes())
            .await
            .map_err(|e| NexarError::transport(format!("tcp bulk write len: {e}")))?;
        writer
            .write_all(data)
            .await
            .map_err(|e| NexarError::transport(format!("tcp bulk write payload: {e}")))?;
        writer
            .flush()
            .await
            .map_err(|e| NexarError::transport(format!("tcp bulk flush: {e}")))?;
        Ok(())
    }

    /// Get or create a per-tag receiver. Returns an `Arc<Mutex<Receiver>>` that
    /// can be locked independently of other tags (no shared lock across `.await`).
    async fn get_tag_receiver(&self, tag: u64) -> Arc<Mutex<mpsc::Receiver<Vec<u8>>>> {
        // Fast path: already registered.
        {
            let map = self.tagged_rx.lock().await;
            if let Some(rx) = map.get(&tag) {
                return Arc::clone(rx);
            }
        }
        // Slow path: create channel, register sender, then flush pending outside lock.
        let (tx, rx) = mpsc::channel(64);
        let flush_tx = tx.clone();
        let pending_data = {
            let mut st = self.state.lock().await;
            let pending = st.pending.remove(&tag);
            st.senders.insert(tag, tx);
            pending
        };
        // Flush outside the lock to avoid holding it across .await.
        if let Some(data_vec) = pending_data {
            for data in data_vec {
                let _ = flush_tx.send(data).await;
            }
        }
        let rx_arc = Arc::new(Mutex::new(rx));
        self.tagged_rx.lock().await.insert(tag, Arc::clone(&rx_arc));
        rx_arc
    }
}

impl BulkTransport for TcpBulkTransport {
    fn send_bulk<'a>(&'a self, data: &'a [u8]) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.write_frame(0, data))
    }

    fn recv_bulk<'a>(&'a self, _expected_size: usize) -> BoxFuture<'a, Result<Vec<u8>>> {
        Box::pin(async move {
            self.untagged_rx
                .lock()
                .await
                .recv()
                .await
                .ok_or_else(|| NexarError::transport("tcp bulk connection closed"))
        })
    }
}

impl TaggedBulkTransport for TcpBulkTransport {
    fn send_bulk_tagged<'a>(&'a self, tag: u64, data: &'a [u8]) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.write_frame(tag, data))
    }

    fn recv_bulk_tagged<'a>(
        &'a self,
        tag: u64,
        _expected_size: usize,
    ) -> BoxFuture<'a, Result<Vec<u8>>> {
        Box::pin(async move {
            let rx_arc = self.get_tag_receiver(tag).await;
            rx_arc
                .lock()
                .await
                .recv()
                .await
                .ok_or_else(|| NexarError::transport("tcp bulk connection closed"))
        })
    }
}

/// Maximum TCP bulk frame size (4 GiB, same limit as the QUIC router).
const MAX_TCP_FRAME_SIZE: usize = 4 * 1024 * 1024 * 1024;

/// Background loop: read frames and route to the appropriate channel.
async fn recv_loop(
    mut reader: tokio::io::ReadHalf<TcpStream>,
    untagged_tx: mpsc::Sender<Vec<u8>>,
    state: Arc<Mutex<RecvState>>,
) {
    let mut tag_buf = [0u8; 8];
    let mut len_buf = [0u8; 8];
    loop {
        if let Err(e) = reader.read_exact(&mut tag_buf).await {
            tracing::debug!("tcp bulk recv loop ended: {e}");
            return;
        }
        if let Err(e) = reader.read_exact(&mut len_buf).await {
            tracing::debug!("tcp bulk recv loop ended reading len: {e}");
            return;
        }
        let tag = u64::from_le_bytes(tag_buf);
        let len = u64::from_le_bytes(len_buf) as usize;

        if len > MAX_TCP_FRAME_SIZE {
            tracing::warn!(len, "tcp bulk: frame too large, closing connection");
            return;
        }

        let mut payload = vec![0u8; len];
        if let Err(e) = reader.read_exact(&mut payload).await {
            tracing::debug!("tcp bulk recv loop ended reading payload: {e}");
            return;
        }

        if tag == 0 {
            if untagged_tx.send(payload).await.is_err() {
                return;
            }
        } else {
            // Clone the sender (or grab pending ref) outside the lock to avoid
            // holding it across the channel send `.await`.
            let tx = {
                let st = state.lock().await;
                st.senders.get(&tag).cloned()
            };
            if let Some(tx) = tx {
                if tx.send(payload).await.is_err() {
                    return;
                }
            } else {
                let mut st = state.lock().await;
                st.pending.entry(tag).or_default().push(payload);
            }
        }
    }
}

/// Listen on a random port and return the listener for one peer to accept.
pub async fn tcp_bulk_listen(
    addr: std::net::SocketAddr,
) -> Result<(TcpListener, std::net::SocketAddr)> {
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| NexarError::transport(format!("tcp bulk listen: {e}")))?;
    let local = listener
        .local_addr()
        .map_err(|e| NexarError::transport(format!("tcp bulk local_addr: {e}")))?;
    Ok((listener, local))
}

/// Connect to a peer's TCP bulk listener and create both ends.
pub async fn tcp_bulk_connect(addr: std::net::SocketAddr) -> Result<TcpBulkTransport> {
    let stream = TcpStream::connect(addr)
        .await
        .map_err(|e| NexarError::transport(format!("tcp bulk connect: {e}")))?;
    stream
        .set_nodelay(true)
        .map_err(|e| NexarError::transport(format!("tcp bulk set_nodelay: {e}")))?;
    Ok(TcpBulkTransport::from_stream(stream))
}

/// Accept one connection from a TCP bulk listener.
pub async fn tcp_bulk_accept(listener: &TcpListener) -> Result<TcpBulkTransport> {
    let (stream, _addr) = listener
        .accept()
        .await
        .map_err(|e| NexarError::transport(format!("tcp bulk accept: {e}")))?;
    stream
        .set_nodelay(true)
        .map_err(|e| NexarError::transport(format!("tcp bulk set_nodelay: {e}")))?;
    Ok(TcpBulkTransport::from_stream(stream))
}
