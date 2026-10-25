use crate::config::DEFAULT_MAX_FRAME_BYTES;
use crate::error::{NexarError, Result};
use crate::transport::BulkTransport;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
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
    /// Why the recv loop stopped. `None` while it runs.
    ended: Option<RecvEnd>,
}

/// Why the recv loop stopped.
#[derive(Clone, Copy)]
enum RecvEnd {
    /// The stream ended or failed.
    Closed,
    /// A frame announced more than the frame limit.
    TooLarge { len: u64, max: usize },
}

impl RecvEnd {
    fn error(self) -> NexarError {
        match self {
            Self::Closed => NexarError::transport("tcp bulk connection closed"),
            Self::TooLarge { len, max } => NexarError::FrameTooLarge { len, max },
        }
    }
}

/// Bytes the recv loop reads and zeroes per step, so a frame's buffer grows
/// only as its data arrives.
const READ_STEP: usize = 4 * 1024 * 1024;

/// Bulk transport over TCP, optionally wrapped in TLS.
///
/// Carries a `[tag: u64 LE][len: u64 LE][payload]` framing so that tagged
/// collectives can bypass QUIC's AES-256-GCM overhead for large tensor data.
/// Tag 0 is used for untagged `send_bulk`/`recv_bulk`.
///
/// When `NexarConfig::encrypt_bulk_transport` is `true` (the default), the
/// underlying TCP stream is wrapped in TLS before framing is applied.
pub struct TcpBulkTransport {
    writer: Mutex<Box<dyn AsyncWrite + Unpin + Send>>,
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
    /// Create a `TcpBulkTransport` from an already-connected `TcpStream` (plaintext).
    pub fn from_stream(stream: TcpStream) -> Self {
        Self::from_stream_with_limit(stream, DEFAULT_MAX_FRAME_BYTES)
    }

    /// Like [`from_stream`](Self::from_stream), with a frame limit of `max_frame_bytes`.
    pub fn from_stream_with_limit(stream: TcpStream, max_frame_bytes: usize) -> Self {
        let (reader, writer) = tokio::io::split(stream);
        Self::from_split_with_limit(Box::new(reader), Box::new(writer), max_frame_bytes)
    }

    /// Create a `TcpBulkTransport` from any async read/write pair.
    ///
    /// This is the generic constructor used by both plaintext and TLS paths.
    pub fn from_split(
        reader: Box<dyn AsyncRead + Unpin + Send>,
        writer: Box<dyn AsyncWrite + Unpin + Send>,
    ) -> Self {
        Self::from_split_with_limit(reader, writer, DEFAULT_MAX_FRAME_BYTES)
    }

    /// Like [`from_split`](Self::from_split), with a frame limit of `max_frame_bytes`.
    ///
    /// A longer incoming frame closes the transport. Receivers then get
    /// [`NexarError::FrameTooLarge`].
    pub fn from_split_with_limit(
        reader: Box<dyn AsyncRead + Unpin + Send>,
        writer: Box<dyn AsyncWrite + Unpin + Send>,
        max_frame_bytes: usize,
    ) -> Self {
        let (untagged_tx, untagged_rx) = mpsc::channel(64);
        let state = Arc::new(Mutex::new(RecvState {
            senders: HashMap::new(),
            pending: HashMap::new(),
            ended: None,
        }));

        let recv_state = Arc::clone(&state);
        let recv_handle = tokio::spawn(async move {
            let end = recv_loop(reader, untagged_tx, &recv_state, max_frame_bytes).await;
            let mut st = recv_state.lock().await;
            st.ended = Some(end);
            // Drop the senders, so receivers wake with the end reason.
            st.senders.clear();
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
            .map_err(|e| NexarError::transport_with_source("tcp bulk write tag", e))?;
        writer
            .write_all(&(data.len() as u64).to_le_bytes())
            .await
            .map_err(|e| NexarError::transport_with_source("tcp bulk write len", e))?;
        writer
            .write_all(data)
            .await
            .map_err(|e| NexarError::transport_with_source("tcp bulk write payload", e))?;
        writer
            .flush()
            .await
            .map_err(|e| NexarError::transport_with_source("tcp bulk flush", e))?;
        Ok(())
    }

    /// Get or create a per-tag receiver. Returns an `Arc<Mutex<Receiver>>` that
    /// can be locked independently of other tags (no shared lock across `.await`).
    /// The error for a receiver whose channel closed.
    async fn closed_error(&self) -> NexarError {
        let ended = self.state.lock().await.ended;
        ended.unwrap_or(RecvEnd::Closed).error()
    }

    async fn get_tag_receiver(&self, tag: u64) -> Arc<Mutex<mpsc::Receiver<Vec<u8>>>> {
        // Hold the map lock for the whole lookup, so two callers for one tag
        // cannot register two channels.
        let mut map = self.tagged_rx.lock().await;
        if let Some(rx) = map.get(&tag) {
            return Arc::clone(rx);
        }
        // Move pending frames and register the sender under one state lock:
        // the recv loop then delivers later frames after them, in order.
        let rx = {
            let mut st = self.state.lock().await;
            let pending = st.pending.remove(&tag).unwrap_or_default();
            let (tx, rx) = mpsc::channel(64 + pending.len());
            for data in pending {
                // The channel has room for every pending frame.
                let _ = tx.try_send(data);
            }
            // After the recv loop ended, no frame follows: drop the sender.
            if st.ended.is_none() {
                st.senders.insert(tag, tx);
            }
            rx
        };
        let rx_arc = Arc::new(Mutex::new(rx));
        map.insert(tag, Arc::clone(&rx_arc));
        rx_arc
    }
}

impl BulkTransport for TcpBulkTransport {
    fn send_bulk<'a>(&'a self, data: &'a [u8]) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.write_frame(0, data))
    }

    /// Flush the stream, send TLS `close_notify` when encrypted, and send FIN.
    /// The receiving side stays open.
    fn shutdown<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.writer
                .lock()
                .await
                .shutdown()
                .await
                .map_err(|e| NexarError::transport_with_source("tcp bulk shutdown", e))
        })
    }

    fn recv_bulk<'a>(&'a self, _expected_size: usize) -> BoxFuture<'a, Result<Vec<u8>>> {
        Box::pin(async move {
            let frame = self.untagged_rx.lock().await.recv().await;
            match frame {
                Some(data) => Ok(data),
                None => Err(self.closed_error().await),
            }
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
            let frame = rx_arc.lock().await.recv().await;
            match frame {
                Some(data) => Ok(data),
                None => Err(self.closed_error().await),
            }
        })
    }
}

/// Read `len` payload bytes, growing the buffer by [`READ_STEP`] as data arrives.
async fn read_payload(
    reader: &mut (dyn AsyncRead + Unpin + Send),
    len: usize,
) -> std::io::Result<Vec<u8>> {
    let mut payload = Vec::new();
    while payload.len() < len {
        let start = payload.len();
        payload.resize(start + (len - start).min(READ_STEP), 0);
        reader.read_exact(&mut payload[start..]).await?;
    }
    Ok(payload)
}

/// Background loop: read frames and route to the appropriate channel.
///
/// Returns why it stopped. A frame longer than `max_frame_bytes` stops it
/// before any payload is read.
async fn recv_loop(
    mut reader: Box<dyn AsyncRead + Unpin + Send>,
    untagged_tx: mpsc::Sender<Vec<u8>>,
    state: &Mutex<RecvState>,
    max_frame_bytes: usize,
) -> RecvEnd {
    let mut tag_buf = [0u8; 8];
    let mut len_buf = [0u8; 8];
    loop {
        if let Err(e) = reader.read_exact(&mut tag_buf).await {
            tracing::debug!("tcp bulk recv loop ended: {e}");
            return RecvEnd::Closed;
        }
        if let Err(e) = reader.read_exact(&mut len_buf).await {
            tracing::debug!("tcp bulk recv loop ended reading len: {e}");
            return RecvEnd::Closed;
        }
        let tag = u64::from_le_bytes(tag_buf);
        let announced = u64::from_le_bytes(len_buf);

        let len = match usize::try_from(announced) {
            Ok(len) if len <= max_frame_bytes => len,
            _ => {
                tracing::warn!(
                    len = announced,
                    max = max_frame_bytes,
                    "tcp bulk: frame too large, closing connection"
                );
                return RecvEnd::TooLarge {
                    len: announced,
                    max: max_frame_bytes,
                };
            }
        };

        let payload = match read_payload(&mut *reader, len).await {
            Ok(payload) => payload,
            Err(e) => {
                tracing::debug!("tcp bulk recv loop ended reading payload: {e}");
                return RecvEnd::Closed;
            }
        };

        if tag == 0 {
            if untagged_tx.send(payload).await.is_err() {
                return RecvEnd::Closed;
            }
        } else {
            // Decide between channel and pending buffer under one lock, so a
            // receiver that registers meanwhile cannot miss this frame. The
            // channel send runs after the lock is released.
            let tx = {
                let mut st = state.lock().await;
                match st.senders.get(&tag) {
                    Some(tx) => tx.clone(),
                    None => {
                        st.pending.entry(tag).or_default().push(payload);
                        continue;
                    }
                }
            };
            if tx.send(payload).await.is_err() {
                return RecvEnd::Closed;
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
        .map_err(|e| NexarError::transport_with_source("tcp bulk listen", e))?;
    let local = listener
        .local_addr()
        .map_err(|e| NexarError::transport_with_source("tcp bulk local_addr", e))?;
    Ok((listener, local))
}

/// Connect to a peer's TCP bulk listener and create both ends.
pub async fn tcp_bulk_connect(
    addr: std::net::SocketAddr,
    max_frame_bytes: usize,
) -> Result<TcpBulkTransport> {
    let stream = TcpStream::connect(addr)
        .await
        .map_err(|e| NexarError::transport_with_source("tcp bulk connect", e))?;
    stream
        .set_nodelay(true)
        .map_err(|e| NexarError::transport_with_source("tcp bulk set_nodelay", e))?;
    Ok(TcpBulkTransport::from_stream_with_limit(
        stream,
        max_frame_bytes,
    ))
}

/// Accept one connection from a TCP bulk listener.
pub async fn tcp_bulk_accept(listener: &TcpListener) -> Result<TcpBulkTransport> {
    let (stream, _addr) = listener
        .accept()
        .await
        .map_err(|e| NexarError::transport_with_source("tcp bulk accept", e))?;
    tcp_bulk_accept_stream(stream, None, DEFAULT_MAX_FRAME_BYTES).await
}

/// Build the server end of a sidecar from a stream a `TcpListener` already accepted.
///
/// `tls: Some` runs the TLS handshake first. Callers that accept in a loop use
/// this so that one slow handshake does not block the next `accept`.
pub async fn tcp_bulk_accept_stream(
    stream: TcpStream,
    tls: Option<Arc<rustls::ServerConfig>>,
    max_frame_bytes: usize,
) -> Result<TcpBulkTransport> {
    stream
        .set_nodelay(true)
        .map_err(|e| NexarError::transport_with_source("tcp bulk set_nodelay", e))?;
    let Some(tls_config) = tls else {
        return Ok(TcpBulkTransport::from_stream_with_limit(
            stream,
            max_frame_bytes,
        ));
    };
    let acceptor = tokio_rustls::TlsAcceptor::from(tls_config);
    let tls_stream = acceptor
        .accept(stream)
        .await
        .map_err(|e| NexarError::transport_with_source("tcp bulk tls handshake (server)", e))?;
    let (reader, writer) = tokio::io::split(tls_stream);
    Ok(TcpBulkTransport::from_split_with_limit(
        Box::new(reader),
        Box::new(writer),
        max_frame_bytes,
    ))
}

/// Connect to a peer's TCP bulk listener with TLS and create the transport.
pub async fn tcp_bulk_connect_tls(
    addr: std::net::SocketAddr,
    tls_config: Arc<rustls::ClientConfig>,
    max_frame_bytes: usize,
) -> Result<TcpBulkTransport> {
    let stream = TcpStream::connect(addr)
        .await
        .map_err(|e| NexarError::transport_with_source("tcp bulk tls connect", e))?;
    stream
        .set_nodelay(true)
        .map_err(|e| NexarError::transport_with_source("tcp bulk tls set_nodelay", e))?;

    let connector = tokio_rustls::TlsConnector::from(tls_config);
    let server_name = rustls::pki_types::ServerName::try_from("localhost")
        .map_err(|e| NexarError::Tls(format!("bulk TLS server name: {e}")))?;
    let tls_stream = connector
        .connect(server_name, stream)
        .await
        .map_err(|e| NexarError::transport_with_source("tcp bulk tls handshake (client)", e))?;

    let (reader, writer) = tokio::io::split(tls_stream);
    Ok(TcpBulkTransport::from_split_with_limit(
        Box::new(reader),
        Box::new(writer),
        max_frame_bytes,
    ))
}

/// Accept one TLS connection from a TCP bulk listener.
pub async fn tcp_bulk_accept_tls(
    listener: &TcpListener,
    tls_config: Arc<rustls::ServerConfig>,
) -> Result<TcpBulkTransport> {
    let (stream, _addr) = listener
        .accept()
        .await
        .map_err(|e| NexarError::transport_with_source("tcp bulk tls accept", e))?;
    tcp_bulk_accept_stream(stream, Some(tls_config), DEFAULT_MAX_FRAME_BYTES).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A connected loopback pair; the receiving side (second) uses `limit`.
    async fn pair(limit: usize) -> (TcpBulkTransport, TcpBulkTransport) {
        let (listener, addr) = tcp_bulk_listen("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let (client, server) = tokio::join!(
            tcp_bulk_connect(addr, DEFAULT_MAX_FRAME_BYTES),
            listener.accept()
        );
        let (stream, _) = server.unwrap();
        (
            client.unwrap(),
            TcpBulkTransport::from_stream_with_limit(stream, limit),
        )
    }

    #[tokio::test]
    async fn test_oversize_frame_is_rejected_with_typed_error() {
        let (tx, rx) = pair(1024).await;
        tx.send_bulk_tagged(5, &[1u8; 2048]).await.unwrap();
        let err = rx.recv_bulk_tagged(5, 2048).await.unwrap_err();
        assert!(
            matches!(
                err,
                NexarError::FrameTooLarge {
                    len: 2048,
                    max: 1024
                }
            ),
            "{err}"
        );
        // Receivers that register after the loop ended get the error too.
        let err = rx.recv_bulk_tagged(6, 0).await.unwrap_err();
        assert!(matches!(err, NexarError::FrameTooLarge { .. }), "{err}");
        let err = rx.recv_bulk(0).await.unwrap_err();
        assert!(matches!(err, NexarError::FrameTooLarge { .. }), "{err}");
    }

    #[tokio::test]
    async fn test_huge_announced_length_allocates_nothing() {
        let (listener, addr) = tcp_bulk_listen("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let mut raw = TcpStream::connect(addr).await.unwrap();
        let (stream, _) = listener.accept().await.unwrap();
        let rx = TcpBulkTransport::from_stream(stream);
        // Announce u64::MAX bytes, send none. The default limit rejects it.
        raw.write_all(&7u64.to_le_bytes()).await.unwrap();
        raw.write_all(&u64::MAX.to_le_bytes()).await.unwrap();
        let err = rx.recv_bulk_tagged(7, 0).await.unwrap_err();
        assert!(
            matches!(err, NexarError::FrameTooLarge { len: u64::MAX, .. }),
            "{err}"
        );
    }

    #[tokio::test]
    async fn test_large_frame_within_limit_passes() {
        // Spans several read steps and ends on a partial step.
        let len = 2 * READ_STEP + 12_345;
        let (tx, rx) = pair(len).await;
        let payload: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let (sent, got) = tokio::join!(
            tx.send_bulk_tagged(9, &payload),
            rx.recv_bulk_tagged(9, len)
        );
        sent.unwrap();
        assert_eq!(got.unwrap(), payload);
    }

    #[tokio::test]
    async fn test_closed_connection_wakes_receivers() {
        let (tx, rx) = pair(1024).await;
        tx.shutdown().await.unwrap();
        let err = rx.recv_bulk_tagged(3, 0).await.unwrap_err();
        assert!(matches!(err, NexarError::Transport { .. }), "{err}");
    }
}
