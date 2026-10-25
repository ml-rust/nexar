use super::lane::{LaneKey, STREAM_TAG_FRAMED, SeqHeader};
use super::seq_send::SeqSend;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::encode_message;
use crate::types::{Priority, Rank};
use futures::future::BoxFuture;
use std::sync::atomic::{AtomicU64, Ordering};

/// Trait for bulk transport accelerators (e.g. RDMA).
///
/// Implementations are attached to `PeerConnection` via the extension slot
/// and used by `send_raw_best_effort` to accelerate data transfers.
/// Recv stays QUIC-only for now (RDMA recv needs pre-posted buffers).
pub trait BulkTransport: Send + Sync + 'static {
    /// Send raw bytes via the accelerated transport.
    fn send_bulk<'a>(&'a self, data: &'a [u8]) -> BoxFuture<'a, Result<()>>;

    /// Receive raw bytes via the accelerated transport.
    ///
    /// Default: not supported (falls back to QUIC in the caller).
    fn recv_bulk<'a>(&'a self, _expected_size: usize) -> BoxFuture<'a, Result<Vec<u8>>> {
        Box::pin(async move {
            Err(NexarError::transport(
                "recv_bulk not supported by this transport",
            ))
        })
    }

    /// Flush pending sends and close the sending side cleanly.
    ///
    /// [`NexarClient::shutdown`](crate::NexarClient::shutdown) calls it once per
    /// peer. Default: nothing to flush.
    fn shutdown<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// A connection to a single peer node, wrapping a QUIC connection.
///
/// Handles the **send side** of communication. All receiving is done by
/// `PeerRouter`, which runs a single `accept_uni` loop per peer and
/// demultiplexes incoming streams into typed channels.
///
/// Every outbound stream begins with a 1-byte stream type tag, so the remote
/// router can dispatch it. Byte messages (raw, tagged, point-to-point data)
/// carry a sequence number from one counter per connection. The remote
/// router uses it to deliver each [`LaneKey`] in send order.
///
/// Transport accelerators (RDMA, GPUDirect) can attach state via the
/// opaque `extensions` slot and provide extension traits for accelerated
/// send/recv paths.
pub struct PeerConnection {
    pub rank: Rank,
    pub(crate) conn: quinn::Connection,
    /// Pre-opened stream pool for reducing `open_uni()` latency.
    stream_pool: super::stream_pool::StreamPool,
    /// Opaque extension slot for transport accelerators (RDMA, GPUDirect).
    /// External crates attach typed state via `add_extension` / `extension`.
    extensions: std::sync::RwLock<Vec<Box<dyn std::any::Any + Send + Sync>>>,
    /// Next sequence number of a byte message on this connection.
    next_seq: AtomicU64,
}

/// Maximum number of pre-opened streams per peer.
const STREAM_POOL_MAX_READY: usize = 8;

impl PeerConnection {
    /// Create a `PeerConnection` from an established QUIC connection.
    pub fn new(rank: Rank, conn: quinn::Connection) -> Self {
        let stream_pool = super::stream_pool::StreamPool::new(conn.clone(), STREAM_POOL_MAX_READY);
        Self {
            rank,
            conn,
            stream_pool,
            extensions: std::sync::RwLock::new(Vec::new()),
            next_seq: AtomicU64::new(0),
        }
    }

    /// Pre-open streams in the pool to reduce first-send latency.
    ///
    /// Should be called once after connection establishment. Non-fatal: if
    /// pre-opening fails (e.g. connection not yet ready), streams will be
    /// opened on-demand at send time.
    pub async fn warm_stream_pool(&self) {
        let _ = self.stream_pool.refill().await;
    }

    /// Attach an extension object (e.g. RDMA state) to this connection.
    pub fn add_extension<T: std::any::Any + Send + Sync + 'static>(
        &self,
        ext: T,
    ) -> crate::error::Result<()> {
        let mut exts = self
            .extensions
            .write()
            .map_err(|_| NexarError::LockPoisoned("extensions"))?;
        exts.push(Box::new(ext));
        Ok(())
    }

    /// Retrieve a reference to an extension by type.
    ///
    /// Returns `None` if no extension of that type has been attached.
    /// Returns `Err` if the lock is poisoned.
    pub fn extension<T: std::any::Any + Send + Sync + 'static>(
        &self,
    ) -> crate::error::Result<Option<impl std::ops::Deref<Target = T> + '_>> {
        let exts = self
            .extensions
            .read()
            .map_err(|_| NexarError::LockPoisoned("extensions"))?;
        let idx = match exts.iter().position(|e| e.downcast_ref::<T>().is_some()) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        Ok(Some(ExtensionRef {
            guard: exts,
            idx,
            _marker: std::marker::PhantomData,
        }))
    }

    /// Send a control message as a framed uni stream (always QUIC).
    pub async fn send_message(&self, msg: &NexarMessage, priority: Priority) -> Result<()> {
        let buf = encode_message(msg, priority)?;
        let mut stream = self.stream_pool.checkout().await?;
        let mut head = [0u8; 9];
        head[0] = STREAM_TAG_FRAMED;
        head[1..].copy_from_slice(&(buf.len() as u64).to_le_bytes());
        for (bytes, what) in [(&head[..], "write frame header"), (&buf[..], "write frame")] {
            stream
                .write_all(bytes)
                .await
                .map_err(|e| NexarError::transport_with_source(what, e))?;
        }
        stream
            .finish()
            .map_err(|e| NexarError::transport_with_source("finish stream", e))
    }

    /// Send raw bytes on the untagged lane of the root communicator.
    pub async fn send_raw(&self, data: &[u8]) -> Result<()> {
        self.send_lane(LaneKey::Raw { comm_id: 0 }, data).await
    }

    /// Send raw bytes on the untagged lane of communicator `comm_id`.
    /// Always uses QUIC (split comms are a logical overlay).
    pub async fn send_raw_comm(&self, comm_id: u64, data: &[u8]) -> Result<()> {
        self.send_lane(LaneKey::Raw { comm_id }, data).await
    }

    /// Send raw bytes using the best available transport.
    ///
    /// Tries any attached `BulkTransport` accelerator first (e.g. RDMA),
    /// falling back to QUIC if none is available or if the accelerated send fails.
    pub async fn send_raw_best_effort(&self, data: &[u8]) -> Result<()> {
        // Extract the BulkTransport Arc and drop the extension guard before .await.
        let bulk: Option<std::sync::Arc<dyn BulkTransport>> = self
            .extension::<std::sync::Arc<dyn BulkTransport>>()?
            .map(|b| std::sync::Arc::clone(&*b));
        if let Some(bulk) = bulk {
            match bulk.send_bulk(data).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!(
                        peer = self.rank,
                        bytes = data.len(),
                        error = %e,
                        "bulk transport send failed, falling back to QUIC"
                    );
                }
            }
        }
        self.send_raw(data).await
    }

    /// Send raw bytes on tag `tag` of the root communicator.
    pub async fn send_raw_tagged(&self, tag: u64, data: &[u8]) -> Result<()> {
        self.send_lane(LaneKey::Tagged { comm_id: 0, tag }, data)
            .await
    }

    /// Send tagged bytes of the root communicator using the best available
    /// transport. See [`send_lane_best_effort`](Self::send_lane_best_effort).
    pub async fn send_raw_tagged_best_effort(&self, tag: u64, data: &[u8]) -> Result<()> {
        self.send_lane_best_effort(LaneKey::Tagged { comm_id: 0, tag }, data)
            .await
    }

    /// Send `data` on `lane` over QUIC, as the next message of this connection.
    ///
    /// Wire format: `[0x05][kind: u8][comm_id: u64][tag: u64][seq: u64][len: u64][payload]`,
    /// little-endian. The remote router delivers the messages of one lane in
    /// the order their sends took a sequence number. A send that fails or is
    /// cancelled sends a skip notice, so later messages do not wait for it.
    pub async fn send_lane(&self, lane: LaneKey, data: &[u8]) -> Result<()> {
        let stream = self.stream_pool.checkout().await?;
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let mut send = SeqSend::new(stream, seq, self.conn.clone());
        let header = SeqHeader::encode(lane, seq, data.len() as u64);
        send.write(&header, "write lane header").await?;
        send.write(data, "write payload").await?;
        send.finish()
    }

    /// Send on `lane` using the best available transport.
    ///
    /// A tagged lane tries `TaggedBulkTransport` first (e.g. the TCP sidecar,
    /// which carries tags natively, see [`LaneKey::sidecar_tag`]), then falls
    /// back to QUIC. Plain `BulkTransport` (RDMA) is NOT used here because it
    /// doesn't carry tags. Other lanes always use QUIC.
    pub async fn send_lane_best_effort(&self, lane: LaneKey, data: &[u8]) -> Result<()> {
        if let LaneKey::Tagged { comm_id, tag } = lane {
            let tagged_bulk: Option<std::sync::Arc<dyn super::TaggedBulkTransport>> = self
                .extension::<std::sync::Arc<dyn super::TaggedBulkTransport>>()?
                .map(|b| std::sync::Arc::clone(&*b));
            if let Some(bulk) = tagged_bulk {
                let wire_tag = LaneKey::sidecar_tag(comm_id, tag);
                match bulk.send_bulk_tagged(wire_tag, data).await {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        tracing::warn!(
                            peer = self.rank,
                            tag,
                            bytes = data.len(),
                            error = %e,
                            "tagged bulk transport send failed, falling back to QUIC"
                        );
                    }
                }
            }
        }
        self.send_lane(lane, data).await
    }

    /// Get the remote address of this connection.
    pub fn remote_addr(&self) -> std::net::SocketAddr {
        self.conn.remote_address()
    }
}

/// RAII guard that holds a read lock on the extensions vec and derefs to the
/// extension at the given index.
struct ExtensionRef<'a, T> {
    guard: std::sync::RwLockReadGuard<'a, Vec<Box<dyn std::any::Any + Send + Sync>>>,
    idx: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: std::any::Any> std::ops::Deref for ExtensionRef<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: `extension()` only constructs an `ExtensionRef` after verifying
        // that `self.guard[self.idx].downcast_ref::<T>()` succeeds, so the type
        // is guaranteed to match here.
        self.guard[self.idx]
            .downcast_ref::<T>()
            .expect("extension type mismatch: index was validated at construction")
    }
}
