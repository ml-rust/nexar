use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::encode_message;
use crate::types::{Priority, Rank};
use futures::future::BoxFuture;

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
}

/// Stream type tag: first byte on every QUIC uni stream.
/// Allows the router to dispatch streams to the correct channel without ambiguity.
pub(crate) const STREAM_TAG_FRAMED: u8 = 0x01;
pub(crate) const STREAM_TAG_RAW: u8 = 0x02;
/// Raw stream with a communicator ID prefix (for split communicators).
pub(crate) const STREAM_TAG_RAW_COMM: u8 = 0x03;
/// Raw stream with a u64 tag prefix (for concurrent collectives / tagged transfers).
pub(crate) const STREAM_TAG_RAW_TAGGED: u8 = 0x04;

/// A connection to a single peer node, wrapping a QUIC connection.
///
/// Handles the **send side** of communication. All receiving is done by
/// `PeerRouter`, which runs a single `accept_uni` loop per peer and
/// demultiplexes incoming streams into typed channels.
///
/// Every outbound stream begins with a 1-byte stream type tag
/// (`STREAM_TAG_FRAMED`, `STREAM_TAG_RAW`, or `STREAM_TAG_RAW_COMM`) so
/// the remote router can dispatch it correctly.
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
        self.send_framed(STREAM_TAG_FRAMED, &[], &buf).await
    }

    /// Send raw bytes on a new unidirectional stream (for bulk tensor data).
    pub async fn send_raw(&self, data: &[u8]) -> Result<()> {
        self.send_framed(STREAM_TAG_RAW, &[], data).await
    }

    /// Send raw bytes tagged with a communicator ID (for split communicators).
    /// Always uses QUIC (split comms are a logical overlay).
    pub async fn send_raw_comm(&self, comm_id: u64, data: &[u8]) -> Result<()> {
        self.send_framed(STREAM_TAG_RAW_COMM, &comm_id.to_le_bytes(), data)
            .await
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

    /// Send raw bytes with a u64 tag on a new unidirectional stream.
    ///
    /// Wire format: `[0x04][tag: u64 LE][len: u64 LE][payload]`.
    /// Used by concurrent collectives to avoid cross-talk on the raw lane.
    pub async fn send_raw_tagged(&self, tag: u64, data: &[u8]) -> Result<()> {
        self.send_framed(STREAM_TAG_RAW_TAGGED, &tag.to_le_bytes(), data)
            .await
    }

    /// Send tagged bytes using the best available transport.
    ///
    /// Tries `TaggedBulkTransport` first (e.g., TCP sidecar which carries tags
    /// natively), then falls back to QUIC tagged send. Plain `BulkTransport`
    /// (RDMA) is NOT used here because it doesn't carry tags.
    pub async fn send_raw_tagged_best_effort(&self, tag: u64, data: &[u8]) -> Result<()> {
        let tagged_bulk: Option<std::sync::Arc<dyn super::TaggedBulkTransport>> = self
            .extension::<std::sync::Arc<dyn super::TaggedBulkTransport>>()?
            .map(|b| std::sync::Arc::clone(&*b));
        if let Some(bulk) = tagged_bulk {
            match bulk.send_bulk_tagged(tag, data).await {
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
        self.send_raw_tagged(tag, data).await
    }

    /// Get the remote address of this connection.
    pub fn remote_addr(&self) -> std::net::SocketAddr {
        self.conn.remote_address()
    }

    /// Open a uni stream, write `[stream_tag][prefix][len][payload]`, then finish.
    ///
    /// All QUIC send methods delegate here. The `prefix` carries per-format
    /// metadata (comm_id for split comms, u64 tag for tagged sends, empty for
    /// plain framed/raw sends).
    ///
    /// Uses the stream pool to avoid `open_uni()` latency when pre-opened
    /// streams are available.
    async fn send_framed(&self, stream_tag: u8, prefix: &[u8], data: &[u8]) -> Result<()> {
        let mut stream = self.stream_pool.checkout().await?;
        stream
            .write_all(&[stream_tag])
            .await
            .map_err(|e| NexarError::transport_with_source("write stream tag", e))?;
        if !prefix.is_empty() {
            stream
                .write_all(prefix)
                .await
                .map_err(|e| NexarError::transport_with_source("write prefix", e))?;
        }
        stream
            .write_all(&(data.len() as u64).to_le_bytes())
            .await
            .map_err(|e| NexarError::transport_with_source("write length", e))?;
        stream
            .write_all(data)
            .await
            .map_err(|e| NexarError::transport_with_source("write payload", e))?;
        stream
            .finish()
            .map_err(|e| NexarError::transport_with_source("finish stream", e))?;
        Ok(())
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
