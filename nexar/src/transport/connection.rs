use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::encode_message;
use crate::types::{Priority, Rank};

/// Stream type tag: first byte on every QUIC uni stream.
/// Allows the router to dispatch streams to the correct channel without ambiguity.
pub(crate) const STREAM_TAG_FRAMED: u8 = 0x01;
pub(crate) const STREAM_TAG_RAW: u8 = 0x02;
/// Raw stream with a communicator ID prefix (for split communicators).
pub(crate) const STREAM_TAG_RAW_COMM: u8 = 0x03;

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
    /// Opaque extension slot for transport accelerators (RDMA, GPUDirect).
    /// External crates attach typed state via `add_extension` / `extension`.
    extensions: std::sync::RwLock<Vec<Box<dyn std::any::Any + Send + Sync>>>,
}

impl PeerConnection {
    /// Create a `PeerConnection` from an established QUIC connection.
    pub fn new(rank: Rank, conn: quinn::Connection) -> Self {
        Self {
            rank,
            conn,
            extensions: std::sync::RwLock::new(Vec::new()),
        }
    }

    /// Attach an extension object (e.g. RDMA state) to this connection.
    pub fn add_extension<T: std::any::Any + Send + Sync + 'static>(&self, ext: T) {
        let mut exts = self.extensions.write().expect("extensions lock poisoned");
        exts.push(Box::new(ext));
    }

    /// Retrieve a reference to an extension by type.
    ///
    /// Returns `None` if no extension of that type has been attached.
    pub fn extension<T: std::any::Any + Send + Sync + 'static>(
        &self,
    ) -> Option<impl std::ops::Deref<Target = T> + '_> {
        let exts = self.extensions.read().expect("extensions lock poisoned");
        let idx = exts.iter().position(|e| e.downcast_ref::<T>().is_some())?;
        Some(ExtensionRef {
            guard: exts,
            idx,
            _marker: std::marker::PhantomData,
        })
    }

    /// Send a control message as a framed uni stream (always QUIC).
    pub async fn send_message(&self, msg: &NexarMessage, priority: Priority) -> Result<()> {
        let buf = encode_message(msg, priority)?;
        self.send_tagged(STREAM_TAG_FRAMED, &buf).await
    }

    /// Send raw bytes on a new unidirectional stream (for bulk tensor data).
    pub async fn send_raw(&self, data: &[u8]) -> Result<()> {
        self.send_tagged(STREAM_TAG_RAW, data).await
    }

    /// Send raw bytes tagged with a communicator ID (for split communicators).
    /// Always uses QUIC (split comms are a logical overlay).
    pub async fn send_raw_comm(&self, comm_id: u32, data: &[u8]) -> Result<()> {
        let mut stream = self
            .conn
            .open_uni()
            .await
            .map_err(|e| NexarError::Transport(format!("open uni stream: {e}")))?;
        stream
            .write_all(&[STREAM_TAG_RAW_COMM])
            .await
            .map_err(|e| NexarError::Transport(format!("write stream tag: {e}")))?;
        stream
            .write_all(&comm_id.to_le_bytes())
            .await
            .map_err(|e| NexarError::Transport(format!("write comm_id: {e}")))?;
        stream
            .write_all(&(data.len() as u64).to_le_bytes())
            .await
            .map_err(|e| NexarError::Transport(format!("write length: {e}")))?;
        stream
            .write_all(data)
            .await
            .map_err(|e| NexarError::Transport(format!("write payload: {e}")))?;
        stream
            .finish()
            .map_err(|e| NexarError::Transport(format!("finish stream: {e}")))?;
        Ok(())
    }

    /// Get the remote address of this connection.
    pub fn remote_addr(&self) -> std::net::SocketAddr {
        self.conn.remote_address()
    }

    /// Open a uni stream, write the stream type tag + length-prefixed payload,
    /// then finish. Shared by both framed and raw sends.
    async fn send_tagged(&self, tag: u8, data: &[u8]) -> Result<()> {
        let mut stream = self
            .conn
            .open_uni()
            .await
            .map_err(|e| NexarError::Transport(format!("open uni stream: {e}")))?;
        stream
            .write_all(&[tag])
            .await
            .map_err(|e| NexarError::Transport(format!("write stream tag: {e}")))?;
        stream
            .write_all(&(data.len() as u64).to_le_bytes())
            .await
            .map_err(|e| NexarError::Transport(format!("write length: {e}")))?;
        stream
            .write_all(data)
            .await
            .map_err(|e| NexarError::Transport(format!("write payload: {e}")))?;
        stream
            .finish()
            .map_err(|e| NexarError::Transport(format!("finish stream: {e}")))?;
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
        self.guard[self.idx]
            .downcast_ref::<T>()
            .expect("extension type mismatch")
    }
}
