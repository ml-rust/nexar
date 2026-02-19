use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::encode_message;
use crate::types::{Priority, Rank};

/// Stream type tag: first byte on every QUIC uni stream.
/// Allows the router to dispatch streams to the correct channel without ambiguity.
pub(crate) const STREAM_TAG_FRAMED: u8 = 0x01;
pub(crate) const STREAM_TAG_RAW: u8 = 0x02;

/// A connection to a single peer node, wrapping a QUIC connection.
///
/// Handles the **send side** of communication. All receiving is done by
/// `PeerRouter`, which runs a single `accept_uni` loop per peer and
/// demultiplexes incoming streams into typed channels.
///
/// Every outbound stream begins with a 1-byte stream type tag
/// (`STREAM_TAG_FRAMED` or `STREAM_TAG_RAW`) so the remote router can
/// dispatch it correctly.
pub struct PeerConnection {
    pub rank: Rank,
    pub(crate) conn: quinn::Connection,
}

impl PeerConnection {
    /// Create a `PeerConnection` from an established QUIC connection.
    pub fn new(rank: Rank, conn: quinn::Connection) -> Self {
        Self { rank, conn }
    }

    /// Send a control message as a framed uni stream.
    pub async fn send_message(&self, msg: &NexarMessage, priority: Priority) -> Result<()> {
        let buf = encode_message(msg, priority)?;
        self.send_tagged(STREAM_TAG_FRAMED, &buf).await
    }

    /// Send raw bytes on a new unidirectional stream (for bulk tensor data).
    pub async fn send_raw(&self, data: &[u8]) -> Result<()> {
        self.send_tagged(STREAM_TAG_RAW, data).await
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
