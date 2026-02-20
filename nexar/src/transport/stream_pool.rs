//! Pre-opened QUIC unidirectional stream pool.
//!
//! QUIC uni streams are one-shot (write, finish, done), so they can't be
//! reused. However, `open_uni()` involves a round-trip that adds latency.
//! This pool pre-opens streams so they're ready when needed, reducing send
//! latency for collective operations.

use crate::error::{NexarError, Result};
use tokio::sync::Mutex;

/// Pool of pre-opened QUIC unidirectional send streams.
///
/// Maintains up to `max_ready` streams ready for immediate use. Streams are
/// checked out one at a time. Call `refill()` to pre-open streams.
pub(crate) struct StreamPool {
    conn: quinn::Connection,
    ready: Mutex<Vec<quinn::SendStream>>,
    max_ready: usize,
}

impl StreamPool {
    /// Create a new stream pool.
    pub fn new(conn: quinn::Connection, max_ready: usize) -> Self {
        Self {
            conn,
            ready: Mutex::new(Vec::with_capacity(max_ready)),
            max_ready,
        }
    }

    /// Pre-open streams to fill the pool up to `max_ready`.
    pub async fn refill(&self) -> Result<()> {
        let mut ready = self.ready.lock().await;
        while ready.len() < self.max_ready {
            let stream = self
                .conn
                .open_uni()
                .await
                .map_err(|e| NexarError::transport_with_source("open uni stream", e))?;
            ready.push(stream);
        }
        Ok(())
    }

    /// Get a pre-opened stream, or open a new one if the pool is empty.
    pub async fn checkout(&self) -> Result<quinn::SendStream> {
        {
            let mut ready = self.ready.lock().await;
            if let Some(stream) = ready.pop() {
                return Ok(stream);
            }
        }
        // Pool empty — open on demand.
        self.conn
            .open_uni()
            .await
            .map_err(|e| NexarError::transport_with_source("open uni stream", e))
    }
}
