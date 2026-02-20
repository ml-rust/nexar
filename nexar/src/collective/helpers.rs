use crate::client::NexarClient;
use crate::error::{NexarError, Result};
use crate::transport::buffer_pool::PooledBuf;
use crate::types::Rank;
use std::time::Duration;

/// Integer ceiling of log2(n). Returns 0 for n <= 1.
pub(crate) fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    // For n > 1: ceil(log2(n)) = 32 - (n-1).leading_zeros()
    u32::BITS - (n - 1).leading_zeros()
}

/// Chunk partition layout for ring-based collectives.
///
/// Divides `count` elements among `world` ranks, distributing remainder
/// elements to the first `remainder` ranks (one extra element each).
pub(crate) struct ChunkLayout {
    pub offsets: Vec<usize>,
    pub base_chunk: usize,
    pub remainder: usize,
}

impl ChunkLayout {
    pub fn new(count: usize, world: usize) -> Self {
        let base_chunk = count / world;
        let remainder = count % world;

        let offsets: Vec<usize> = (0..world)
            .scan(0usize, |acc, i| {
                let off = *acc;
                *acc += if i < remainder {
                    base_chunk + 1
                } else {
                    base_chunk
                };
                Some(off)
            })
            .collect();

        Self {
            offsets,
            base_chunk,
            remainder,
        }
    }

    pub fn chunk_count(&self, i: usize) -> usize {
        if i < self.remainder {
            self.base_chunk + 1
        } else {
            self.base_chunk
        }
    }
}

/// Get the collective timeout from the client's config.
pub(crate) fn collective_timeout(client: &NexarClient) -> Duration {
    client.config.collective_timeout
}

/// Tag context for collective operations: `None` for untagged (default lane),
/// `Some(tag)` for tagged transport (concurrent collectives).
pub(crate) type CollectiveTag = Option<u64>;

/// Send bytes to a peer with optional tag and timeout.
pub(crate) async fn collective_send_with_tag(
    client: &NexarClient,
    dest: Rank,
    data: &[u8],
    operation: &'static str,
    tag: CollectiveTag,
) -> Result<()> {
    let timeout = collective_timeout(client);
    let result = match tag {
        Some(t) => {
            tokio::time::timeout(timeout, client.send_bytes_tagged_best_effort(dest, t, data)).await
        }
        None => tokio::time::timeout(timeout, client.send_bytes_best_effort(dest, data)).await,
    };
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(NexarError::CollectiveFailed {
            operation,
            rank: dest,
            reason: e.to_string(),
        }),
        Err(_) => Err(NexarError::CollectiveFailed {
            operation,
            rank: dest,
            reason: format!("send timed out after {}s", timeout.as_secs()),
        }),
    }
}

/// Receive bytes from a peer with optional tag and timeout.
pub(crate) async fn collective_recv_with_tag(
    client: &NexarClient,
    src: Rank,
    operation: &'static str,
    tag: CollectiveTag,
) -> Result<PooledBuf> {
    let timeout = collective_timeout(client);
    let result = match tag {
        Some(t) => {
            tokio::time::timeout(timeout, client.recv_bytes_tagged_best_effort(src, t, 0)).await
        }
        None => tokio::time::timeout(timeout, client.recv_bytes_best_effort(src, 0)).await,
    };
    match result {
        Ok(Ok(buf)) => Ok(buf),
        Ok(Err(e)) => Err(NexarError::CollectiveFailed {
            operation,
            rank: src,
            reason: e.to_string(),
        }),
        Err(_) => Err(NexarError::CollectiveFailed {
            operation,
            rank: src,
            reason: format!("recv timed out after {}s", timeout.as_secs()),
        }),
    }
}
