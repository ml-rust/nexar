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

/// Tag context for collective operations.
///
/// All collectives require a tag (`Some(tag)`) for message isolation via
/// tagged transport. `None` falls back to the untagged raw lane (only used
/// internally by barrier).
pub(crate) type CollectiveTag = Option<u64>;

/// Offset a collective tag by a step number to produce a unique per-round tag.
///
/// Essential for sparse topologies where relay messages use separate QUIC
/// streams and can arrive out of order. Each round of a multi-step collective
/// must use a distinct tag to prevent message confusion.
pub(crate) fn step_tag(base: CollectiveTag, step: usize) -> CollectiveTag {
    base.map(|t| t.wrapping_mul(1000).wrapping_add(step as u64))
}

/// Send bytes to a peer with tag-based routing and timeout.
///
/// For sparse topologies, automatically relays through intermediate hops
/// when `dest` is not a direct neighbor.
pub(crate) async fn collective_send(
    client: &NexarClient,
    dest: Rank,
    data: &[u8],
    operation: &'static str,
    tag: CollectiveTag,
) -> Result<()> {
    let timeout = collective_timeout(client);

    // Sparse topology relay path: if dest is not a direct peer, relay.
    if !client.has_direct_peer(dest)
        && let (Some(rt), Some(t)) = (&client.routing_table, tag)
    {
        let result = tokio::time::timeout(
            timeout,
            crate::transport::relay::send_or_relay_tagged(
                client.rank(),
                &client.peers,
                rt,
                dest,
                t,
                data,
            ),
        )
        .await;
        return match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(NexarError::CollectiveFailed {
                operation,
                rank: dest,
                reason: e.to_string(),
            }),
            Err(_) => Err(NexarError::CollectiveFailed {
                operation,
                rank: dest,
                reason: format!("relay send timed out after {}s", timeout.as_secs()),
            }),
        };
    }

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

/// Receive bytes from a peer with tag-based routing and timeout.
///
/// For sparse topologies, receives from relay delivery channels when
/// `src` is not a direct neighbor.
pub(crate) async fn collective_recv(
    client: &NexarClient,
    src: Rank,
    operation: &'static str,
    tag: CollectiveTag,
) -> Result<PooledBuf> {
    let timeout = collective_timeout(client);

    // Sparse topology relay path: if src is not a direct peer, receive via relay.
    if !client.has_direct_peer(src)
        && let (Some(deliveries), Some(t)) = (&client.relay_deliveries, tag)
    {
        let result = tokio::time::timeout(timeout, deliveries.recv_tagged(src, t)).await;
        return match result {
            Ok(Ok(buf)) => Ok(buf),
            Ok(Err(e)) => Err(NexarError::CollectiveFailed {
                operation,
                rank: src,
                reason: e.to_string(),
            }),
            Err(_) => Err(NexarError::CollectiveFailed {
                operation,
                rank: src,
                reason: format!("relay recv timed out after {}s", timeout.as_secs()),
            }),
        };
    }

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
