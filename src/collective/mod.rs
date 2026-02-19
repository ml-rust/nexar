mod allgather;
mod allreduce;
mod barrier;
mod broadcast;
mod reduce_scatter;

pub use allgather::ring_allgather;
pub use allreduce::ring_allreduce;
pub use barrier::two_phase_barrier;
pub use broadcast::tree_broadcast;
pub use reduce_scatter::ring_reduce_scatter;

use crate::client::NexarClient;
use crate::error::{NexarError, Result};
use crate::transport::buffer_pool::PooledBuf;
use crate::types::Rank;
use std::time::Duration;

/// Default timeout for individual send/recv operations within collectives.
const COLLECTIVE_TIMEOUT: Duration = Duration::from_secs(30);

/// Send bytes to a peer with timeout, wrapping errors as `CollectiveFailed`.
pub(crate) async fn collective_send(
    client: &NexarClient,
    dest: Rank,
    data: &[u8],
    operation: &'static str,
) -> Result<()> {
    match tokio::time::timeout(COLLECTIVE_TIMEOUT, client.send_bytes(dest, data)).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(NexarError::CollectiveFailed {
            operation,
            rank: dest,
            reason: e.to_string(),
        }),
        Err(_) => Err(NexarError::CollectiveFailed {
            operation,
            rank: dest,
            reason: format!("send timed out after {}s", COLLECTIVE_TIMEOUT.as_secs()),
        }),
    }
}

/// Receive bytes from a peer with timeout, wrapping errors as `CollectiveFailed`.
pub(crate) async fn collective_recv(
    client: &NexarClient,
    src: Rank,
    operation: &'static str,
) -> Result<PooledBuf> {
    match tokio::time::timeout(COLLECTIVE_TIMEOUT, client.recv_bytes(src)).await {
        Ok(Ok(buf)) => Ok(buf),
        Ok(Err(e)) => Err(NexarError::CollectiveFailed {
            operation,
            rank: src,
            reason: e.to_string(),
        }),
        Err(_) => Err(NexarError::CollectiveFailed {
            operation,
            rank: src,
            reason: format!("recv timed out after {}s", COLLECTIVE_TIMEOUT.as_secs()),
        }),
    }
}
