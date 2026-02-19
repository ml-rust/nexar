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

/// Default timeout for individual send/recv operations within collectives.
pub(crate) const COLLECTIVE_TIMEOUT: Duration = Duration::from_secs(30);

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
