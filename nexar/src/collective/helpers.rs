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

/// Receive bytes using best available transport (RDMA then QUIC) with timeout.
///
/// Only useful when the expected size is known (e.g., fixed-size collectives).
#[allow(dead_code)]
pub(crate) async fn collective_recv_best_effort(
    client: &NexarClient,
    src: Rank,
    expected_size: usize,
    operation: &'static str,
) -> Result<PooledBuf> {
    match tokio::time::timeout(
        COLLECTIVE_TIMEOUT,
        client.recv_bytes_best_effort(src, expected_size),
    )
    .await
    {
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

/// Tag context for collective operations: `None` for untagged (default lane),
/// `Some(tag)` for tagged transport (concurrent collectives).
pub(crate) type CollectiveTag = Option<u64>;

/// Send bytes to a peer with timeout, wrapping errors as `CollectiveFailed`.
///
/// If `tag` is `Some`, uses tagged transport; otherwise uses the best
/// available transport (RDMA if attached, QUIC fallback).
pub(crate) async fn collective_send(
    client: &NexarClient,
    dest: Rank,
    data: &[u8],
    operation: &'static str,
) -> Result<()> {
    collective_send_with_tag(client, dest, data, operation, None).await
}

/// Receive bytes from a peer with timeout, wrapping errors as `CollectiveFailed`.
pub(crate) async fn collective_recv(
    client: &NexarClient,
    src: Rank,
    operation: &'static str,
) -> Result<PooledBuf> {
    collective_recv_with_tag(client, src, operation, None).await
}

/// Send bytes to a peer with optional tag and timeout.
pub(crate) async fn collective_send_with_tag(
    client: &NexarClient,
    dest: Rank,
    data: &[u8],
    operation: &'static str,
    tag: CollectiveTag,
) -> Result<()> {
    let result = match tag {
        Some(t) => {
            tokio::time::timeout(
                COLLECTIVE_TIMEOUT,
                client.send_bytes_tagged_best_effort(dest, t, data),
            )
            .await
        }
        None => {
            tokio::time::timeout(
                COLLECTIVE_TIMEOUT,
                client.send_bytes_best_effort(dest, data),
            )
            .await
        }
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
            reason: format!("send timed out after {}s", COLLECTIVE_TIMEOUT.as_secs()),
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
    let result = match tag {
        Some(t) => {
            tokio::time::timeout(
                COLLECTIVE_TIMEOUT,
                client.recv_bytes_tagged_best_effort(src, t, 0),
            )
            .await
        }
        None => tokio::time::timeout(COLLECTIVE_TIMEOUT, client.recv_bytes(src)).await,
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
            reason: format!("recv timed out after {}s", COLLECTIVE_TIMEOUT.as_secs()),
        }),
    }
}
