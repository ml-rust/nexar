use crate::client::NexarClient;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::types::Priority;
use std::time::Duration;

/// Threshold: use two-phase barrier for small worlds, dissemination for larger.
const DISSEMINATION_THRESHOLD: u32 = 5;

/// Barrier: blocks until all ranks reach this point.
///
/// Dispatches to the appropriate algorithm based on world size:
/// - `two_phase_barrier` for world_size <= 4 (lower constant overhead)
/// - `dissemination_barrier` for world_size >= 5 (O(log N) rounds, no coordinator)
pub async fn barrier(client: &NexarClient, timeout: Duration) -> Result<()> {
    let world = client.world_size();
    if world <= 1 {
        return Ok(());
    }

    if world < DISSEMINATION_THRESHOLD {
        two_phase_barrier(client, timeout).await
    } else {
        dissemination_barrier(client, timeout).await
    }
}

/// Two-phase barrier: all ranks send to rank 0, rank 0 broadcasts ack.
///
/// Phase 1: Every rank (except 0) sends `Barrier { epoch }` to rank 0.
/// Phase 2: Rank 0 waits for all, then sends `BarrierAck { epoch }` to all.
///
/// Efficient for small world sizes (<=4). For larger worlds, the O(N) gather
/// and O(N) scatter at rank 0 becomes a bottleneck.
async fn two_phase_barrier(client: &NexarClient, timeout: Duration) -> Result<()> {
    let epoch = client.next_barrier_epoch();
    let rank = client.rank();
    let world = client.world_size();

    if rank == 0 {
        // Rank 0: collect barriers from all other ranks.
        for r in 1..world {
            let msg = tokio::time::timeout(timeout, client.recv_control(r))
                .await
                .map_err(|_| NexarError::CollectiveFailed {
                    operation: "barrier",
                    rank: r,
                    reason: format!(
                        "timed out waiting for Barrier(epoch={epoch}) after {}ms",
                        timeout.as_millis()
                    ),
                })?
                .map_err(|e| NexarError::CollectiveFailed {
                    operation: "barrier",
                    rank: r,
                    reason: e.to_string(),
                })?;

            match msg {
                NexarMessage::Barrier { epoch: e } if e == epoch => {}
                other => {
                    return Err(NexarError::CollectiveFailed {
                        operation: "barrier",
                        rank: r,
                        reason: format!("expected Barrier(epoch={epoch}), got {other:?}"),
                    });
                }
            }
        }

        // Broadcast BarrierAck.
        let ack = NexarMessage::BarrierAck { epoch };
        for r in 1..world {
            client
                .peer(r)?
                .send_message(&ack, Priority::Critical)
                .await
                .map_err(|e| NexarError::CollectiveFailed {
                    operation: "barrier",
                    rank: r,
                    reason: e.to_string(),
                })?;
        }
    } else {
        // Non-zero rank: send Barrier to rank 0, wait for BarrierAck.
        let barrier_msg = NexarMessage::Barrier { epoch };
        client
            .peer(0)?
            .send_message(&barrier_msg, Priority::Critical)
            .await
            .map_err(|e| NexarError::CollectiveFailed {
                operation: "barrier",
                rank: 0,
                reason: e.to_string(),
            })?;

        let ack = tokio::time::timeout(timeout, client.recv_control(0))
            .await
            .map_err(|_| NexarError::CollectiveFailed {
                operation: "barrier",
                rank: 0,
                reason: format!(
                    "timed out waiting for BarrierAck(epoch={epoch}) after {}ms",
                    timeout.as_millis()
                ),
            })?
            .map_err(|e| NexarError::CollectiveFailed {
                operation: "barrier",
                rank: 0,
                reason: e.to_string(),
            })?;

        match ack {
            NexarMessage::BarrierAck { epoch: e } if e == epoch => {}
            other => {
                return Err(NexarError::CollectiveFailed {
                    operation: "barrier",
                    rank: 0,
                    reason: format!("expected BarrierAck(epoch={epoch}), got {other:?}"),
                });
            }
        }
    }

    Ok(())
}

/// Dissemination barrier: O(log N) rounds, no single coordinator.
///
/// In round r, rank i sends to rank `(i + 2^r) % N` and receives from
/// rank `(i - 2^r + N) % N`. After `ceil(log2(N))` rounds, every rank
/// has transitively heard from every other rank.
///
/// Advantages over two-phase:
/// - No coordinator bottleneck (symmetric workload)
/// - O(log N) rounds instead of O(N) at rank 0
/// - Better scalability for large clusters
async fn dissemination_barrier(client: &NexarClient, timeout: Duration) -> Result<()> {
    let epoch = client.next_barrier_epoch();
    let rank = client.rank();
    let world = client.world_size();

    // Number of rounds: ceil(log2(world))
    let num_rounds = (world as f64).log2().ceil() as u32;

    for round in 0..num_rounds {
        let distance = 1u32 << round;
        let send_to = (rank + distance) % world;
        let recv_from = (rank + world - distance) % world;

        let msg = NexarMessage::Barrier { epoch };

        // Send and receive concurrently.
        let send_fut = async {
            client
                .peer(send_to)?
                .send_message(&msg, Priority::Critical)
                .await
                .map_err(|e| NexarError::CollectiveFailed {
                    operation: "barrier",
                    rank: send_to,
                    reason: e.to_string(),
                })
        };

        let recv_fut = async {
            let received = tokio::time::timeout(timeout, client.recv_control(recv_from))
                .await
                .map_err(|_| NexarError::CollectiveFailed {
                    operation: "barrier",
                    rank: recv_from,
                    reason: format!(
                        "timed out in dissemination round {round} after {}ms",
                        timeout.as_millis()
                    ),
                })?
                .map_err(|e| NexarError::CollectiveFailed {
                    operation: "barrier",
                    rank: recv_from,
                    reason: e.to_string(),
                })?;

            match received {
                NexarMessage::Barrier { epoch: e } if e == epoch => Ok(()),
                other => Err(NexarError::CollectiveFailed {
                    operation: "barrier",
                    rank: recv_from,
                    reason: format!("expected Barrier(epoch={epoch}), got {other:?}"),
                }),
            }
        };

        let (send_result, recv_result) = tokio::join!(send_fut, recv_fut);
        send_result?;
        recv_result?;
    }

    Ok(())
}
