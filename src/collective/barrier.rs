use crate::client::NexarClient;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::types::Priority;
use std::time::Duration;

/// Two-phase barrier: all ranks send to rank 0, rank 0 broadcasts ack.
///
/// Phase 1: Every rank (except 0) sends `Barrier { epoch }` to rank 0.
/// Phase 2: Rank 0 waits for all, then sends `BarrierAck { epoch }` to all.
pub async fn two_phase_barrier(client: &NexarClient, timeout: Duration) -> Result<()> {
    let epoch = client.next_barrier_epoch();
    let rank = client.rank();
    let world = client.world_size();

    if world <= 1 {
        return Ok(());
    }

    if rank == 0 {
        // Rank 0: collect barriers from all other ranks.
        for r in 1..world {
            let msg = tokio::time::timeout(timeout, client.recv_control(r))
                .await
                .map_err(|_| NexarError::BarrierTimeout {
                    epoch,
                    timeout_ms: timeout.as_millis() as u64,
                })??;

            match msg {
                NexarMessage::Barrier { epoch: e } if e == epoch => {}
                other => {
                    return Err(NexarError::DecodeFailed(format!(
                        "barrier: expected Barrier(epoch={epoch}), got {other:?}"
                    )));
                }
            }
        }

        // Broadcast BarrierAck.
        let ack = NexarMessage::BarrierAck { epoch };
        for r in 1..world {
            client
                .peer(r)?
                .send_message(&ack, Priority::Critical)
                .await?;
        }
    } else {
        // Non-zero rank: send Barrier to rank 0, wait for BarrierAck.
        let barrier_msg = NexarMessage::Barrier { epoch };
        client
            .peer(0)?
            .send_message(&barrier_msg, Priority::Critical)
            .await?;

        let ack = tokio::time::timeout(timeout, client.recv_control(0))
            .await
            .map_err(|_| NexarError::BarrierTimeout {
                epoch,
                timeout_ms: timeout.as_millis() as u64,
            })??;

        match ack {
            NexarMessage::BarrierAck { epoch: e } if e == epoch => {}
            other => {
                return Err(NexarError::DecodeFailed(format!(
                    "barrier: expected BarrierAck(epoch={epoch}), got {other:?}"
                )));
            }
        }
    }

    Ok(())
}
