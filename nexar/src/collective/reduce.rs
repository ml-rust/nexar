use crate::client::NexarClient;
use crate::collective::helpers::{collective_recv, collective_send};
use crate::error::{NexarError, Result};
use crate::reduce::reduce_slice;
use crate::types::{DataType, Rank, ReduceOp};

/// Tree reduce: reduce data from all ranks to a single root rank.
///
/// Uses a binomial tree algorithm with log₂(N) communication rounds.
/// For non-power-of-2 world sizes, excess ranks first send their data
/// to partners in the lower range, reducing to a power-of-2 problem.
///
/// After completion, only `root` holds the reduced result. All other
/// ranks' buffers are left in an unspecified state.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn tree_reduce(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    root: Rank,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;
    let root = root as usize;

    if world <= 1 {
        return Ok(());
    }

    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };
    let mut buf = data;

    // Remap so root becomes virtual rank 0. This lets us use the standard
    // binomial tree where rank 0 is always the final destination.
    let vrank = (rank + world - root) % world;

    // Find largest power of 2 <= world.
    let p2 = if world.is_power_of_two() {
        world
    } else {
        world.next_power_of_two() >> 1
    };
    let excess = world - p2;

    // Step 1: Excess ranks (virtual ranks [p2..world)) send to [0..excess).
    let mut participating = true;
    if vrank < excess {
        let partner_vrank = vrank + p2;
        let partner_real = (partner_vrank + root) % world;
        let received = collective_recv(client, partner_real as u32, "reduce").await?;
        if received.len() != total_bytes {
            return Err(NexarError::BufferSizeMismatch {
                expected: total_bytes,
                actual: received.len(),
            });
        }
        reduce_slice(&mut buf, &received, count, dtype, op)?;
    } else if vrank >= p2 {
        let partner_vrank = vrank - p2;
        let partner_real = (partner_vrank + root) % world;
        collective_send(client, partner_real as u32, &buf, "reduce").await?;
        participating = false;
    }

    // Step 2: Binomial tree reduce among the p2 participating virtual ranks.
    if participating {
        let adjusted_vrank = vrank;
        let log2 = p2.trailing_zeros() as usize;

        for round in 0..log2 {
            let mask = 1 << round;
            if adjusted_vrank & mask != 0 {
                // This rank sends to partner and exits.
                let partner_vrank = adjusted_vrank ^ mask;
                let partner_real = (partner_vrank + root) % world;
                collective_send(client, partner_real as u32, &buf, "reduce").await?;
                break;
            } else {
                // This rank receives from partner (if partner exists).
                let partner_vrank = adjusted_vrank ^ mask;
                if partner_vrank < p2 {
                    let partner_real = (partner_vrank + root) % world;
                    let received = collective_recv(client, partner_real as u32, "reduce").await?;
                    if received.len() != total_bytes {
                        return Err(NexarError::BufferSizeMismatch {
                            expected: total_bytes,
                            actual: received.len(),
                        });
                    }
                    reduce_slice(&mut buf, &received, count, dtype, op)?;
                }
            }
        }
    }

    // Only root writes the result back.
    if rank == root {
        unsafe { client.adapter().receive_to_device(&buf, ptr)? };
    }

    Ok(())
}
