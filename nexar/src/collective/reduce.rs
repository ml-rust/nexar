use crate::client::NexarClient;
use crate::collective::helpers::{
    CollectiveTag, collective_recv_with_tag, collective_send_with_tag,
};
use crate::error::{NexarError, Result};
use crate::reduce::reduce_slice;
use crate::types::{DataType, Rank, ReduceOp};

/// Tree reduce: reduce data from all ranks to a single root rank.
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
    unsafe { tree_reduce_with_tag(client, ptr, count, dtype, op, root, None).await }
}

/// Tagged variant for non-blocking collectives.
pub(crate) async unsafe fn tree_reduce_with_tag(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    root: Rank,
    tag: CollectiveTag,
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

    let vrank = (rank + world - root) % world;

    let p2 = if world.is_power_of_two() {
        world
    } else {
        world.next_power_of_two() >> 1
    };
    let excess = world - p2;

    let mut participating = true;
    if vrank < excess {
        let partner_vrank = vrank + p2;
        let partner_real = (partner_vrank + root) % world;
        let received = collective_recv_with_tag(client, partner_real as u32, "reduce", tag).await?;
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
        collective_send_with_tag(client, partner_real as u32, &buf, "reduce", tag).await?;
        participating = false;
    }

    if participating {
        let adjusted_vrank = vrank;
        let log2 = p2.trailing_zeros() as usize;

        for round in 0..log2 {
            let mask = 1 << round;
            if adjusted_vrank & mask != 0 {
                let partner_vrank = adjusted_vrank ^ mask;
                let partner_real = (partner_vrank + root) % world;
                collective_send_with_tag(client, partner_real as u32, &buf, "reduce", tag).await?;
                break;
            } else {
                let partner_vrank = adjusted_vrank ^ mask;
                if partner_vrank < p2 {
                    let partner_real = (partner_vrank + root) % world;
                    let received =
                        collective_recv_with_tag(client, partner_real as u32, "reduce", tag)
                            .await?;
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

    if rank == root {
        unsafe { client.adapter().receive_to_device(&buf, ptr)? };
    }

    Ok(())
}
