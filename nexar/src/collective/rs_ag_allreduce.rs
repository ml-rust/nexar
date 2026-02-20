//! Reduce-Scatter + Allgather allreduce decomposition.
//!
//! Alternative allreduce via RS→AG composition. Useful for ZeRO-style sharding
//! where intermediate reduce-scatter results are needed, or when the two-phase
//! decomposition maps better to the network topology.
//!
//! Handles arbitrary element counts (not necessarily divisible by world size)
//! by distributing remainder elements across ranks via [`ChunkLayout`].

use crate::client::NexarClient;
use crate::collective::helpers::{
    ChunkLayout, CollectiveTag, collective_recv_with_tag, collective_send_with_tag,
};
use crate::error::{NexarError, Result};
use crate::reduce::reduce_slice;
use crate::types::{DataType, ReduceOp};

/// Tagged variant for non-blocking RS+AG allreduce.
pub(crate) async unsafe fn rs_ag_allreduce_with_tag(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;

    if world <= 1 {
        return Ok(());
    }

    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;
    let layout = ChunkLayout::new(count, world);

    let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };
    let mut buf = data;

    let next = (rank + 1) % world;
    let prev = (rank + world - 1) % world;

    // Use distinct tags for RS and AG phases to avoid cross-talk.
    let rs_tag = tag.map(|t| t.wrapping_mul(2));
    let ag_tag = tag.map(|t| t.wrapping_mul(2).wrapping_add(1));

    // Phase 1: Scatter-reduce (N-1 rounds).
    // Same algorithm as ring_allreduce_impl phase 1, using ChunkLayout
    // for variable-sized chunks.
    for step in 0..(world - 1) {
        let send_idx = (rank + world - step) % world;
        let send_off = layout.offsets[send_idx] * elem_size;
        let send_len = layout.chunk_count(send_idx) * elem_size;

        let recv_idx = (rank + world - step - 1) % world;
        let recv_off = layout.offsets[recv_idx] * elem_size;
        let recv_count = layout.chunk_count(recv_idx);
        let recv_len = recv_count * elem_size;

        let send_slice = &buf[send_off..send_off + send_len];

        let (_, received) = tokio::try_join!(
            collective_send_with_tag(client, next as u32, send_slice, "rs_ag_allreduce", rs_tag),
            collective_recv_with_tag(client, prev as u32, "rs_ag_allreduce", rs_tag),
        )?;

        if received.len() != recv_len {
            return Err(NexarError::BufferSizeMismatch {
                expected: recv_len,
                actual: received.len(),
            });
        }
        let dst_slice = &mut buf[recv_off..recv_off + recv_len];
        reduce_slice(dst_slice, &received, recv_count, dtype, op)?;
    }

    // Phase 2: Allgather (N-1 rounds).
    for step in 0..(world - 1) {
        let send_idx = (rank + world + 1 - step) % world;
        let send_off = layout.offsets[send_idx] * elem_size;
        let send_len = layout.chunk_count(send_idx) * elem_size;

        let recv_idx = (rank + world - step) % world;
        let recv_off = layout.offsets[recv_idx] * elem_size;
        let recv_len = layout.chunk_count(recv_idx) * elem_size;

        let send_slice = &buf[send_off..send_off + send_len];

        let (_, received) = tokio::try_join!(
            collective_send_with_tag(client, next as u32, send_slice, "rs_ag_allreduce", ag_tag),
            collective_recv_with_tag(client, prev as u32, "rs_ag_allreduce", ag_tag),
        )?;

        if received.len() != recv_len {
            return Err(NexarError::BufferSizeMismatch {
                expected: recv_len,
                actual: received.len(),
            });
        }
        buf[recv_off..recv_off + recv_len].copy_from_slice(&received);
    }

    unsafe { client.adapter().receive_to_device(&buf, ptr)? };

    Ok(())
}
