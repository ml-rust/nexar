//! Reduce-Scatter + Allgather allreduce decomposition.
//!
//! Alternative allreduce via RS→AG composition. Useful for ZeRO-style sharding
//! where intermediate reduce-scatter results are needed, or when the two-phase
//! decomposition maps better to the network topology.

use crate::client::NexarClient;
use crate::collective::allgather::ring_allgather_with_tag;
use crate::collective::helpers::CollectiveTag;
use crate::collective::reduce_scatter::ring_reduce_scatter_with_tag;
use crate::error::{NexarError, Result};
use crate::types::{DataType, ReduceOp};

/// Allreduce via reduce-scatter followed by allgather.
///
/// Equivalent to `ring_allreduce` but decomposed into two phases:
/// 1. Reduce-scatter: each rank gets `count / world` reduced elements
/// 2. Allgather: reconstruct the full reduced tensor on all ranks
///
/// # Errors
/// Returns [`NexarError::IndivisibleCount`] if `count` is not evenly
/// divisible by the world size.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn rs_ag_allreduce(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    unsafe { rs_ag_allreduce_with_tag(client, ptr, count, dtype, op, None).await }
}

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

    if world <= 1 {
        return Ok(());
    }

    if !count.is_multiple_of(world) {
        return Err(NexarError::IndivisibleCount {
            count,
            world_size: world,
            operation: "rs_ag_allreduce",
        });
    }

    let elem_size = dtype.size_in_bytes();
    let chunk_count = count / world;

    // Reduce-scatter: each rank gets its chunk of the reduced result.
    // send_ptr = ptr (full tensor), recv_ptr = temp buffer for this rank's chunk.
    let chunk_bytes = chunk_count * elem_size;
    let mut rs_output = vec![0u8; chunk_bytes];
    let rs_output_ptr = rs_output.as_mut_ptr() as u64;

    // Use distinct tags for RS and AG phases to avoid cross-talk.
    let rs_tag = tag.map(|t| t.wrapping_mul(2));
    let ag_tag = tag.map(|t| t.wrapping_mul(2).wrapping_add(1));

    unsafe {
        ring_reduce_scatter_with_tag(client, ptr, rs_output_ptr, chunk_count, dtype, op, rs_tag)
            .await?;
    }

    // Allgather: reconstruct the full reduced tensor from each rank's chunk.
    // send_ptr = rs_output (this rank's chunk), recv_ptr = ptr (full tensor output).
    unsafe {
        ring_allgather_with_tag(client, rs_output_ptr, ptr, chunk_count, dtype, ag_tag).await?;
    }

    Ok(())
}
