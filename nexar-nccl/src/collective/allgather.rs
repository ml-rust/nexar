use nexar::types::DataType;

use crate::comm::HierarchicalComm;
use crate::error::{NcclCommError, Result};

/// Hierarchical allgather: each rank contributes `count` elements.
///
/// Algorithm:
/// - NCCL allgather intra-node (each GPU now has all local data).
/// - nexar allgather inter-node on leads (leads exchange node-level chunks).
/// - NCCL broadcast from lead to all local GPUs.
///
/// # Safety
/// - `send_ptr`: `count * dtype.size_in_bytes()` bytes per rank.
/// - `recv_ptr`: `count * world_size * dtype.size_in_bytes()` bytes total.
pub async unsafe fn hierarchical_allgather(
    comm: &HierarchicalComm,
    send_ptr: u64,
    recv_ptr: u64,
    count: usize,
    dtype: DataType,
) -> Result<()> {
    // Single-node: just NCCL allgather.
    if comm.is_single_node() {
        unsafe {
            comm.nccl().allgather(send_ptr, recv_ptr, count, dtype)?;
        }
        comm.synchronize()?;
        return Ok(());
    }

    let local_world = comm.local_world_size();
    let elem_size = dtype.size_in_bytes();
    let per_rank_bytes = count * elem_size;

    // Step 1: NCCL allgather intra-node.
    // After this, each local GPU has data from all local GPUs in the recv buffer
    // at positions corresponding to their local rank within this node's chunk.
    // However, we need to place data at the correct global positions.
    //
    // Strategy: Each rank places its data at its global rank position in recv_ptr.
    // First, copy own data to the correct global slot.
    // Then NCCL allgather among local ranks, writing into the node's portion of recv_ptr.
    let topo = comm.topology();
    let node_start_rank = topo.local_ranks[0] as usize;
    let node_recv_offset = (node_start_rank * per_rank_bytes) as u64;

    // NCCL allgather: send_ptr → recv_ptr[node_start..node_start+local_world]
    unsafe {
        comm.nccl()
            .allgather(send_ptr, recv_ptr + node_recv_offset, count, dtype)?;
    }
    comm.synchronize()?;

    // Step 2: Leads exchange inter-node data via nexar allgather.
    if comm.is_lead() {
        if let Some(inter) = comm.inter_node() {
            // Each lead has local_world * count elements (its node's data).
            // Allgather across leads to fill the entire recv_ptr.
            let node_chunk_count = local_world * count;
            let node_chunk_ptr = recv_ptr + node_recv_offset;

            unsafe {
                inter
                    .all_gather(node_chunk_ptr, recv_ptr, node_chunk_count, dtype)
                    .await
                    .map_err(NcclCommError::Nexar)?;
            }
        }
    }

    // Step 3: NCCL broadcast the full recv buffer from lead to all local GPUs.
    let total_count = count * comm.world_size() as usize;
    unsafe {
        comm.nccl()
            .broadcast_inplace(recv_ptr, total_count, dtype, 0)?;
    }
    comm.synchronize()?;

    Ok(())
}
