use nexar::types::{DataType, ReduceOp};

use crate::comm::HierarchicalComm;
use crate::error::{NcclCommError, Result};

/// Threshold below which we use the simple path (NCCL allreduce + nexar allreduce)
/// instead of the optimized reduce-scatter / allgather decomposition.
const SMALL_MSG_THRESHOLD: usize = 256 * 1024; // 256 KB

/// Hierarchical allreduce: NCCL intra-node, nexar inter-node.
///
/// Algorithm selection:
/// - Single node → NCCL allreduce (nothing else needed).
/// - Small message (< 256KB) → NCCL allreduce intra + nexar allreduce on leads + NCCL broadcast.
/// - Large message (evenly divisible) → NCCL reduce-scatter + nexar allreduce chunk + NCCL allgather.
/// - Large message (not evenly divisible) → falls back to small message path.
///
/// # Safety
/// `ptr` must be a valid GPU device pointer for `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn hierarchical_allreduce(
    comm: &HierarchicalComm,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    // Single-node: just NCCL allreduce.
    if comm.is_single_node() {
        unsafe {
            comm.nccl().allreduce_inplace(ptr, count, dtype, op)?;
        }
        comm.synchronize()?;
        return Ok(());
    }

    let msg_size = count * dtype.size_in_bytes();
    let local_world = comm.local_world_size();

    // Use the optimized reduce-scatter path only when the count is evenly
    // divisible by local_world AND the message is large enough to benefit.
    if msg_size >= SMALL_MSG_THRESHOLD && count % local_world == 0 {
        unsafe { large_msg_allreduce(comm, ptr, count, dtype, op).await }
    } else {
        unsafe { small_msg_allreduce(comm, ptr, count, dtype, op).await }
    }
}

/// Small message path: NCCL allreduce intra → nexar allreduce leads → NCCL broadcast.
///
/// Also used as fallback when count is not evenly divisible by local_world.
///
/// # Safety
/// `ptr` must be a valid GPU device pointer.
async unsafe fn small_msg_allreduce(
    comm: &HierarchicalComm,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    let nccl = comm.nccl();

    // Step 1: NCCL allreduce across local GPUs.
    unsafe {
        nccl.allreduce_inplace(ptr, count, dtype, op)?;
    }
    comm.synchronize()?;

    // Step 2: Lead rank runs nexar allreduce across nodes.
    if comm.is_lead() {
        let inter = comm
            .inter_node()
            .ok_or_else(|| NcclCommError::NotLeadRank { rank: comm.rank() })?;
        unsafe {
            inter
                .all_reduce(ptr, count, dtype, op)
                .await
                .map_err(NcclCommError::Nexar)?;
        }
    }

    // Step 3: NCCL broadcast from lead (local_rank 0) to all local GPUs.
    unsafe {
        nccl.broadcast_inplace(ptr, count, dtype, 0)?;
    }
    comm.synchronize()?;

    Ok(())
}

/// Large message path: NCCL reduce-scatter → nexar allreduce on chunk → NCCL allgather.
///
/// Precondition: `count % local_world == 0` (caller guarantees this).
///
/// This minimizes inter-node traffic: only 1/local_world of the data crosses the network.
///
/// # Safety
/// `ptr` must be a valid GPU device pointer.
async unsafe fn large_msg_allreduce(
    comm: &HierarchicalComm,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    let nccl = comm.nccl();
    let local_world = comm.local_world_size();
    let local_rank = comm.local_rank();
    let elem_size = dtype.size_in_bytes();
    let chunk_count = count / local_world;
    let chunk_offset = (local_rank * chunk_count * elem_size) as u64;

    // Step 1: NCCL reduce-scatter. Each GPU gets 1/N of the reduced data.
    unsafe {
        nccl.reduce_scatter(ptr, ptr + chunk_offset, chunk_count, dtype, op)?;
    }
    comm.synchronize()?;

    // Step 2: Lead rank runs nexar allreduce on its chunk.
    if comm.is_lead() {
        if let Some(inter) = comm.inter_node() {
            // Lead is local_rank 0, so chunk_offset = 0.
            unsafe {
                inter
                    .all_reduce(ptr, chunk_count, dtype, op)
                    .await
                    .map_err(NcclCommError::Nexar)?;
            }
        }
    }

    // Step 3: NCCL allgather to reconstruct the full tensor on all local GPUs.
    unsafe {
        nccl.allgather(ptr + chunk_offset, ptr, chunk_count, dtype)?;
    }
    comm.synchronize()?;

    Ok(())
}
