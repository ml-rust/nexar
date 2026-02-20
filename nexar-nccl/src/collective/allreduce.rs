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
/// Uses CUDA events to overlap NCCL and nexar operations, reducing the number
/// of stream synchronizations from 3 to 1.
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

/// Small message path with event-based overlap.
///
/// Flow:
/// 1. NCCL allreduce on nccl_stream
/// 2. Record event E1 on nccl_stream
/// 3. [Lead only] Wait E1 on staging/nccl stream, D2H copy, nexar allreduce, H2D copy
/// 4. Record event E2
/// 5. nccl_stream waits E2
/// 6. NCCL broadcast on nccl_stream
/// 7. Single synchronize at end
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
    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    // Step 1: NCCL allreduce across local GPUs.
    unsafe {
        nccl.allreduce_inplace(ptr, count, dtype, op)?;
    }

    // Step 2: Lead rank runs nexar allreduce across nodes with event overlap.
    if comm.is_lead() {
        let inter = comm
            .inter_node()
            .ok_or_else(|| NcclCommError::NotLeadRank { rank: comm.rank() })?;

        // Record event after NCCL allreduce completes.
        let event = nccl.create_event()?;
        unsafe { nccl.record_event(event)? };

        // Use staging stream for true overlap: async D2H on staging stream
        // avoids blocking the NCCL stream while data is copied to host.
        let xfer_stream = comm.staging_stream().unwrap_or_else(|| nccl.stream());

        // Make the transfer stream wait for NCCL allreduce to complete.
        unsafe { nccl.stream_wait_event(xfer_stream, event)? };

        // Async D2H on the transfer stream, then sync only that stream.
        let mut host_buf = vec![0u8; total_bytes];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                &mut host_buf,
                ptr as cudarc::driver::sys::CUdeviceptr,
                xfer_stream.cu_stream(),
            )
            .map_err(NcclCommError::CudaDriver)?;
            cudarc::driver::result::stream::synchronize(xfer_stream.cu_stream() as _)
                .map_err(NcclCommError::CudaDriver)?;
        }

        // Nexar allreduce on host (network transfer).
        unsafe {
            inter
                .all_reduce(host_buf.as_mut_ptr() as u64, count, dtype, op)
                .await
                .map_err(NcclCommError::Nexar)?;
        }

        // Async H2D on the transfer stream.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                ptr as cudarc::driver::sys::CUdeviceptr,
                &host_buf,
                xfer_stream.cu_stream(),
            )
            .map_err(NcclCommError::CudaDriver)?;
        }

        // Record event after H2D is enqueued so NCCL broadcast can proceed.
        let event2 = nccl.create_event()?;
        unsafe {
            cudarc::driver::result::event::record(event2, xfer_stream.cu_stream())
                .map_err(NcclCommError::CudaDriver)?;
        }

        // Make the NCCL stream wait for the H2D to complete.
        unsafe { nccl.stream_wait_event(nccl.stream(), event2)? };

        // Cleanup events.
        unsafe {
            nccl.destroy_event(event)?;
            nccl.destroy_event(event2)?;
        }
    } else {
        // Non-leads must synchronize to ensure NCCL allreduce is done before broadcast.
        comm.synchronize()?;
    }

    // Step 3: NCCL broadcast from lead (local_rank 0) to all local GPUs.
    unsafe {
        nccl.broadcast_inplace(ptr, count, dtype, 0)?;
    }
    comm.synchronize()?;

    Ok(())
}

/// Large message path with event-based overlap.
///
/// Flow:
/// 1. NCCL reduce-scatter → record E1
/// 2. [Lead] Wait E1, D2H chunk, nexar allreduce on chunk, H2D → record E2
/// 3. nccl_stream waits E2 → NCCL allgather → single synchronize
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
    let chunk_bytes = chunk_count * elem_size;

    // Step 1: NCCL reduce-scatter. Each GPU gets 1/N of the reduced data.
    unsafe {
        nccl.reduce_scatter(ptr, ptr + chunk_offset, chunk_count, dtype, op)?;
    }

    // Step 2: Lead rank runs nexar allreduce on its chunk with event overlap.
    if comm.is_lead() {
        if let Some(inter) = comm.inter_node() {
            let event = nccl.create_event()?;
            unsafe { nccl.record_event(event)? };

            let xfer_stream = comm.staging_stream().unwrap_or_else(|| nccl.stream());

            // Make transfer stream wait for reduce-scatter to complete.
            unsafe { nccl.stream_wait_event(xfer_stream, event)? };

            // Lead is local_rank 0, so chunk_offset = 0.
            // Async D2H on the transfer stream.
            let mut host_chunk = vec![0u8; chunk_bytes];
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(
                    &mut host_chunk,
                    ptr as cudarc::driver::sys::CUdeviceptr,
                    xfer_stream.cu_stream(),
                )
                .map_err(NcclCommError::CudaDriver)?;
                cudarc::driver::result::stream::synchronize(xfer_stream.cu_stream() as _)
                    .map_err(NcclCommError::CudaDriver)?;
            }

            // Nexar allreduce on host.
            unsafe {
                inter
                    .all_reduce(host_chunk.as_mut_ptr() as u64, chunk_count, dtype, op)
                    .await
                    .map_err(NcclCommError::Nexar)?;
            }

            // Async H2D on transfer stream.
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    ptr as cudarc::driver::sys::CUdeviceptr,
                    &host_chunk,
                    xfer_stream.cu_stream(),
                )
                .map_err(NcclCommError::CudaDriver)?;
            }

            // Record event after H2D enqueued, make NCCL stream wait.
            let event2 = nccl.create_event()?;
            unsafe {
                cudarc::driver::result::event::record(event2, xfer_stream.cu_stream())
                    .map_err(NcclCommError::CudaDriver)?;
            }
            unsafe { nccl.stream_wait_event(nccl.stream(), event2)? };

            unsafe {
                nccl.destroy_event(event)?;
                nccl.destroy_event(event2)?;
            }
        }
    } else {
        // Non-leads sync to ensure reduce-scatter is done before allgather.
        comm.synchronize()?;
    }

    // Step 3: NCCL allgather to reconstruct the full tensor on all local GPUs.
    unsafe {
        nccl.allgather(ptr + chunk_offset, ptr, chunk_count, dtype)?;
    }
    comm.synchronize()?;

    Ok(())
}
