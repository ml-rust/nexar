//! Bucketed allreduce: fuse multiple small tensors into one allreduce.
//!
//! Gathers non-contiguous `(ptr, count)` pairs into a single contiguous buffer,
//! runs a single ring allreduce, then scatters the results back. This amortizes
//! the per-collective overhead (latency, synchronization) across many small
//! tensors — critical for gradient synchronization where models have hundreds
//! of parameter groups.

use crate::client::NexarClient;
use crate::collective::allreduce::ring_allreduce_with_tag;
use crate::collective::helpers::CollectiveTag;
use crate::error::Result;
use crate::types::{DataType, IoVec, ReduceOp};

/// Tagged variant for non-blocking bucketed allreduce.
pub(crate) async unsafe fn allreduce_bucketed_with_tag(
    client: &NexarClient,
    entries: &[(u64, usize)],
    dtype: DataType,
    op: ReduceOp,
    tag: CollectiveTag,
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

    if !client.adapter().supports_host_offload() {
        return Err(crate::error::NexarError::CollectiveFailed {
            operation: "allreduce_bucketed",
            rank: client.rank(),
            reason:
                "bucketed allreduce requires a host-offload capable adapter (e.g. CpuAdapter); \
                     GPU users should use nexar-nccl's on-device bucketed operations"
                    .into(),
        });
    }

    let elem_size = dtype.size_in_bytes();

    // Build IoVec regions for gather/scatter.
    let regions: Vec<IoVec> = entries
        .iter()
        .map(|&(ptr, count)| IoVec {
            ptr,
            len: count * elem_size,
        })
        .collect();

    // Gather all entries from device into a single contiguous host buffer.
    let flat = unsafe { client.adapter().stage_for_send_iov(&regions)? };
    let total_count: usize = entries.iter().map(|&(_, c)| c).sum();
    let total_bytes = total_count * elem_size;

    // CpuAdapter treats host pointers as device pointers, so we allreduce
    // directly in the host buffer and scatter the result back.
    let mut buf = flat;
    debug_assert_eq!(buf.len(), total_bytes);

    let buf_ptr = buf.as_mut_ptr() as u64;
    unsafe {
        ring_allreduce_with_tag(client, buf_ptr, total_count, dtype, op, tag).await?;
    }

    // Scatter the reduced data back to original device locations.
    unsafe { client.adapter().receive_to_device_iov(&buf, &regions)? };

    Ok(())
}
