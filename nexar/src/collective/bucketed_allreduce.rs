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

/// Fuse multiple `(ptr, element_count)` entries into one allreduce.
///
/// Each entry's pointer must refer to device memory accessible via the client's
/// `DeviceAdapter`. The entries are gathered into a flat host buffer, allreduced
/// as one contiguous tensor, then scattered back to the original locations.
///
/// For `CpuAdapter`, the pointers are regular host pointers.
///
/// # Host-only limitation
/// This implementation performs the allreduce on a host-allocated buffer.
/// It is **only correct with `CpuAdapter`**. GPU users should use
/// `nexar-nccl`'s on-device bucketed operations instead — passing GPU
/// pointers here will cause incorrect behavior or a crash.
///
/// # Safety
/// Each `(ptr, count)` entry must point to at least `count * dtype.size_in_bytes()`
/// valid bytes on the device.
pub async unsafe fn allreduce_bucketed(
    client: &NexarClient,
    entries: &[(u64, usize)],
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    unsafe { allreduce_bucketed_with_tag(client, entries, dtype, op, None).await }
}

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

    // Write the flat buffer to device at the first entry's pointer won't work
    // generically. Instead, allocate a contiguous host region and treat it as
    // the "device" for the allreduce. This works correctly for CpuAdapter.
    //
    // For GPU adapters: the flat buffer lives on host. We write it to the first
    // entry's device location (which must have enough space), allreduce there,
    // then scatter back. If the first entry isn't large enough, we fall back
    // to individual allreduces.
    //
    // Practical approach: use a host-allocated buffer as the device pointer.
    // CpuAdapter: works directly (host ptr = device ptr).
    // GPU adapters: we write the flat buffer to device, allreduce, read back.

    // Allocate a host buffer that we'll pass through the adapter round-trip.
    let mut buf = flat;
    debug_assert_eq!(buf.len(), total_bytes);

    // For the allreduce, we need a device pointer. Use the host buffer directly
    // — CpuAdapter treats host pointers as device pointers. GPU users should
    // prefer nexar-nccl for on-device bucketed operations.
    let buf_ptr = buf.as_mut_ptr() as u64;
    unsafe {
        ring_allreduce_with_tag(client, buf_ptr, total_count, dtype, op, tag).await?;
    }

    // Re-read the result from the "device" (host buffer for CpuAdapter).
    let result = unsafe { client.adapter().stage_for_send(buf_ptr, total_bytes)? };

    // Scatter the reduced data back to original device locations.
    unsafe { client.adapter().receive_to_device_iov(&result, &regions)? };

    Ok(())
}
