use crate::error::Result;
use crate::types::{DataType, IoVec, ReduceOp};

/// Bridges device memory with nexar's network transport.
///
/// Nexar operates on raw `u64` pointers + byte counts. The `DeviceAdapter`
/// handles staging data between device memory and host buffers for network I/O.
///
/// - `CpuAdapter` (built-in): direct pointer access for host memory.
/// - GPU adapters: device-to-host / host-to-device copies (implemented externally).
///
/// # GPU memory and network I/O
///
/// For inter-node communication, GPU→host→network→host→GPU transfers are
/// **unavoidable** — the network card reads from host memory, not device memory.
/// This is not the forbidden GPU↔CPU pattern; it is the physical reality of
/// network I/O. NCCL does the same thing internally (GPUDirect RDMA merely
/// hides the copy in hardware).
pub trait DeviceAdapter: Send + Sync {
    /// Copy from device memory to a host buffer for network send.
    ///
    /// For CPU: read directly from the pointer.
    /// For GPU: device-to-host copy.
    ///
    /// # Safety
    /// `ptr` must be a valid pointer to at least `size_bytes` bytes.
    unsafe fn stage_for_send(&self, ptr: u64, size_bytes: usize) -> Result<Vec<u8>>;

    /// Copy received data from host into device memory.
    ///
    /// # Safety
    /// `dst_ptr` must be a valid pointer to at least `data.len()` bytes.
    unsafe fn receive_to_device(&self, data: &[u8], dst_ptr: u64) -> Result<()>;

    /// In-place reduce: `dst[i] = op(dst[i], src[i])` for each element.
    ///
    /// Used by collective algorithms for local reduction steps.
    ///
    /// # Safety
    /// `dst_ptr` must be a valid pointer to at least `count * dtype.size_in_bytes()` bytes.
    unsafe fn reduce_inplace(
        &self,
        dst_ptr: u64,
        src: &[u8],
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()>;

    /// Gather multiple non-contiguous regions into a single contiguous buffer for send.
    ///
    /// Default implementation calls `stage_for_send` per region and concatenates.
    ///
    /// # Safety
    /// Each region's `ptr` must be valid for its `len` bytes.
    unsafe fn stage_for_send_iov(&self, regions: &[IoVec]) -> Result<Vec<u8>> {
        let total: usize = regions.iter().map(|r| r.len).sum();
        let mut buf = Vec::with_capacity(total);
        for region in regions {
            let chunk = unsafe { self.stage_for_send(region.ptr, region.len)? };
            buf.extend_from_slice(&chunk);
        }
        Ok(buf)
    }

    /// Async D2H copy on a specific CUDA stream.
    ///
    /// Default: delegates to synchronous `stage_for_send` (ignores stream).
    /// GPU adapters should override to use async copies for compute/comms overlap.
    ///
    /// # Safety
    /// `ptr` must be a valid pointer to at least `size_bytes` bytes.
    /// `stream` must be a valid CUDA stream handle (or 0 for default stream).
    unsafe fn stage_for_send_on_stream(
        &self,
        ptr: u64,
        size_bytes: usize,
        _stream: u64,
    ) -> Result<Vec<u8>> {
        unsafe { self.stage_for_send(ptr, size_bytes) }
    }

    /// Async H2D copy on a specific CUDA stream.
    ///
    /// Default: delegates to synchronous `receive_to_device` (ignores stream).
    /// GPU adapters should override to use async copies for compute/comms overlap.
    ///
    /// # Safety
    /// `dst_ptr` must be a valid pointer to at least `data.len()` bytes.
    /// `stream` must be a valid CUDA stream handle (or 0 for default stream).
    unsafe fn receive_to_device_on_stream(
        &self,
        data: &[u8],
        dst_ptr: u64,
        _stream: u64,
    ) -> Result<()> {
        unsafe { self.receive_to_device(data, dst_ptr) }
    }

    /// Scatter received contiguous data into multiple non-contiguous device regions.
    ///
    /// Default implementation calls `receive_to_device` per region from successive
    /// slices of `data`.
    ///
    /// # Safety
    /// Each region's `ptr` must be valid for its `len` bytes.
    /// `data.len()` must equal the sum of all region lengths.
    unsafe fn receive_to_device_iov(&self, data: &[u8], regions: &[IoVec]) -> Result<()> {
        let mut offset = 0;
        for region in regions {
            unsafe {
                self.receive_to_device(&data[offset..offset + region.len], region.ptr)?;
            }
            offset += region.len;
        }
        Ok(())
    }
}
