use crate::error::Result;
use crate::types::{DataType, ReduceOp};

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
}
