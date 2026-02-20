//! GPUDirect RDMA context: raw FFI device, PD management.

use nexar::error::{NexarError, Result};
use std::os::raw::c_int;

use super::mr::GpuMr;
use super::qp::PreparedGpuDirectQp;

/// Self-contained RDMA context for GPUDirect operations via raw `ibverbs-sys` FFI.
pub struct GpuDirectContext {
    pub(super) ctx: *mut ibverbs_sys::ibv_context,
    pub(super) pd: *mut ibverbs_sys::ibv_pd,
}

unsafe impl Send for GpuDirectContext {}
unsafe impl Sync for GpuDirectContext {}

impl GpuDirectContext {
    /// Open an RDMA device and allocate PD for GPUDirect operations.
    pub fn new(device_index: Option<usize>) -> Result<Self> {
        unsafe {
            let mut num_devices: c_int = 0;
            let dev_list = ibverbs_sys::ibv_get_device_list(&mut num_devices);
            if dev_list.is_null() || num_devices == 0 {
                return Err(NexarError::device("GPUDirect: no RDMA devices found"));
            }

            let idx = device_index.unwrap_or(0);
            if idx >= num_devices as usize {
                ibverbs_sys::ibv_free_device_list(dev_list);
                return Err(NexarError::device(format!(
                    "GPUDirect: device index {idx} out of range (have {num_devices})"
                )));
            }

            let dev = *dev_list.add(idx);
            let ctx = ibverbs_sys::ibv_open_device(dev);
            ibverbs_sys::ibv_free_device_list(dev_list);

            if ctx.is_null() {
                return Err(NexarError::device("GPUDirect: ibv_open_device failed"));
            }

            let pd = ibverbs_sys::ibv_alloc_pd(ctx);
            if pd.is_null() {
                ibverbs_sys::ibv_close_device(ctx);
                return Err(NexarError::device("GPUDirect: ibv_alloc_pd failed"));
            }

            Ok(Self { ctx, pd })
        }
    }

    /// Register a CUDA device pointer as an RDMA memory region.
    ///
    /// # Safety
    ///
    /// - `gpu_ptr` must be a valid CUDA device pointer.
    /// - The GPU memory must remain allocated for the lifetime of the returned `GpuMr`.
    pub unsafe fn register_gpu_memory(&self, gpu_ptr: u64, size: usize) -> Result<GpuMr> {
        let access = ibverbs_sys::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | ibverbs_sys::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
            | ibverbs_sys::ibv_access_flags::IBV_ACCESS_REMOTE_READ;

        let mr =
            unsafe { ibverbs_sys::ibv_reg_mr(self.pd, gpu_ptr as *mut _, size, access.0 as c_int) };
        if mr.is_null() {
            return Err(NexarError::device(format!(
                "GPUDirect: ibv_reg_mr failed for gpu_ptr=0x{gpu_ptr:x} size={size}. \
                 Is nvidia-peermem loaded? (modprobe nvidia-peermem)"
            )));
        }

        Ok(GpuMr::new(mr, gpu_ptr, size))
    }

    /// Prepare a QP for GPUDirect sends/recvs.
    pub fn prepare_qp(&self) -> Result<PreparedGpuDirectQp> {
        PreparedGpuDirectQp::create(self)
    }
}

impl Drop for GpuDirectContext {
    fn drop(&mut self) {
        unsafe {
            if !self.pd.is_null() {
                ibverbs_sys::ibv_dealloc_pd(self.pd);
            }
            if !self.ctx.is_null() {
                ibverbs_sys::ibv_close_device(self.ctx);
            }
        }
    }
}
