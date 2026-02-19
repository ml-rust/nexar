//! CUDA device adapter for GPU memory staging.
//!
//! Uses cudarc 0.19 to copy GPU memory to/from host for network transfer.
//! Without GPUDirect RDMA, the path is GPU → Host → NIC → Host → GPU.

#![cfg(feature = "gpudirect")]

use nexar::device::DeviceAdapter;
use nexar::error::{NexarError, Result};
use nexar::types::{DataType, ReduceOp};
use std::sync::Arc;

/// Device adapter for NVIDIA GPUs using cudarc.
///
/// Handles device↔host staging for network I/O and CPU-side reduction.
pub struct CudaAdapter {
    ctx: Arc<cudarc::driver::CudaContext>,
}

impl CudaAdapter {
    /// Create a new CUDA adapter for the given GPU ordinal.
    pub fn new(device_ordinal: u32) -> Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(device_ordinal as usize).map_err(|e| {
            NexarError::DeviceError(format!(
                "failed to create CUDA context for GPU {device_ordinal}: {e}"
            ))
        })?;
        Ok(Self { ctx })
    }
}

impl DeviceAdapter for CudaAdapter {
    unsafe fn stage_for_send(&self, ptr: u64, size_bytes: usize) -> Result<Vec<u8>> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| NexarError::DeviceError(format!("CUDA bind_to_thread failed: {e}")))?;
        let mut buf = vec![0u8; size_bytes];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(
                &mut buf,
                ptr as cudarc::driver::sys::CUdeviceptr,
            )
        }
        .map_err(|e| NexarError::DeviceError(format!("cuMemcpyDtoH failed: {e}")))?;
        Ok(buf)
    }

    unsafe fn receive_to_device(&self, data: &[u8], dst_ptr: u64) -> Result<()> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| NexarError::DeviceError(format!("CUDA bind_to_thread failed: {e}")))?;
        unsafe {
            cudarc::driver::result::memcpy_htod_sync(
                dst_ptr as cudarc::driver::sys::CUdeviceptr,
                data,
            )
        }
        .map_err(|e| NexarError::DeviceError(format!("cuMemcpyHtoD failed: {e}")))?;
        Ok(())
    }

    unsafe fn reduce_inplace(
        &self,
        dst_ptr: u64,
        src: &[u8],
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        let elem_size = dtype.size_in_bytes();
        let total_bytes = count * elem_size;

        self.ctx
            .bind_to_thread()
            .map_err(|e| NexarError::DeviceError(format!("CUDA bind_to_thread failed: {e}")))?;

        let mut dst_host = vec![0u8; total_bytes];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(
                &mut dst_host,
                dst_ptr as cudarc::driver::sys::CUdeviceptr,
            )
        }
        .map_err(|e| NexarError::DeviceError(format!("cuMemcpyDtoH (reduce) failed: {e}")))?;

        // Perform CPU-side reduction into dst_host.
        let dst_host_ptr = dst_host.as_mut_ptr() as u64;
        unsafe { nexar::reduce::reduce_ptr(dst_host_ptr, src, count, dtype, op)? };

        unsafe {
            cudarc::driver::result::memcpy_htod_sync(
                dst_ptr as cudarc::driver::sys::CUdeviceptr,
                &dst_host,
            )
        }
        .map_err(|e| NexarError::DeviceError(format!("cuMemcpyHtoD (reduce) failed: {e}")))?;

        Ok(())
    }
}
