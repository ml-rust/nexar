//! GPU memory region and pooling for GPUDirect RDMA.

use crossbeam_queue::ArrayQueue;
use nexar::error::Result;
use std::sync::Arc;

use super::context::GpuDirectContext;

/// A GPU device pointer registered as an RDMA memory region.
pub struct GpuMr {
    mr: *mut ibverbs_sys::ibv_mr,
    gpu_ptr: u64,
    size: usize,
}

unsafe impl Send for GpuMr {}
unsafe impl Sync for GpuMr {}

impl GpuMr {
    pub(super) fn new(mr: *mut ibverbs_sys::ibv_mr, gpu_ptr: u64, size: usize) -> Self {
        Self { mr, gpu_ptr, size }
    }

    pub fn lkey(&self) -> u32 {
        unsafe { (*self.mr).lkey }
    }

    pub fn rkey(&self) -> u32 {
        unsafe { (*self.mr).rkey }
    }

    pub fn gpu_ptr(&self) -> u64 {
        self.gpu_ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for GpuMr {
    fn drop(&mut self) {
        if !self.mr.is_null() {
            unsafe {
                ibverbs_sys::ibv_dereg_mr(self.mr);
            }
        }
    }
}

/// Pool of pre-registered GPU memory regions for reuse.
pub struct GpuDirectPool {
    queue: ArrayQueue<GpuMr>,
    #[allow(dead_code)]
    ctx: Arc<GpuDirectContext>,
    buf_size: usize,
}

unsafe impl Send for GpuDirectPool {}
unsafe impl Sync for GpuDirectPool {}

impl GpuDirectPool {
    /// Create a pool by registering `count` GPU buffers starting at `gpu_base_ptr`.
    ///
    /// # Safety
    ///
    /// The GPU memory must remain allocated for the lifetime of this pool.
    pub unsafe fn new(
        ctx: Arc<GpuDirectContext>,
        gpu_base_ptr: u64,
        buf_size: usize,
        count: usize,
    ) -> Result<Arc<Self>> {
        let queue = ArrayQueue::new(count);
        for i in 0..count {
            let ptr = gpu_base_ptr + (i * buf_size) as u64;
            let mr = unsafe { ctx.register_gpu_memory(ptr, buf_size)? };
            let _ = queue.push(mr);
        }
        Ok(Arc::new(Self {
            queue,
            ctx,
            buf_size,
        }))
    }

    /// Create an empty pool (no pre-registered buffers).
    pub fn empty(ctx: Arc<GpuDirectContext>) -> Self {
        Self {
            queue: ArrayQueue::new(1),
            ctx,
            buf_size: 0,
        }
    }

    /// Checkout a registered GPU MR from the pool.
    pub fn checkout(self: &Arc<Self>) -> Option<PooledGpuMr> {
        self.queue.pop().map(|mr| PooledGpuMr {
            mr: Some(mr),
            pool: Arc::clone(self),
        })
    }

    pub fn buf_size(&self) -> usize {
        self.buf_size
    }
}

/// A pooled GPU MR that auto-returns to the pool on drop.
pub struct PooledGpuMr {
    mr: Option<GpuMr>,
    pool: Arc<GpuDirectPool>,
}

unsafe impl Send for PooledGpuMr {}
unsafe impl Sync for PooledGpuMr {}

impl PooledGpuMr {
    pub fn mr(&self) -> &GpuMr {
        self.mr.as_ref().expect("GpuMr taken after drop")
    }
}

impl Drop for PooledGpuMr {
    fn drop(&mut self) {
        if let Some(mr) = self.mr.take() {
            let _ = self.pool.queue.push(mr);
        }
    }
}
