//! RDMA memory region pool.
//!
//! RDMA requires all send/recv buffers to be registered with the NIC's
//! protection domain. Registration pins memory and is expensive, so we
//! pre-allocate a pool of registered buffers and reuse them.

use crate::error::Result;
use crate::transport::rdma::RdmaContext;
use crossbeam_queue::ArrayQueue;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// A pool of pre-registered RDMA memory regions for zero-allocation send/recv.
pub struct RdmaMemoryPool {
    ctx: Arc<RdmaContext>,
    buf_size: usize,
    queue: ArrayQueue<ibverbs::MemoryRegion<u8>>,
}

// Safety: MemoryRegion<u8> is Send+Sync per ibverbs docs, Arc is Send+Sync.
unsafe impl Send for RdmaMemoryPool {}
unsafe impl Sync for RdmaMemoryPool {}

impl RdmaMemoryPool {
    /// Create a pool with `pool_size` pre-registered buffers of `buf_size` bytes.
    pub fn new(ctx: &Arc<RdmaContext>, pool_size: usize, buf_size: usize) -> Result<Arc<Self>> {
        let queue = ArrayQueue::new(pool_size);
        for _ in 0..pool_size {
            let mr = ctx.allocate(buf_size)?;
            let _ = queue.push(mr);
        }
        Ok(Arc::new(Self {
            ctx: Arc::clone(ctx),
            buf_size,
            queue,
        }))
    }

    /// Checkout a registered buffer from the pool.
    ///
    /// If empty, allocates a fresh buffer from the protection domain.
    pub fn checkout(self: &Arc<Self>) -> Result<RdmaPooledBuf> {
        let mr = match self.queue.pop() {
            Some(mr) => mr,
            None => self.ctx.allocate(self.buf_size)?,
        };
        Ok(RdmaPooledBuf {
            mr: Some(mr),
            pool: Arc::clone(self),
        })
    }

    fn return_buf(&self, mr: ibverbs::MemoryRegion<u8>) {
        // If full, MR is dropped (deregistered).
        let _ = self.queue.push(mr);
    }

    /// The size of each buffer in the pool.
    pub fn buf_size(&self) -> usize {
        self.buf_size
    }
}

/// A pooled RDMA memory region that auto-returns to the pool on drop.
pub struct RdmaPooledBuf {
    mr: Option<ibverbs::MemoryRegion<u8>>,
    pool: Arc<RdmaMemoryPool>,
}

impl RdmaPooledBuf {
    /// Access the underlying `MemoryRegion` for use with `post_send`/`post_receive`.
    pub fn mr(&self) -> &ibverbs::MemoryRegion<u8> {
        self.mr.as_ref().expect("MR taken after drop")
    }

    /// Access the underlying `MemoryRegion` mutably.
    pub fn mr_mut(&mut self) -> &mut ibverbs::MemoryRegion<u8> {
        self.mr.as_mut().expect("MR taken after drop")
    }
}

impl Deref for RdmaPooledBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.mr()
    }
}

impl DerefMut for RdmaPooledBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.mr_mut()
    }
}

impl Drop for RdmaPooledBuf {
    fn drop(&mut self) {
        if let Some(mr) = self.mr.take() {
            self.pool.return_buf(mr);
        }
    }
}
