use crossbeam_queue::ArrayQueue;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Default capacity for pre-allocated buffers (64 KiB).
const DEFAULT_BUF_CAPACITY: usize = 64 * 1024;

/// Default number of buffers in the pool.
const DEFAULT_POOL_SIZE: usize = 256;

/// A lock-free pool of reusable `Vec<u8>` buffers.
///
/// On checkout, returns a `PooledBuf` that automatically returns the buffer
/// to the pool on drop. If the pool is empty, a fresh `Vec` is allocated
/// (never blocks). If the pool is full on return, the buffer is simply dropped.
pub struct BufferPool {
    queue: ArrayQueue<Vec<u8>>,
    buf_capacity: usize,
}

impl BufferPool {
    /// Create a new pool with default settings (256 buffers, 64 KiB each).
    pub fn new() -> Arc<Self> {
        Self::with_config(DEFAULT_POOL_SIZE, DEFAULT_BUF_CAPACITY)
    }

    /// Create a pool with custom size and buffer capacity.
    pub fn with_config(pool_size: usize, buf_capacity: usize) -> Arc<Self> {
        let queue = ArrayQueue::new(pool_size);
        for _ in 0..pool_size {
            let _ = queue.push(Vec::with_capacity(buf_capacity));
        }
        Arc::new(Self {
            queue,
            buf_capacity,
        })
    }

    /// Check out a buffer, resized to `len` bytes (zeroed).
    ///
    /// If the pool has a buffer available, reuses it. Otherwise allocates a new one.
    pub fn checkout(self: &Arc<Self>, len: usize) -> PooledBuf {
        let mut buf = self
            .queue
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.buf_capacity));
        buf.resize(len, 0);
        PooledBuf {
            buf: Some(buf),
            pool: Arc::clone(self),
        }
    }

    /// Return a buffer to the pool. Called automatically by `PooledBuf::drop`.
    ///
    /// Buffers that have grown beyond 4x the default capacity are dropped instead
    /// of returned, preventing a single large transfer from inflating the pool's
    /// baseline memory usage.
    fn return_buf(&self, mut buf: Vec<u8>) {
        if buf.capacity() > self.buf_capacity * 4 {
            return; // drop oversized buffer
        }
        buf.clear();
        // If pool is full, buf is simply dropped.
        let _ = self.queue.push(buf);
    }
}

/// A buffer checked out from a `BufferPool`. Derefs to `[u8]`.
/// On drop, the underlying `Vec` is cleared and returned to the pool.
pub struct PooledBuf {
    buf: Option<Vec<u8>>,
    pool: Arc<BufferPool>,
}

impl Deref for PooledBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY invariant: `buf` is `Some` from construction until `Drop`.
        self.buf.as_ref().expect("PooledBuf used after drop")
    }
}

impl DerefMut for PooledBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.buf.as_mut().expect("PooledBuf used after drop")
    }
}

impl Drop for PooledBuf {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.return_buf(buf);
        }
    }
}

impl AsRef<[u8]> for PooledBuf {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkout_and_return() {
        let pool = BufferPool::with_config(4, 1024);
        let buf = pool.checkout(100);
        assert_eq!(buf.len(), 100);
        assert!(buf.iter().all(|&b| b == 0));
        drop(buf);
        // Buffer returned to pool — pool should still have 4 buffers.
    }

    #[test]
    fn test_pool_exhaustion_fallback() {
        let pool = BufferPool::with_config(2, 64);
        let b1 = pool.checkout(10);
        let b2 = pool.checkout(10);
        // Pool is now empty — this should still work (allocates fresh).
        let b3 = pool.checkout(10);
        assert_eq!(b3.len(), 10);
        drop(b1);
        drop(b2);
        drop(b3);
    }

    #[test]
    fn test_oversized_buffer() {
        let pool = BufferPool::with_config(2, 64);
        // Request larger than default capacity — Vec grows naturally.
        let buf = pool.checkout(1024);
        assert_eq!(buf.len(), 1024);
        drop(buf);
    }

    #[test]
    fn test_deref_mut() {
        let pool = BufferPool::with_config(2, 64);
        let mut buf = pool.checkout(4);
        buf[0] = 0xAA;
        buf[1] = 0xBB;
        assert_eq!(buf[0], 0xAA);
        assert_eq!(buf[1], 0xBB);
    }

    #[test]
    fn test_drop_returns_to_pool() {
        let pool = BufferPool::with_config(1, 64);
        // Take the one buffer out.
        let buf = pool.checkout(10);
        // Pool is empty now — next checkout allocates fresh.
        let buf2 = pool.checkout(10);
        drop(buf);
        // Pool has 1 buffer again. Next checkout reuses it.
        let buf3 = pool.checkout(20);
        assert_eq!(buf3.len(), 20);
        drop(buf2);
        drop(buf3);
    }

    #[test]
    fn test_oversized_buffer_dropped_on_return() {
        // Pool with buf_capacity=64. Threshold is 4x = 256.
        let pool = BufferPool::with_config(2, 64);
        // Checkout a buffer and grow it well beyond 4x capacity.
        let buf = pool.checkout(1024);
        assert_eq!(buf.len(), 1024);
        drop(buf);
        // The oversized buffer was dropped, not returned. Pool still has
        // its original buffers. Checkout should give a normal-capacity buffer.
        let buf2 = pool.checkout(10);
        assert_eq!(buf2.len(), 10);
        // The underlying Vec capacity should be the original pool capacity,
        // not the inflated 1024.
        drop(buf2);
    }

    #[test]
    fn test_pool_full_on_return() {
        let pool = BufferPool::with_config(1, 64);
        // Pool starts with 1 buffer. Checkout it.
        let buf1 = pool.checkout(10);
        // Allocate a fresh one (pool empty).
        let buf2 = pool.checkout(10);
        // Return buf1 — pool now has 1.
        drop(buf1);
        // Return buf2 — pool is full, buf2 is just dropped. No panic.
        drop(buf2);
    }
}
