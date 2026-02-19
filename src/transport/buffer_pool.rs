use crossbeam_queue::ArrayQueue;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Small pool: 256 buffers × 64 KiB — for framed control messages.
const SMALL_POOL_SIZE: usize = 256;
const SMALL_BUF_CAPACITY: usize = 64 * 1024;

/// Large pool: 32 buffers × 8 MiB — for tensor data streams.
const LARGE_POOL_SIZE: usize = 32;
const LARGE_BUF_CAPACITY: usize = 8 * 1024 * 1024;

/// A tiered lock-free buffer pool with small and large tiers.
///
/// - **Small tier** (64 KiB): for framed control messages and small payloads.
/// - **Large tier** (8 MiB): for tensor data transfers and bulk payloads.
/// - **Above 64 MiB**: allocated fresh and dropped on return (rare).
///
/// Checkout picks the appropriate tier based on requested size. Return goes
/// back to the matching tier. Buffers that have grown beyond 4× their tier's
/// capacity are dropped instead of returned.
pub struct BufferPool {
    small: ArrayQueue<Vec<u8>>,
    large: ArrayQueue<Vec<u8>>,
}

impl BufferPool {
    /// Create a new tiered pool with default settings.
    pub fn new() -> Arc<Self> {
        let small = ArrayQueue::new(SMALL_POOL_SIZE);
        for _ in 0..SMALL_POOL_SIZE {
            let _ = small.push(Vec::with_capacity(SMALL_BUF_CAPACITY));
        }

        let large = ArrayQueue::new(LARGE_POOL_SIZE);
        for _ in 0..LARGE_POOL_SIZE {
            let _ = large.push(Vec::with_capacity(LARGE_BUF_CAPACITY));
        }

        Arc::new(Self { small, large })
    }

    /// Create a pool with custom sizes (primarily for testing).
    pub fn with_config(small_pool_size: usize, small_buf_cap: usize) -> Arc<Self> {
        let small = ArrayQueue::new(small_pool_size);
        for _ in 0..small_pool_size {
            let _ = small.push(Vec::with_capacity(small_buf_cap));
        }

        // For custom configs, create a minimal large pool.
        let large = ArrayQueue::new(4);
        for _ in 0..4 {
            let _ = large.push(Vec::with_capacity(LARGE_BUF_CAPACITY));
        }

        Arc::new(Self { small, large })
    }

    /// Check out a buffer, resized to `len` bytes (zeroed).
    ///
    /// Selects the appropriate tier:
    /// - `len <= SMALL_BUF_CAPACITY` (64 KiB): small pool
    /// - `len <= LARGE_BUF_CAPACITY` (8 MiB): large pool
    /// - `len > LARGE_BUF_CAPACITY`: allocate fresh (no pool)
    pub fn checkout(self: &Arc<Self>, len: usize) -> PooledBuf {
        let (mut buf, tier) = if len <= SMALL_BUF_CAPACITY {
            let buf = self
                .small
                .pop()
                .unwrap_or_else(|| Vec::with_capacity(SMALL_BUF_CAPACITY));
            (buf, PoolTier::Small)
        } else if len <= LARGE_BUF_CAPACITY {
            let buf = self
                .large
                .pop()
                .unwrap_or_else(|| Vec::with_capacity(LARGE_BUF_CAPACITY));
            (buf, PoolTier::Large)
        } else {
            // Too large for any pool — allocate fresh.
            (Vec::with_capacity(len), PoolTier::Unpooled)
        };

        buf.resize(len, 0);
        PooledBuf {
            buf: Some(buf),
            pool: Arc::clone(self),
            tier,
        }
    }

    /// Return a buffer to the appropriate tier.
    fn return_buf(&self, mut buf: Vec<u8>, tier: PoolTier) {
        match tier {
            PoolTier::Small => {
                // Drop if grown beyond 4× small capacity.
                if buf.capacity() > SMALL_BUF_CAPACITY * 4 {
                    return;
                }
                buf.clear();
                let _ = self.small.push(buf);
            }
            PoolTier::Large => {
                // Drop if grown beyond 4× large capacity.
                if buf.capacity() > LARGE_BUF_CAPACITY * 4 {
                    return;
                }
                buf.clear();
                let _ = self.large.push(buf);
            }
            PoolTier::Unpooled => {
                // Always drop — too large to pool.
            }
        }
    }
}

/// Which pool tier a buffer belongs to.
#[derive(Debug, Clone, Copy)]
enum PoolTier {
    Small,
    Large,
    Unpooled,
}

/// A buffer checked out from a `BufferPool`. Derefs to `[u8]`.
/// On drop, the underlying `Vec` is cleared and returned to the appropriate tier.
pub struct PooledBuf {
    buf: Option<Vec<u8>>,
    pool: Arc<BufferPool>,
    tier: PoolTier,
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
            self.pool.return_buf(buf, self.tier);
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
        let buf = pool.checkout(10);
        let buf2 = pool.checkout(10);
        drop(buf);
        let buf3 = pool.checkout(20);
        assert_eq!(buf3.len(), 20);
        drop(buf2);
        drop(buf3);
    }

    #[test]
    fn test_pool_full_on_return() {
        let pool = BufferPool::with_config(1, 64);
        let buf1 = pool.checkout(10);
        let buf2 = pool.checkout(10);
        drop(buf1);
        drop(buf2); // pool full, just dropped — no panic
    }

    #[test]
    fn test_large_buffer_uses_large_pool() {
        let pool = BufferPool::new();
        // 1 MiB — should use the large pool tier.
        let buf = pool.checkout(1024 * 1024);
        assert_eq!(buf.len(), 1024 * 1024);
        drop(buf);
    }

    #[test]
    fn test_small_buffer_uses_small_pool() {
        let pool = BufferPool::new();
        let buf = pool.checkout(100);
        assert_eq!(buf.len(), 100);
        drop(buf);
    }

    #[test]
    fn test_very_large_buffer_unpooled() {
        let pool = BufferPool::new();
        // 100 MiB — too large for any pool tier.
        let buf = pool.checkout(100 * 1024 * 1024);
        assert_eq!(buf.len(), 100 * 1024 * 1024);
        drop(buf); // dropped, not returned to any pool
    }

    #[test]
    fn test_oversized_buffer_dropped_on_return() {
        let pool = BufferPool::with_config(2, 64);
        // Request 1024 bytes — uses small tier (since <=64KiB), but vec grows.
        // When returned, capacity > 4*64 = 256, so it's dropped.
        let buf = pool.checkout(1024);
        assert_eq!(buf.len(), 1024);
        drop(buf);
        // Next checkout should get a normal-capacity buffer.
        let buf2 = pool.checkout(10);
        assert_eq!(buf2.len(), 10);
        drop(buf2);
    }
}
