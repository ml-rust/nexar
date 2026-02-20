use crossbeam_queue::ArrayQueue;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Small pool: 256 buffers × 64 KiB — for framed control messages.
const SMALL_POOL_SIZE: usize = 256;
const SMALL_BUF_CAPACITY: usize = 64 * 1024;

/// Large pool capacity: 8 MiB.
const LARGE_BUF_CAPACITY: usize = 8 * 1024 * 1024;

/// Huge pool capacity: 64 MiB.
const HUGE_BUF_CAPACITY: usize = 64 * 1024 * 1024;

/// Giant pool capacity: 256 MiB.
const GIANT_BUF_CAPACITY: usize = 256 * 1024 * 1024;

/// Workload profile that determines pool tier sizes.
///
/// Training and inference have fundamentally different buffer usage patterns.
/// Choosing the right profile avoids wasting memory (inference) or thrashing
/// allocations (training).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolProfile {
    /// Optimized for distributed training (gradient allreduce, broadcast).
    ///
    /// Pre-allocates ~2.3 GiB:
    /// - Small: 256 × 64 KiB = 16 MiB (control messages)
    /// - Large: 32 × 8 MiB = 256 MiB (embedding gradients)
    /// - Huge: 16 × 64 MiB = 1024 MiB (ring allreduce chunks)
    /// - Giant: 4 × 256 MiB = 1024 MiB (halving-doubling, broadcasts)
    Training,

    /// Optimized for distributed inference (pipeline parallelism, KV cache
    /// migration, MoE expert dispatch).
    ///
    /// Pre-allocates ~296 MiB:
    /// - Small: 256 × 64 KiB = 16 MiB (control, routing decisions)
    /// - Large: 16 × 8 MiB = 128 MiB (MoE token dispatch, small activations)
    /// - Huge: 4 × 64 MiB = 256 MiB (KV cache layers, pipeline activations)
    /// - Giant: 0 (no gradient chunks during inference)
    ///
    /// Multiple inference workers can coexist on the same node without
    /// exhausting memory on empty buffer pools.
    Inference,
}

/// A tiered lock-free buffer pool with small, large, huge, and giant tiers.
///
/// Use [`PoolProfile`] to select the right sizing for your workload:
/// - [`PoolProfile::Training`]: 2.3 GiB pre-allocated, handles 405B gradient chunks.
/// - [`PoolProfile::Inference`]: 296 MiB pre-allocated, handles KV cache and activations.
///
/// Checkout picks the appropriate tier based on requested size. Return goes
/// back to the matching tier. Buffers that have grown beyond 4× their tier's
/// capacity are dropped instead of returned.
pub struct BufferPool {
    small: ArrayQueue<Vec<u8>>,
    large: ArrayQueue<Vec<u8>>,
    huge: ArrayQueue<Vec<u8>>,
    giant: ArrayQueue<Vec<u8>>,
}

impl BufferPool {
    /// Create a pool with the default training profile.
    ///
    /// Equivalent to `BufferPool::with_profile(PoolProfile::Training)`.
    pub fn new() -> Arc<Self> {
        Self::with_profile(PoolProfile::Training)
    }

    /// Create a pool sized for the given workload profile.
    pub fn with_profile(profile: PoolProfile) -> Arc<Self> {
        let (large_count, huge_count, giant_count) = match profile {
            PoolProfile::Training => (32, 16, 4),
            PoolProfile::Inference => (16, 4, 0),
        };
        Self::with_tier_sizes(
            SMALL_POOL_SIZE,
            SMALL_BUF_CAPACITY,
            large_count,
            huge_count,
            giant_count,
        )
    }

    /// Create a pool with custom small tier sizes (primarily for testing).
    ///
    /// Large, huge, and giant tiers use minimal sizes to avoid excessive
    /// memory usage in test environments.
    pub fn with_config(small_pool_size: usize, small_buf_cap: usize) -> Arc<Self> {
        Self::with_tier_sizes(small_pool_size, small_buf_cap, 4, 2, 1)
    }

    fn with_tier_sizes(
        small_count: usize,
        small_cap: usize,
        large_count: usize,
        huge_count: usize,
        giant_count: usize,
    ) -> Arc<Self> {
        fn fill_tier(count: usize, capacity: usize) -> ArrayQueue<Vec<u8>> {
            let queue = ArrayQueue::new(count.max(1));
            for _ in 0..count {
                let _ = queue.push(Vec::with_capacity(capacity));
            }
            queue
        }

        Arc::new(Self {
            small: fill_tier(small_count, small_cap),
            large: fill_tier(large_count, LARGE_BUF_CAPACITY),
            huge: fill_tier(huge_count, HUGE_BUF_CAPACITY),
            giant: fill_tier(giant_count, GIANT_BUF_CAPACITY),
        })
    }

    /// Check out a buffer, resized to `len` bytes (zeroed).
    ///
    /// Selects the appropriate tier:
    /// - `len <= 64 KiB`: small pool
    /// - `len <= 8 MiB`: large pool
    /// - `len <= 64 MiB`: huge pool
    /// - `len <= 256 MiB`: giant pool
    /// - `len > 256 MiB`: allocate fresh (no pool)
    pub fn checkout(self: &Arc<Self>, len: usize) -> PooledBuf {
        let (queue, tier, capacity) = self.tier_for_size(len);
        let mut buf = match queue {
            Some(q) => q.pop().unwrap_or_else(|| Vec::with_capacity(capacity)),
            None => Vec::with_capacity(len),
        };
        buf.resize(len, 0);
        PooledBuf {
            buf: Some(buf),
            pool: Arc::clone(self),
            tier,
        }
    }

    /// Select the pool tier for a given buffer size.
    fn tier_for_size(&self, len: usize) -> (Option<&ArrayQueue<Vec<u8>>>, PoolTier, usize) {
        if len <= SMALL_BUF_CAPACITY {
            (Some(&self.small), PoolTier::Small, SMALL_BUF_CAPACITY)
        } else if len <= LARGE_BUF_CAPACITY {
            (Some(&self.large), PoolTier::Large, LARGE_BUF_CAPACITY)
        } else if len <= HUGE_BUF_CAPACITY {
            (Some(&self.huge), PoolTier::Huge, HUGE_BUF_CAPACITY)
        } else if len <= GIANT_BUF_CAPACITY {
            (Some(&self.giant), PoolTier::Giant, GIANT_BUF_CAPACITY)
        } else {
            (None, PoolTier::Unpooled, len)
        }
    }

    /// Return a buffer to the appropriate tier.
    fn return_buf(&self, mut buf: Vec<u8>, tier: PoolTier) {
        let (queue, max_cap) = match tier {
            PoolTier::Small => (Some(&self.small), SMALL_BUF_CAPACITY * 4),
            PoolTier::Large => (Some(&self.large), LARGE_BUF_CAPACITY * 4),
            PoolTier::Huge => (Some(&self.huge), HUGE_BUF_CAPACITY * 4),
            PoolTier::Giant => (Some(&self.giant), GIANT_BUF_CAPACITY * 4),
            PoolTier::Unpooled => (None, 0),
        };
        if let Some(q) = queue
            && buf.capacity() <= max_cap
        {
            buf.clear();
            let _ = q.push(buf);
        }
    }
}

/// Which pool tier a buffer belongs to.
#[derive(Debug, Clone, Copy)]
enum PoolTier {
    Small,
    Large,
    Huge,
    Giant,
    Unpooled,
}

/// A buffer checked out from a `BufferPool`. Derefs to `[u8]`.
/// On drop, the underlying `Vec` is cleared and returned to the appropriate tier.
pub struct PooledBuf {
    buf: Option<Vec<u8>>,
    pool: Arc<BufferPool>,
    tier: PoolTier,
}

impl PooledBuf {
    /// Wrap an externally-received `Vec<u8>` as a `PooledBuf`.
    ///
    /// The buffer will be returned to the pool's appropriate tier on drop.
    /// Useful for wrapping data received via non-QUIC transports (e.g., RDMA).
    pub fn from_vec(v: Vec<u8>, pool: Arc<BufferPool>) -> Self {
        let len = v.len();
        let (_, tier, _) = pool.tier_for_size(len);
        Self {
            buf: Some(v),
            pool,
            tier,
        }
    }
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
    fn test_huge_buffer_uses_huge_pool() {
        let pool = BufferPool::new();
        // 32 MiB — should use the huge pool tier.
        let buf = pool.checkout(32 * 1024 * 1024);
        assert_eq!(buf.len(), 32 * 1024 * 1024);
        drop(buf);
    }

    #[test]
    fn test_giant_buffer_uses_giant_pool() {
        let pool = BufferPool::new();
        // 128 MiB — should use the giant pool tier.
        let buf = pool.checkout(128 * 1024 * 1024);
        assert_eq!(buf.len(), 128 * 1024 * 1024);
        drop(buf);
    }

    #[test]
    fn test_very_large_buffer_unpooled() {
        let pool = BufferPool::new();
        // 512 MiB — too large for any pool tier.
        let buf = pool.checkout(512 * 1024 * 1024);
        assert_eq!(buf.len(), 512 * 1024 * 1024);
        drop(buf); // dropped, not returned to any pool
    }

    #[test]
    fn test_oversized_buffer_dropped_on_return() {
        let pool = BufferPool::with_config(2, 64);
        let buf = pool.checkout(1024);
        assert_eq!(buf.len(), 1024);
        drop(buf);
        let buf2 = pool.checkout(10);
        assert_eq!(buf2.len(), 10);
        drop(buf2);
    }

    /// Verify checkout/return at workload-realistic sizes for both profiles.
    fn assert_checkout(pool: &Arc<BufferPool>, sizes: &[usize]) {
        let bufs: Vec<_> = sizes
            .iter()
            .map(|&s| {
                let b = pool.checkout(s);
                assert_eq!(b.len(), s);
                b
            })
            .collect();
        drop(bufs);
    }

    #[test]
    fn test_training_workload_sizes() {
        let pool = BufferPool::new();
        let m = 1024 * 1024;
        // 3 concurrent 14 MiB ring allreduce chunks, 208 MiB halving-doubling, 48 MiB broadcast.
        assert_checkout(&pool, &[14 * m, 14 * m, 14 * m, 208 * m, 48 * m]);
    }

    #[test]
    fn test_inference_workload_sizes() {
        let pool = BufferPool::with_profile(PoolProfile::Inference);
        let m = 1024 * 1024;
        // 16 MiB KV layer, 2 concurrent KV transfers, 64 MiB pipeline activation,
        // 4 concurrent 1 MiB MoE dispatches, 128 MiB large activation fallback.
        assert_checkout(&pool, &[16 * m, 16 * m, 64 * m, m, m, m, m, 128 * m]);
    }
}
