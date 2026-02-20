use crossbeam_queue::ArrayQueue;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Default small buffer capacity: 64 KiB.
const SMALL_BUF_CAPACITY: usize = 64 * 1024;

/// Default large buffer capacity: 8 MiB.
const LARGE_BUF_CAPACITY: usize = 8 * 1024 * 1024;

/// Default huge buffer capacity: 64 MiB.
const HUGE_BUF_CAPACITY: usize = 64 * 1024 * 1024;

/// Default giant buffer capacity: 256 MiB.
const GIANT_BUF_CAPACITY: usize = 256 * 1024 * 1024;

/// Workload profile that determines pool tier sizes.
///
/// Training and inference have fundamentally different buffer usage patterns.
/// Choosing the right profile avoids wasting memory (inference) or thrashing
/// allocations (training).
///
/// All profiles use **lazy allocation**: queues are created at construction
/// with the specified capacity limits, but buffers are only allocated on first
/// checkout and recycled on return. No memory is consumed until buffers are
/// actually needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolProfile {
    /// Optimized for distributed training (gradient allreduce, broadcast).
    ///
    /// Queue capacities (lazy, allocated on demand):
    /// - Small: up to 256 × 64 KiB (control messages)
    /// - Large: up to 32 × 8 MiB (embedding gradients)
    /// - Huge: up to 16 × 64 MiB (ring allreduce chunks)
    /// - Giant: up to 4 × 256 MiB (halving-doubling, broadcasts)
    Training,

    /// Optimized for distributed inference (pipeline parallelism, KV cache
    /// migration, MoE expert dispatch).
    ///
    /// Queue capacities (lazy, allocated on demand):
    /// - Small: up to 256 × 64 KiB (control, routing decisions)
    /// - Large: up to 16 × 8 MiB (MoE token dispatch, small activations)
    /// - Huge: up to 4 × 64 MiB (KV cache layers, pipeline activations)
    /// - Giant: 0 (no gradient chunks during inference)
    Inference,
}

/// Configuration for a single buffer tier.
#[derive(Debug, Clone, Copy)]
pub struct TierConfig {
    /// Maximum number of buffers to keep in this tier's queue.
    pub count: usize,
    /// Capacity of each buffer in bytes.
    pub capacity: usize,
}

/// Builder for custom buffer pool configurations.
///
/// Use this when the built-in [`PoolProfile`] variants don't fit your workload.
///
/// # Example
///
/// ```
/// use nexar::transport::buffer_pool::PoolBuilder;
/// use nexar::transport::buffer_pool::TierConfig;
///
/// let pool = PoolBuilder::new()
///     .small(TierConfig { count: 64, capacity: 32 * 1024 })
///     .large(TierConfig { count: 8, capacity: 4 * 1024 * 1024 })
///     .huge(TierConfig { count: 2, capacity: 32 * 1024 * 1024 })
///     .build();
/// ```
pub struct PoolBuilder {
    small: TierConfig,
    large: TierConfig,
    huge: TierConfig,
    giant: TierConfig,
}

impl PoolBuilder {
    /// Create a builder with minimal defaults (all tiers have count=1).
    pub fn new() -> Self {
        Self {
            small: TierConfig {
                count: 1,
                capacity: SMALL_BUF_CAPACITY,
            },
            large: TierConfig {
                count: 1,
                capacity: LARGE_BUF_CAPACITY,
            },
            huge: TierConfig {
                count: 1,
                capacity: HUGE_BUF_CAPACITY,
            },
            giant: TierConfig {
                count: 0,
                capacity: GIANT_BUF_CAPACITY,
            },
        }
    }

    /// Configure the small tier (default: 64 KiB buffers).
    pub fn small(mut self, config: TierConfig) -> Self {
        self.small = config;
        self
    }

    /// Configure the large tier (default: 8 MiB buffers).
    pub fn large(mut self, config: TierConfig) -> Self {
        self.large = config;
        self
    }

    /// Configure the huge tier (default: 64 MiB buffers).
    pub fn huge(mut self, config: TierConfig) -> Self {
        self.huge = config;
        self
    }

    /// Configure the giant tier (default: 256 MiB buffers).
    pub fn giant(mut self, config: TierConfig) -> Self {
        self.giant = config;
        self
    }

    /// Build the pool. All tiers start empty (lazy allocation).
    pub fn build(self) -> Arc<BufferPool> {
        BufferPool::from_tier_configs(self.small, self.large, self.huge, self.giant)
    }
}

impl Default for PoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A tiered lock-free buffer pool with small, large, huge, and giant tiers.
///
/// Buffers are allocated **lazily**: the pool starts empty and buffers are
/// created on first checkout. When returned, they are recycled into the
/// appropriate tier queue for reuse. This means zero memory is consumed
/// until buffers are actually needed.
///
/// Use [`PoolProfile`] for preset configurations or [`PoolBuilder`] for
/// full control over tier counts and capacities.
///
/// Checkout picks the appropriate tier based on requested size. Return goes
/// back to the matching tier. Buffers that have grown beyond 4× their tier's
/// capacity are dropped instead of returned.
pub struct BufferPool {
    small: ArrayQueue<Vec<u8>>,
    large: ArrayQueue<Vec<u8>>,
    huge: ArrayQueue<Vec<u8>>,
    giant: ArrayQueue<Vec<u8>>,
    small_cap: usize,
    large_cap: usize,
    huge_cap: usize,
    giant_cap: usize,
}

impl BufferPool {
    /// Create a pool with the default training profile (lazy allocation).
    ///
    /// Equivalent to `BufferPool::with_profile(PoolProfile::Training)`.
    pub fn new() -> Arc<Self> {
        Self::with_profile(PoolProfile::Training)
    }

    /// Create a pool sized for the given workload profile (lazy allocation).
    pub fn with_profile(profile: PoolProfile) -> Arc<Self> {
        let (small_count, large_count, huge_count, giant_count) = match profile {
            PoolProfile::Training => (256, 32, 16, 4),
            PoolProfile::Inference => (256, 16, 4, 0),
        };
        Self::from_tier_configs(
            TierConfig {
                count: small_count,
                capacity: SMALL_BUF_CAPACITY,
            },
            TierConfig {
                count: large_count,
                capacity: LARGE_BUF_CAPACITY,
            },
            TierConfig {
                count: huge_count,
                capacity: HUGE_BUF_CAPACITY,
            },
            TierConfig {
                count: giant_count,
                capacity: GIANT_BUF_CAPACITY,
            },
        )
    }

    /// Create a pool with custom small tier sizes (primarily for testing).
    ///
    /// Large, huge, and giant tiers use minimal sizes to avoid excessive
    /// memory usage in test environments.
    pub fn with_config(small_pool_size: usize, small_buf_cap: usize) -> Arc<Self> {
        Self::from_tier_configs(
            TierConfig {
                count: small_pool_size,
                capacity: small_buf_cap,
            },
            TierConfig {
                count: 4,
                capacity: LARGE_BUF_CAPACITY,
            },
            TierConfig {
                count: 2,
                capacity: HUGE_BUF_CAPACITY,
            },
            TierConfig {
                count: 1,
                capacity: GIANT_BUF_CAPACITY,
            },
        )
    }

    fn from_tier_configs(
        small: TierConfig,
        large: TierConfig,
        huge: TierConfig,
        giant: TierConfig,
    ) -> Arc<Self> {
        Arc::new(Self {
            small: ArrayQueue::new(small.count.max(1)),
            large: ArrayQueue::new(large.count.max(1)),
            huge: ArrayQueue::new(huge.count.max(1)),
            giant: ArrayQueue::new(giant.count.max(1)),
            small_cap: small.capacity,
            large_cap: large.capacity,
            huge_cap: huge.capacity,
            giant_cap: giant.capacity,
        })
    }

    /// Check out a buffer, resized to `len` bytes (zeroed).
    ///
    /// Selects the appropriate tier based on the configured tier capacities.
    /// If the tier queue is empty, a fresh buffer is allocated (lazy).
    /// If `len` exceeds all tier capacities, an unpooled buffer is allocated.
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
        if len <= self.small_cap {
            (Some(&self.small), PoolTier::Small, self.small_cap)
        } else if len <= self.large_cap {
            (Some(&self.large), PoolTier::Large, self.large_cap)
        } else if len <= self.huge_cap {
            (Some(&self.huge), PoolTier::Huge, self.huge_cap)
        } else if len <= self.giant_cap {
            (Some(&self.giant), PoolTier::Giant, self.giant_cap)
        } else {
            (None, PoolTier::Unpooled, len)
        }
    }

    /// Return a buffer to the appropriate tier.
    fn return_buf(&self, mut buf: Vec<u8>, tier: PoolTier) {
        let (queue, max_cap) = match tier {
            PoolTier::Small => (Some(&self.small), self.small_cap * 4),
            PoolTier::Large => (Some(&self.large), self.large_cap * 4),
            PoolTier::Huge => (Some(&self.huge), self.huge_cap * 4),
            PoolTier::Giant => (Some(&self.giant), self.giant_cap * 4),
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

    #[test]
    fn test_lazy_allocation_no_upfront_memory() {
        // Pool starts empty — no buffers pre-allocated.
        let pool = BufferPool::with_profile(PoolProfile::Training);
        // First checkout allocates on demand.
        let buf = pool.checkout(1024);
        assert_eq!(buf.len(), 1024);
        // Return populates the pool for reuse.
        drop(buf);
        // Second checkout reuses the returned buffer (no new allocation).
        let buf2 = pool.checkout(512);
        assert_eq!(buf2.len(), 512);
        drop(buf2);
    }

    #[test]
    fn test_pool_builder_custom_tiers() {
        let pool = PoolBuilder::new()
            .small(TierConfig {
                count: 4,
                capacity: 1024,
            })
            .large(TierConfig {
                count: 2,
                capacity: 1024 * 1024,
            })
            .huge(TierConfig {
                count: 1,
                capacity: 16 * 1024 * 1024,
            })
            .giant(TierConfig {
                count: 0,
                capacity: 64 * 1024 * 1024,
            })
            .build();

        // Small tier: up to 1024 bytes.
        let buf = pool.checkout(500);
        assert_eq!(buf.len(), 500);
        drop(buf);

        // Large tier: up to 1 MiB.
        let buf = pool.checkout(512 * 1024);
        assert_eq!(buf.len(), 512 * 1024);
        drop(buf);

        // Beyond giant capacity → unpooled.
        let buf = pool.checkout(128 * 1024 * 1024);
        assert_eq!(buf.len(), 128 * 1024 * 1024);
        drop(buf);
    }

    #[test]
    fn test_pool_builder_default() {
        let pool = PoolBuilder::default().build();
        let buf = pool.checkout(100);
        assert_eq!(buf.len(), 100);
        drop(buf);
    }
}
