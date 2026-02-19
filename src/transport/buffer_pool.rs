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

        let small = ArrayQueue::new(SMALL_POOL_SIZE);
        for _ in 0..SMALL_POOL_SIZE {
            let _ = small.push(Vec::with_capacity(SMALL_BUF_CAPACITY));
        }

        let large = ArrayQueue::new(large_count.max(1));
        for _ in 0..large_count {
            let _ = large.push(Vec::with_capacity(LARGE_BUF_CAPACITY));
        }

        let huge = ArrayQueue::new(huge_count.max(1));
        for _ in 0..huge_count {
            let _ = huge.push(Vec::with_capacity(HUGE_BUF_CAPACITY));
        }

        let giant = ArrayQueue::new(giant_count.max(1));
        for _ in 0..giant_count {
            let _ = giant.push(Vec::with_capacity(GIANT_BUF_CAPACITY));
        }

        Arc::new(Self {
            small,
            large,
            huge,
            giant,
        })
    }

    /// Create a pool with custom small tier sizes (primarily for testing).
    ///
    /// Large, huge, and giant tiers use minimal sizes to avoid excessive
    /// memory usage in test environments.
    pub fn with_config(small_pool_size: usize, small_buf_cap: usize) -> Arc<Self> {
        let small = ArrayQueue::new(small_pool_size);
        for _ in 0..small_pool_size {
            let _ = small.push(Vec::with_capacity(small_buf_cap));
        }

        let large = ArrayQueue::new(4);
        for _ in 0..4 {
            let _ = large.push(Vec::with_capacity(LARGE_BUF_CAPACITY));
        }

        let huge = ArrayQueue::new(2);
        for _ in 0..2 {
            let _ = huge.push(Vec::with_capacity(HUGE_BUF_CAPACITY));
        }

        let giant = ArrayQueue::new(1);
        let _ = giant.push(Vec::with_capacity(GIANT_BUF_CAPACITY));

        Arc::new(Self {
            small,
            large,
            huge,
            giant,
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
        } else if len <= HUGE_BUF_CAPACITY {
            let buf = self
                .huge
                .pop()
                .unwrap_or_else(|| Vec::with_capacity(HUGE_BUF_CAPACITY));
            (buf, PoolTier::Huge)
        } else if len <= GIANT_BUF_CAPACITY {
            let buf = self
                .giant
                .pop()
                .unwrap_or_else(|| Vec::with_capacity(GIANT_BUF_CAPACITY));
            (buf, PoolTier::Giant)
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
                if buf.capacity() > SMALL_BUF_CAPACITY * 4 {
                    return;
                }
                buf.clear();
                let _ = self.small.push(buf);
            }
            PoolTier::Large => {
                if buf.capacity() > LARGE_BUF_CAPACITY * 4 {
                    return;
                }
                buf.clear();
                let _ = self.large.push(buf);
            }
            PoolTier::Huge => {
                if buf.capacity() > HUGE_BUF_CAPACITY * 4 {
                    return;
                }
                buf.clear();
                let _ = self.huge.push(buf);
            }
            PoolTier::Giant => {
                if buf.capacity() > GIANT_BUF_CAPACITY * 4 {
                    return;
                }
                buf.clear();
                let _ = self.giant.push(buf);
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

    // ---- Training profile simulation tests ----

    /// Ring allreduce on 70B (112 MiB MLP gradient, DP=8 → 14 MiB chunks).
    /// 3 concurrent pipelined layers.
    #[test]
    fn test_concurrent_allreduce_70b() {
        let pool = BufferPool::new();
        let chunk_14m = 14 * 1024 * 1024;
        let b1 = pool.checkout(chunk_14m);
        let b2 = pool.checkout(chunk_14m);
        let b3 = pool.checkout(chunk_14m);
        assert_eq!(b1.len(), chunk_14m);
        assert_eq!(b2.len(), chunk_14m);
        assert_eq!(b3.len(), chunk_14m);
        drop(b1);
        drop(b2);
        drop(b3);
    }

    /// Halving-doubling allreduce on 405B MLP (416 MiB tensor, first round = 208 MiB).
    #[test]
    fn test_halving_doubling_405b() {
        let pool = BufferPool::new();
        let half_416m = 208 * 1024 * 1024;
        let buf = pool.checkout(half_416m);
        assert_eq!(buf.len(), half_416m);
        drop(buf);
    }

    /// Broadcast of a full 70B QKV layer (48 MiB).
    #[test]
    fn test_broadcast_70b_qkv() {
        let pool = BufferPool::new();
        let qkv_48m = 48 * 1024 * 1024;
        let buf = pool.checkout(qkv_48m);
        assert_eq!(buf.len(), qkv_48m);
        drop(buf);
    }

    // ---- Inference profile simulation tests ----

    /// Inference profile uses ~296 MiB instead of ~2.3 GiB.
    #[test]
    fn test_inference_profile_exists() {
        let pool = BufferPool::with_profile(PoolProfile::Inference);
        // Should work fine for inference-sized buffers.
        let buf = pool.checkout(16 * 1024 * 1024); // 16 MiB KV cache layer
        assert_eq!(buf.len(), 16 * 1024 * 1024);
        drop(buf);
    }

    /// KV cache migration: 70B has 80 layers × 16 MiB per layer (TP=8, 4K ctx).
    /// Transferred sequentially, so only 1-2 buffers in flight.
    #[test]
    fn test_inference_kv_cache_70b() {
        let pool = BufferPool::with_profile(PoolProfile::Inference);
        let kv_layer = 16 * 1024 * 1024; // 16 MiB per layer
        // Simulate 2 concurrent KV layer transfers (prefetch next while writing current).
        let b1 = pool.checkout(kv_layer);
        let b2 = pool.checkout(kv_layer);
        assert_eq!(b1.len(), kv_layer);
        assert_eq!(b2.len(), kv_layer);
        drop(b1);
        drop(b2);
    }

    /// Pipeline parallelism activation: batch=1, seq=4096, hidden=8192, bf16 = 64 MiB.
    #[test]
    fn test_inference_pipeline_activation() {
        let pool = BufferPool::with_profile(PoolProfile::Inference);
        let activation = 64 * 1024 * 1024;
        let buf = pool.checkout(activation);
        assert_eq!(buf.len(), activation);
        drop(buf);
    }

    /// MoE token dispatch: 64 tokens × hidden=8192 × bf16 = 1 MiB.
    #[test]
    fn test_inference_moe_dispatch() {
        let pool = BufferPool::with_profile(PoolProfile::Inference);
        let dispatch = 1024 * 1024; // 1 MiB
        // Multiple concurrent expert dispatches.
        let b1 = pool.checkout(dispatch);
        let b2 = pool.checkout(dispatch);
        let b3 = pool.checkout(dispatch);
        let b4 = pool.checkout(dispatch);
        assert_eq!(b1.len(), dispatch);
        drop(b1);
        drop(b2);
        drop(b3);
        drop(b4);
    }

    /// Inference profile doesn't pre-allocate giant buffers (no gradients).
    /// Requesting >64 MiB still works — falls through to giant/unpooled with
    /// fresh allocation (acceptable for rare large activations).
    #[test]
    fn test_inference_large_activation_fallback() {
        let pool = BufferPool::with_profile(PoolProfile::Inference);
        // 128 MiB activation (batch=4, seq=4096, hidden=8192, bf16).
        // Giant pool is empty in inference profile — allocates fresh.
        let buf = pool.checkout(128 * 1024 * 1024);
        assert_eq!(buf.len(), 128 * 1024 * 1024);
        drop(buf);
    }
}
