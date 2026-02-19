use crate::types::Rank;

/// A globally-addressable pointer referencing memory on a specific rank.
///
/// Pairs a rank with a local device pointer (`u64`), allowing remote
/// reads/writes in RDMA-style one-sided communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalPtr {
    /// Rank that owns this memory.
    pub rank: Rank,
    /// Local device pointer on that rank.
    pub ptr: u64,
    /// Size of the allocation in bytes.
    pub size_bytes: usize,
}

impl GlobalPtr {
    /// Create a new global pointer.
    pub const fn new(rank: Rank, ptr: u64, size_bytes: usize) -> Self {
        Self {
            rank,
            ptr,
            size_bytes,
        }
    }

    /// Returns true if this pointer refers to memory on the given rank.
    pub const fn is_local(&self, my_rank: Rank) -> bool {
        self.rank == my_rank
    }
}

impl std::fmt::Display for GlobalPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GlobalPtr(rank={}, ptr=0x{:x}, {}B)",
            self.rank, self.ptr, self.size_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_ptr_construction() {
        let gp = GlobalPtr::new(2, 0xDEAD_BEEF, 1024);
        assert_eq!(gp.rank, 2);
        assert_eq!(gp.ptr, 0xDEAD_BEEF);
        assert_eq!(gp.size_bytes, 1024);
    }

    #[test]
    fn test_is_local() {
        let gp = GlobalPtr::new(3, 0x1000, 64);
        assert!(gp.is_local(3));
        assert!(!gp.is_local(0));
        assert!(!gp.is_local(4));
    }

    #[test]
    fn test_display() {
        let gp = GlobalPtr::new(0, 0xFF, 256);
        let s = gp.to_string();
        assert!(s.contains("rank=0"));
        assert!(s.contains("0xff"));
        assert!(s.contains("256B"));
    }

    #[test]
    fn test_eq_hash() {
        use std::collections::HashSet;
        let a = GlobalPtr::new(1, 0x100, 32);
        let b = GlobalPtr::new(1, 0x100, 32);
        let c = GlobalPtr::new(2, 0x100, 32);
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }
}
