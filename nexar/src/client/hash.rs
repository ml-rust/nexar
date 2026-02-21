/// Compute a non-zero FNV-1a hash over an iterator of byte slices.
///
/// Used to derive deterministic `comm_id` values that all ranks agree on,
/// e.g. after split, elastic grow, or rebuild operations. Returns a non-zero
/// `u64` (zero is reserved for the root communicator).
pub(super) fn fnv1a_comm_id<I, S>(parts: I) -> u64
where
    I: IntoIterator<Item = S>,
    S: AsRef<[u8]>,
{
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for part in parts {
        for &b in part.as_ref() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    // Ensure non-zero (0 is reserved for root comm).
    if h == 0 { 1 } else { h }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_deterministic() {
        let a = fnv1a_comm_id([&1u64.to_le_bytes()[..], &2u32.to_le_bytes()]);
        let b = fnv1a_comm_id([&1u64.to_le_bytes()[..], &2u32.to_le_bytes()]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        let a = fnv1a_comm_id([&1u64.to_le_bytes()[..]]);
        let b = fnv1a_comm_id([&2u64.to_le_bytes()[..]]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_fnv1a_non_zero() {
        // Exhaustive check isn't feasible, but verify a few cases don't return 0.
        for i in 0..1000u64 {
            assert_ne!(fnv1a_comm_id([&i.to_le_bytes()[..]]), 0);
        }
    }
}
