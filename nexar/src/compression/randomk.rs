//! Random-K sampling: randomly select K% of elements.
//!
//! Simpler than TopK (no sort needed), and unbiased in expectation when
//! combined with error feedback. Good for very large tensors where TopK's
//! O(n log n) sort is too expensive.

use crate::types::DataType;

use super::traits::{CompressedTensor, Compressor};

/// Random-K gradient compressor.
///
/// Randomly samples `ratio` fraction of elements. Combined with error
/// feedback (residual accumulation), this is unbiased in expectation.
pub struct RandomKCompressor {
    /// Fraction of elements to keep (0.0, 1.0].
    ratio: f64,
}

impl RandomKCompressor {
    pub fn new(ratio: f64) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0, "ratio must be in (0.0, 1.0]");
        Self { ratio }
    }
}

impl Compressor for RandomKCompressor {
    fn compress(
        &self,
        input: &[u8],
        count: usize,
        dtype: DataType,
        residual: &mut [u8],
    ) -> CompressedTensor {
        let elem_size = dtype.size_in_bytes();
        let k = ((count as f64 * self.ratio).ceil() as usize)
            .max(1)
            .min(count);

        // Add input to residual (error feedback).
        add_bytes(residual, input, count, elem_size);

        // Deterministic pseudo-random selection using a simple LCG seeded from residual content.
        // This avoids pulling in a rand dependency. The seed changes each call because the
        // residual content changes.
        let seed = residual_hash(residual);
        let indices = sample_indices(count, k, seed);

        // Extract values from residual at selected indices.
        let mut values = vec![0u8; k * elem_size];
        for (i, &idx) in indices.iter().enumerate() {
            let src_off = idx as usize * elem_size;
            let dst_off = i * elem_size;
            values[dst_off..dst_off + elem_size]
                .copy_from_slice(&residual[src_off..src_off + elem_size]);
        }

        // Zero selected positions in residual.
        for &idx in &indices {
            let off = idx as usize * elem_size;
            for b in &mut residual[off..off + elem_size] {
                *b = 0;
            }
        }

        CompressedTensor::encode(&indices, &values, count, dtype)
    }

    fn decompress(&self, compressed: &CompressedTensor, output: &mut [u8]) {
        let k = compressed.k();
        let elem_size = compressed.dtype.size_in_bytes();
        let indices = compressed.decode_indices();
        let values = compressed.values_bytes();

        for (i, &idx) in indices.iter().enumerate().take(k) {
            let src_off = i * elem_size;
            let dst_off = idx as usize * elem_size;
            output[dst_off..dst_off + elem_size]
                .copy_from_slice(&values[src_off..src_off + elem_size]);
        }
    }
}

/// Element-wise add input bytes to residual, treating as native-endian values.
fn add_bytes(residual: &mut [u8], input: &[u8], count: usize, elem_size: usize) {
    // For simplicity, add as f32/f64 when possible, else byte-wise XOR (lossy but functional).
    match elem_size {
        4 => {
            for i in 0..count {
                let off = i * 4;
                let r = f32::from_le_bytes([
                    residual[off],
                    residual[off + 1],
                    residual[off + 2],
                    residual[off + 3],
                ]);
                let v = f32::from_le_bytes([
                    input[off],
                    input[off + 1],
                    input[off + 2],
                    input[off + 3],
                ]);
                residual[off..off + 4].copy_from_slice(&(r + v).to_le_bytes());
            }
        }
        8 => {
            for i in 0..count {
                let off = i * 8;
                let r = f64::from_le_bytes([
                    residual[off],
                    residual[off + 1],
                    residual[off + 2],
                    residual[off + 3],
                    residual[off + 4],
                    residual[off + 5],
                    residual[off + 6],
                    residual[off + 7],
                ]);
                let v = f64::from_le_bytes([
                    input[off],
                    input[off + 1],
                    input[off + 2],
                    input[off + 3],
                    input[off + 4],
                    input[off + 5],
                    input[off + 6],
                    input[off + 7],
                ]);
                residual[off..off + 8].copy_from_slice(&(r + v).to_le_bytes());
            }
        }
        _ => {
            // Fallback: treat as integer bytes and add with wrapping.
            for i in 0..residual.len().min(input.len()) {
                residual[i] = residual[i].wrapping_add(input[i]);
            }
        }
    }
}

/// Simple hash of residual content for seeding the sampler.
fn residual_hash(residual: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in residual.iter().step_by(64).take(256) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Fisher-Yates partial shuffle to select k unique indices from [0, n).
fn sample_indices(n: usize, k: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut pool: Vec<u32> = (0..n as u32).collect();
    for i in 0..k {
        // LCG step.
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = i + (state as usize % (n - i));
        pool.swap(i, j);
    }
    let mut selected = pool[..k].to_vec();
    selected.sort_unstable();
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randomk_compress_f32() {
        let compressor = RandomKCompressor::new(0.5);
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let mut residual = vec![0u8; 16];

        let ct = compressor.compress(input_bytes, 4, DataType::F32, &mut residual);
        assert_eq!(ct.k(), 2);
        assert_eq!(ct.original_count, 4);

        // Decompress.
        let mut output = vec![0u8; 16];
        compressor.decompress(&ct, &mut output);

        // Exactly 2 values should be non-zero.
        let out_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(output.as_ptr() as *const f32, 4) };
        let nonzero_count = out_f32.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero_count, 2);
    }

    #[test]
    fn test_sample_indices_unique() {
        let indices = sample_indices(100, 10, 42);
        assert_eq!(indices.len(), 10);
        // All unique.
        let mut sorted = indices.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
        // All in range.
        for &idx in &indices {
            assert!(idx < 100);
        }
    }
}
