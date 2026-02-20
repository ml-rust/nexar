//! TopK sparsification: keep the top K% elements by magnitude.

use crate::types::DataType;

use super::traits::{CompressedTensor, Compressor};

/// TopK gradient compressor.
///
/// Keeps the top `ratio` fraction of elements (by absolute magnitude).
/// Uses error feedback: residual accumulates compression error so that
/// small gradients are eventually communicated.
pub struct TopKCompressor {
    /// Fraction of elements to keep (0.0, 1.0]. E.g. 0.01 = top 1%.
    ratio: f64,
}

impl TopKCompressor {
    pub fn new(ratio: f64) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0, "ratio must be in (0.0, 1.0]");
        Self { ratio }
    }
}

impl Compressor for TopKCompressor {
    fn compress(
        &self,
        input: &[u8],
        count: usize,
        dtype: DataType,
        residual: &mut [u8],
    ) -> CompressedTensor {
        match dtype {
            DataType::F32 => compress_topk::<f32>(input, count, self.ratio, residual, dtype),
            DataType::F64 => compress_topk::<f64>(input, count, self.ratio, residual, dtype),
            _ => {
                // For unsupported types, pass through uncompressed.
                let indices: Vec<u32> = (0..count as u32).collect();
                CompressedTensor::encode(&indices, input, count, dtype)
            }
        }
    }

    fn decompress(&self, compressed: &CompressedTensor, output: &mut [u8]) {
        decompress_sparse(compressed, output);
    }
}

trait FloatAbs: Copy + PartialOrd {
    fn abs_val(self) -> Self;
    fn read_le(bytes: &[u8]) -> Self;
    fn write_le(self, bytes: &mut [u8]);
    fn add(self, other: Self) -> Self;
    fn zero() -> Self;
}

impl FloatAbs for f32 {
    #[inline]
    fn abs_val(self) -> Self {
        self.abs()
    }
    #[inline]
    fn read_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
    #[inline]
    fn write_le(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

impl FloatAbs for f64 {
    #[inline]
    fn abs_val(self) -> Self {
        self.abs()
    }
    #[inline]
    fn read_le(bytes: &[u8]) -> Self {
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
    #[inline]
    fn write_le(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

fn compress_topk<T: FloatAbs>(
    input: &[u8],
    count: usize,
    ratio: f64,
    residual: &mut [u8],
    dtype: DataType,
) -> CompressedTensor {
    let elem_size = std::mem::size_of::<T>();
    let k = ((count as f64 * ratio).ceil() as usize).max(1).min(count);

    // Add input to residual (error feedback).
    for i in 0..count {
        let off = i * elem_size;
        let r = T::read_le(&residual[off..off + elem_size]);
        let v = T::read_le(&input[off..off + elem_size]);
        T::add(r, v).write_le(&mut residual[off..off + elem_size]);
    }

    // Find top-K by magnitude using partial sort.
    let mut indices_by_mag: Vec<u32> = (0..count as u32).collect();
    indices_by_mag.sort_unstable_by(|&a, &b| {
        let off_a = a as usize * elem_size;
        let off_b = b as usize * elem_size;
        let va = T::read_le(&residual[off_a..off_a + elem_size]).abs_val();
        let vb = T::read_le(&residual[off_b..off_b + elem_size]).abs_val();
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });
    indices_by_mag.truncate(k);

    // Extract values from residual at selected indices.
    let mut values = vec![0u8; k * elem_size];
    for (i, &idx) in indices_by_mag.iter().enumerate() {
        let src_off = idx as usize * elem_size;
        let dst_off = i * elem_size;
        values[dst_off..dst_off + elem_size]
            .copy_from_slice(&residual[src_off..src_off + elem_size]);
    }

    // Zero selected positions in residual.
    for &idx in &indices_by_mag {
        let off = idx as usize * elem_size;
        T::zero().write_le(&mut residual[off..off + elem_size]);
    }

    // Sort indices for deterministic wire order.
    // We need to keep values aligned with indices, so sort pairs.
    let mut pairs: Vec<(u32, Vec<u8>)> = indices_by_mag
        .iter()
        .enumerate()
        .map(|(i, &idx)| {
            let off = i * elem_size;
            (idx, values[off..off + elem_size].to_vec())
        })
        .collect();
    pairs.sort_unstable_by_key(|&(idx, _)| idx);

    let sorted_indices: Vec<u32> = pairs.iter().map(|p| p.0).collect();
    let mut sorted_values = vec![0u8; k * elem_size];
    for (i, (_, v)) in pairs.iter().enumerate() {
        sorted_values[i * elem_size..(i + 1) * elem_size].copy_from_slice(v);
    }

    CompressedTensor::encode(&sorted_indices, &sorted_values, count, dtype)
}

/// Decompress sparse format into a dense zero-filled output.
fn decompress_sparse(compressed: &CompressedTensor, output: &mut [u8]) {
    let k = compressed.k();
    let elem_size = compressed.dtype.size_in_bytes();
    let indices = compressed.decode_indices();
    let values = compressed.values_bytes();

    for (i, &idx) in indices.iter().enumerate().take(k) {
        let src_off = i * elem_size;
        let dst_off = idx as usize * elem_size;
        output[dst_off..dst_off + elem_size].copy_from_slice(&values[src_off..src_off + elem_size]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_compress_f32() {
        let compressor = TopKCompressor::new(0.5); // keep 50%
        let input: Vec<f32> = vec![1.0, -5.0, 0.1, 3.0];
        let input_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let mut residual = vec![0u8; 16];

        let ct = compressor.compress(input_bytes, 4, DataType::F32, &mut residual);
        assert_eq!(ct.k(), 2); // top 50% of 4 = 2
        assert_eq!(ct.original_count, 4);

        // Decompress and verify.
        let mut output = vec![0u8; 16];
        compressor.decompress(&ct, &mut output);

        let out_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(output.as_ptr() as *const f32, 4) };
        // Should have the two largest by magnitude: -5.0 (idx 1) and 3.0 (idx 3).
        assert_eq!(out_f32[0], 0.0); // not selected
        assert_eq!(out_f32[1], -5.0);
        assert_eq!(out_f32[2], 0.0); // not selected
        assert_eq!(out_f32[3], 3.0);
    }

    #[test]
    fn test_topk_error_feedback() {
        let compressor = TopKCompressor::new(0.25); // keep 1 of 4
        let input: Vec<f32> = vec![0.1, 0.2, 0.3, 10.0];
        let input_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let mut residual = vec![0u8; 16];

        // First round: only index 3 (10.0) selected.
        let ct1 = compressor.compress(input_bytes, 4, DataType::F32, &mut residual);
        assert_eq!(ct1.k(), 1);

        // Residual should have accumulated error for indices 0,1,2.
        let res_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(residual.as_ptr() as *const f32, 4) };
        assert!((res_f32[0] - 0.1).abs() < 1e-6);
        assert!((res_f32[1] - 0.2).abs() < 1e-6);
        assert!((res_f32[2] - 0.3).abs() < 1e-6);
        assert_eq!(res_f32[3], 0.0); // was zeroed
    }
}
