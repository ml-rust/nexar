//! Identity (no-op) compressor. Passes data through unmodified.

use crate::types::DataType;

use super::traits::{CompressedTensor, Compressor};

/// No-op compressor that passes all elements through.
pub struct NoCompression;

impl Compressor for NoCompression {
    fn compress(
        &self,
        input: &[u8],
        count: usize,
        dtype: DataType,
        _residual: &mut [u8],
    ) -> CompressedTensor {
        let indices: Vec<u32> = (0..count as u32).collect();
        CompressedTensor::encode(&indices, input, count, dtype)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compression_roundtrip() {
        let compressor = NoCompression;
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let mut residual = vec![0u8; 16];

        let ct = compressor.compress(input_bytes, 4, DataType::F32, &mut residual);
        assert_eq!(ct.k(), 4);

        let mut output = vec![0u8; 16];
        compressor.decompress(&ct, &mut output);

        let out_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(output.as_ptr() as *const f32, 4) };
        assert_eq!(out_f32, &[1.0, 2.0, 3.0, 4.0]);
    }
}
