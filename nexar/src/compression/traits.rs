//! Compression trait and wire format for gradient compression.

use crate::types::DataType;

/// Compressed representation of a tensor.
///
/// Wire format: `[k:u32][indices:k*u32][values:k*elem_size]`
/// where `k` is the number of non-zero elements after compression.
pub struct CompressedTensor {
    /// Serialized compressed data.
    pub data: Vec<u8>,
    /// Number of elements in the original uncompressed tensor.
    pub original_count: usize,
    /// Element data type.
    pub dtype: DataType,
}

impl CompressedTensor {
    /// Number of selected elements (k) from the wire format header.
    pub fn k(&self) -> usize {
        if self.data.len() < 4 {
            return 0;
        }
        u32::from_le_bytes([self.data[0], self.data[1], self.data[2], self.data[3]]) as usize
    }

    /// Encode a compressed tensor from indices and values.
    pub fn encode(indices: &[u32], values: &[u8], original_count: usize, dtype: DataType) -> Self {
        let k = indices.len();
        let elem_size = dtype.size_in_bytes();
        let mut data = Vec::with_capacity(4 + k * 4 + k * elem_size);
        data.extend_from_slice(&(k as u32).to_le_bytes());
        for &idx in indices {
            data.extend_from_slice(&idx.to_le_bytes());
        }
        data.extend_from_slice(&values[..k * elem_size]);
        Self {
            data,
            original_count,
            dtype,
        }
    }

    /// Decode indices and a slice reference to values from the wire format.
    pub fn decode_indices(&self) -> Vec<u32> {
        let k = self.k();
        let mut indices = Vec::with_capacity(k);
        for i in 0..k {
            let off = 4 + i * 4;
            indices.push(u32::from_le_bytes([
                self.data[off],
                self.data[off + 1],
                self.data[off + 2],
                self.data[off + 3],
            ]));
        }
        indices
    }

    /// Byte slice of compressed values (after the index array).
    pub fn values_bytes(&self) -> &[u8] {
        let k = self.k();
        let values_start = 4 + k * 4;
        &self.data[values_start..]
    }
}

/// Trait for gradient compressors.
///
/// Implementations compress a gradient tensor (byte slice) into a sparse
/// representation for bandwidth-efficient allreduce. Error feedback is
/// supported via a residual buffer that accumulates compression error.
pub trait Compressor: Send + Sync {
    /// Compress `input` (raw bytes for `count` elements of `dtype`).
    ///
    /// The residual buffer accumulates error feedback: before compression,
    /// `residual += input`; after selecting top-K positions, those positions
    /// are zeroed in the residual. On the first call, pass a zero-filled
    /// residual of the same size as input.
    fn compress(
        &self,
        input: &[u8],
        count: usize,
        dtype: DataType,
        residual: &mut [u8],
    ) -> CompressedTensor;

    /// Decompress a `CompressedTensor` into a dense output buffer.
    ///
    /// `output` must be pre-zeroed and have size `original_count * dtype.size_in_bytes()`.
    fn decompress(&self, compressed: &CompressedTensor, output: &mut [u8]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_tensor_encode_decode() {
        let indices = vec![1u32, 5, 10];
        let values: Vec<u8> = vec![
            // 3 f32 values
            0, 0, 128, 63, // 1.0
            0, 0, 0, 64, // 2.0
            0, 0, 64, 64, // 3.0
        ];
        let ct = CompressedTensor::encode(&indices, &values, 100, DataType::F32);
        assert_eq!(ct.k(), 3);
        assert_eq!(ct.original_count, 100);
        assert_eq!(ct.decode_indices(), vec![1, 5, 10]);
        assert_eq!(ct.values_bytes(), &values[..]);
    }
}
