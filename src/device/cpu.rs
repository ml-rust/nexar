use crate::device::adapter::DeviceAdapter;
use crate::error::{NexarError, Result};
use crate::types::{DataType, ReduceOp};

/// DeviceAdapter for host (CPU) memory. Direct pointer access, no copies needed.
#[derive(Debug, Clone, Default)]
pub struct CpuAdapter;

impl CpuAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl DeviceAdapter for CpuAdapter {
    unsafe fn stage_for_send(&self, ptr: u64, size_bytes: usize) -> Result<Vec<u8>> {
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size_bytes) };
        Ok(slice.to_vec())
    }

    unsafe fn receive_to_device(&self, data: &[u8], dst_ptr: u64) -> Result<()> {
        let dst = dst_ptr as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }

    unsafe fn reduce_inplace(
        &self,
        dst_ptr: u64,
        src: &[u8],
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        let expected = count * dtype.size_in_bytes();
        if src.len() != expected {
            return Err(NexarError::BufferSizeMismatch {
                expected,
                actual: src.len(),
            });
        }

        unsafe {
            match dtype {
                DataType::F32 => reduce_typed::<f32>(dst_ptr, src, count, op),
                DataType::F64 => reduce_typed::<f64>(dst_ptr, src, count, op),
                DataType::I32 => reduce_typed::<i32>(dst_ptr, src, count, op),
                DataType::I64 => reduce_typed::<i64>(dst_ptr, src, count, op),
                DataType::U32 => reduce_typed::<u32>(dst_ptr, src, count, op),
                DataType::U64 => reduce_typed::<u64>(dst_ptr, src, count, op),
                DataType::I8 => reduce_typed::<i8>(dst_ptr, src, count, op),
                DataType::U8 => reduce_typed::<u8>(dst_ptr, src, count, op),
                _ => {
                    return Err(NexarError::UnsupportedDType {
                        dtype,
                        op: "reduce_inplace",
                    });
                }
            }
        }
        Ok(())
    }
}

/// Trait for types that support the four reduction operations.
trait Reducible: Copy {
    fn reduce(self, other: Self, op: ReduceOp) -> Self;
}

macro_rules! impl_reducible_int {
    ($($ty:ty),*) => {
        $(
            impl Reducible for $ty {
                #[inline]
                fn reduce(self, other: Self, op: ReduceOp) -> Self {
                    match op {
                        ReduceOp::Sum => self.wrapping_add(other),
                        ReduceOp::Prod => self.wrapping_mul(other),
                        ReduceOp::Min => self.min(other),
                        ReduceOp::Max => self.max(other),
                    }
                }
            }
        )*
    };
}

impl_reducible_int!(i8, i32, i64, u8, u32, u64);

macro_rules! impl_reducible_float {
    ($($ty:ty),*) => {
        $(
            impl Reducible for $ty {
                #[inline]
                fn reduce(self, other: Self, op: ReduceOp) -> Self {
                    match op {
                        ReduceOp::Sum => self + other,
                        ReduceOp::Prod => self * other,
                        ReduceOp::Min => self.min(other),
                        ReduceOp::Max => self.max(other),
                    }
                }
            }
        )*
    };
}

impl_reducible_float!(f32, f64);

/// Apply reduction element-wise: `dst[i] = op(dst[i], src[i])`.
///
/// # Safety
/// `dst_ptr` must point to at least `count` elements of type `T`.
/// `src` must contain exactly `count * size_of::<T>()` bytes.
unsafe fn reduce_typed<T: Reducible>(dst_ptr: u64, src: &[u8], count: usize, op: ReduceOp) {
    let dst = dst_ptr as *mut T;
    let src_typed = src.as_ptr() as *const T;
    for i in 0..count {
        let d = unsafe { *dst.add(i) };
        let s = unsafe { *src_typed.add(i) };
        unsafe { *dst.add(i) = d.reduce(s, op) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_for_send_roundtrip() {
        let adapter = CpuAdapter::new();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let ptr = data.as_ptr() as u64;
        let size = data.len() * std::mem::size_of::<f32>();

        let staged = unsafe { adapter.stage_for_send(ptr, size).unwrap() };
        assert_eq!(staged.len(), size);

        // Verify content matches.
        let recovered: &[f32] =
            unsafe { std::slice::from_raw_parts(staged.as_ptr() as *const f32, 4) };
        assert_eq!(recovered, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_receive_to_device() {
        let adapter = CpuAdapter::new();
        let src = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut dst = [0u8; 4];

        unsafe {
            adapter
                .receive_to_device(&src, dst.as_mut_ptr() as u64)
                .unwrap();
        }
        assert_eq!(dst, src);
    }

    #[test]
    fn test_reduce_sum_f32() {
        let adapter = CpuAdapter::new();
        let mut dst: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let src: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };

        unsafe {
            adapter
                .reduce_inplace(
                    dst.as_mut_ptr() as u64,
                    src_bytes,
                    4,
                    DataType::F32,
                    ReduceOp::Sum,
                )
                .unwrap();
        }
        assert_eq!(dst, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_reduce_prod_f32() {
        let adapter = CpuAdapter::new();
        let mut dst: Vec<f32> = vec![2.0, 3.0, 4.0];
        let src: Vec<f32> = vec![5.0, 6.0, 7.0];
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };

        unsafe {
            adapter
                .reduce_inplace(
                    dst.as_mut_ptr() as u64,
                    src_bytes,
                    3,
                    DataType::F32,
                    ReduceOp::Prod,
                )
                .unwrap();
        }
        assert_eq!(dst, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_reduce_min_max_i32() {
        let adapter = CpuAdapter::new();
        let mut dst: Vec<i32> = vec![5, 1, 8, 3];
        let src: Vec<i32> = vec![2, 7, 4, 9];
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };

        unsafe {
            adapter
                .reduce_inplace(
                    dst.as_mut_ptr() as u64,
                    src_bytes,
                    4,
                    DataType::I32,
                    ReduceOp::Min,
                )
                .unwrap();
        }
        assert_eq!(dst, vec![2, 1, 4, 3]);

        // Reset and test Max.
        dst = vec![5, 1, 8, 3];
        unsafe {
            adapter
                .reduce_inplace(
                    dst.as_mut_ptr() as u64,
                    src_bytes,
                    4,
                    DataType::I32,
                    ReduceOp::Max,
                )
                .unwrap();
        }
        assert_eq!(dst, vec![5, 7, 8, 9]);
    }

    #[test]
    fn test_reduce_f64() {
        let adapter = CpuAdapter::new();
        let mut dst: Vec<f64> = vec![1.5, 2.5];
        let src: Vec<f64> = vec![3.5, 4.5];
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 8) };

        unsafe {
            adapter
                .reduce_inplace(
                    dst.as_mut_ptr() as u64,
                    src_bytes,
                    2,
                    DataType::F64,
                    ReduceOp::Sum,
                )
                .unwrap();
        }
        assert_eq!(dst, vec![5.0, 7.0]);
    }

    #[test]
    fn test_reduce_buffer_mismatch() {
        let adapter = CpuAdapter::new();
        let mut dst: Vec<f32> = vec![1.0, 2.0];
        let short_src = [0u8; 4]; // Only 4 bytes, but count=2 expects 8

        let result = unsafe {
            adapter.reduce_inplace(
                dst.as_mut_ptr() as u64,
                &short_src,
                2,
                DataType::F32,
                ReduceOp::Sum,
            )
        };
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_unsupported_dtype() {
        let adapter = CpuAdapter::new();
        let mut dst = [0u8; 4];
        let src = [0u8; 4];

        let result = unsafe {
            adapter.reduce_inplace(
                dst.as_mut_ptr() as u64,
                &src,
                2,
                DataType::F16, // F16 not supported for reduce
                ReduceOp::Sum,
            )
        };
        assert!(result.is_err());
    }
}
