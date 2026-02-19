//! Shared element-wise reduction primitives used by collective algorithms
//! and the CPU device adapter.

use crate::error::{NexarError, Result};
use crate::types::{DataType, ReduceOp};

/// Trait for types that support the four reduction operations.
pub(crate) trait Reducible: Copy + 'static {
    fn reduce(a: Self, b: Self, op: ReduceOp) -> Self;
}

macro_rules! impl_reducible {
    (int: $($ty:ty),*) => {
        $(
            impl Reducible for $ty {
                #[inline]
                fn reduce(a: Self, b: Self, op: ReduceOp) -> Self {
                    match op {
                        ReduceOp::Sum => a.wrapping_add(b),
                        ReduceOp::Prod => a.wrapping_mul(b),
                        ReduceOp::Min => a.min(b),
                        ReduceOp::Max => a.max(b),
                    }
                }
            }
        )*
    };
    (float: $($ty:ty),*) => {
        $(
            impl Reducible for $ty {
                #[inline]
                fn reduce(a: Self, b: Self, op: ReduceOp) -> Self {
                    match op {
                        ReduceOp::Sum => a + b,
                        ReduceOp::Prod => a * b,
                        ReduceOp::Min => a.min(b),
                        ReduceOp::Max => a.max(b),
                    }
                }
            }
        )*
    };
}

impl_reducible!(int: i8, i32, i64, u8, u32, u64);
impl_reducible!(float: f32, f64);

/// Element-wise reduce on byte slices interpreted as `dtype` elements.
///
/// `dst` and `src` must both contain exactly `count * dtype.size_in_bytes()` bytes.
pub(crate) fn reduce_slice(
    dst: &mut [u8],
    src: &[u8],
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    match dtype {
        DataType::F32 => reduce_slice_typed::<f32>(dst, src, count, op),
        DataType::F64 => reduce_slice_typed::<f64>(dst, src, count, op),
        DataType::I32 => reduce_slice_typed::<i32>(dst, src, count, op),
        DataType::I64 => reduce_slice_typed::<i64>(dst, src, count, op),
        DataType::U32 => reduce_slice_typed::<u32>(dst, src, count, op),
        DataType::U64 => reduce_slice_typed::<u64>(dst, src, count, op),
        DataType::I8 => reduce_slice_typed::<i8>(dst, src, count, op),
        DataType::U8 => reduce_slice_typed::<u8>(dst, src, count, op),
        _ => {
            return Err(NexarError::UnsupportedDType {
                dtype,
                op: "reduce",
            });
        }
    }
    Ok(())
}

/// Element-wise reduce via raw pointer (for device adapter).
///
/// # Safety
/// `dst_ptr` must point to at least `count` elements of type matching `dtype`.
/// `src` must contain exactly `count * dtype.size_in_bytes()` bytes.
pub unsafe fn reduce_ptr(
    dst_ptr: u64,
    src: &[u8],
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    unsafe {
        match dtype {
            DataType::F32 => reduce_ptr_typed::<f32>(dst_ptr, src, count, op),
            DataType::F64 => reduce_ptr_typed::<f64>(dst_ptr, src, count, op),
            DataType::I32 => reduce_ptr_typed::<i32>(dst_ptr, src, count, op),
            DataType::I64 => reduce_ptr_typed::<i64>(dst_ptr, src, count, op),
            DataType::U32 => reduce_ptr_typed::<u32>(dst_ptr, src, count, op),
            DataType::U64 => reduce_ptr_typed::<u64>(dst_ptr, src, count, op),
            DataType::I8 => reduce_ptr_typed::<i8>(dst_ptr, src, count, op),
            DataType::U8 => reduce_ptr_typed::<u8>(dst_ptr, src, count, op),
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

/// Read a value from a little-endian byte slice (alignment-safe).
trait LeBytes: Sized {
    fn read_le(bytes: &[u8]) -> Self;
    fn write_le(self, bytes: &mut [u8]);
}

macro_rules! impl_le_bytes {
    ($($ty:ty),*) => {
        $(
            impl LeBytes for $ty {
                #[inline]
                fn read_le(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(
                        bytes.try_into().expect("slice length matches type size"),
                    )
                }
                #[inline]
                fn write_le(self, bytes: &mut [u8]) {
                    bytes.copy_from_slice(&self.to_le_bytes());
                }
            }
        )*
    };
}

impl_le_bytes!(i8, i32, i64, u8, u32, u64, f32, f64);

fn reduce_slice_typed<T: Reducible + LeBytes>(
    dst: &mut [u8],
    src: &[u8],
    count: usize,
    op: ReduceOp,
) {
    let t_size = std::mem::size_of::<T>();
    for i in 0..count {
        let off = i * t_size;
        let a = T::read_le(&dst[off..off + t_size]);
        let b = T::read_le(&src[off..off + t_size]);
        let r = T::reduce(a, b, op);
        r.write_le(&mut dst[off..off + t_size]);
    }
}

/// Returns a byte buffer of `count * dtype.size_in_bytes()` filled with the
/// identity element for the given op+dtype combination.
///
/// Identity values: Sum→0, Prod→1, Min→type::MAX, Max→type::MIN.
pub(crate) fn identity_slice(count: usize, dtype: DataType, op: ReduceOp) -> Result<Vec<u8>> {
    match dtype {
        DataType::F32 => Ok(identity_slice_typed::<f32>(count, op)),
        DataType::F64 => Ok(identity_slice_typed::<f64>(count, op)),
        DataType::I32 => Ok(identity_slice_typed::<i32>(count, op)),
        DataType::I64 => Ok(identity_slice_typed::<i64>(count, op)),
        DataType::U32 => Ok(identity_slice_typed::<u32>(count, op)),
        DataType::U64 => Ok(identity_slice_typed::<u64>(count, op)),
        DataType::I8 => Ok(identity_slice_typed::<i8>(count, op)),
        DataType::U8 => Ok(identity_slice_typed::<u8>(count, op)),
        _ => Err(NexarError::UnsupportedDType {
            dtype,
            op: "identity",
        }),
    }
}

/// Identity element for a reduction operation.
trait Identity: LeBytes + Copy {
    fn identity(op: ReduceOp) -> Self;
}

macro_rules! impl_identity {
    (int: $($ty:ty),*) => {
        $(
            impl Identity for $ty {
                #[inline]
                fn identity(op: ReduceOp) -> Self {
                    match op {
                        ReduceOp::Sum => 0,
                        ReduceOp::Prod => 1,
                        ReduceOp::Min => <$ty>::MAX,
                        ReduceOp::Max => <$ty>::MIN,
                    }
                }
            }
        )*
    };
    (float: $($ty:ty),*) => {
        $(
            impl Identity for $ty {
                #[inline]
                fn identity(op: ReduceOp) -> Self {
                    match op {
                        ReduceOp::Sum => 0.0,
                        ReduceOp::Prod => 1.0,
                        ReduceOp::Min => <$ty>::MAX,
                        ReduceOp::Max => <$ty>::MIN,
                    }
                }
            }
        )*
    };
}

impl_identity!(int: i8, i32, i64, u8, u32, u64);
impl_identity!(float: f32, f64);

fn identity_slice_typed<T: Identity>(count: usize, op: ReduceOp) -> Vec<u8> {
    let val = T::identity(op);
    let t_size = std::mem::size_of::<T>();
    let mut buf = vec![0u8; count * t_size];
    for i in 0..count {
        val.write_le(&mut buf[i * t_size..(i + 1) * t_size]);
    }
    buf
}

/// # Safety
/// `dst_ptr` must point to at least `count` elements of type `T`.
unsafe fn reduce_ptr_typed<T: Reducible>(dst_ptr: u64, src: &[u8], count: usize, op: ReduceOp) {
    let dst = dst_ptr as *mut T;
    let src_typed = src.as_ptr() as *const T;
    for i in 0..count {
        let d = unsafe { *dst.add(i) };
        let s = unsafe { *src_typed.add(i) };
        unsafe { *dst.add(i) = T::reduce(d, s, op) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_slice_sum_f32() {
        let mut dst = [1.0f32, 2.0, 3.0, 4.0];
        let src = [10.0f32, 20.0, 30.0, 40.0];
        let dst_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len() * 4) };
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        reduce_slice(dst_bytes, src_bytes, 4, DataType::F32, ReduceOp::Sum).unwrap();
        assert_eq!(dst, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_reduce_slice_unsupported() {
        let mut dst = [0u8; 4];
        let src = [0u8; 4];
        let result = reduce_slice(&mut dst, &src, 2, DataType::F16, ReduceOp::Sum);
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_ptr_sum_i32() {
        let mut dst: Vec<i32> = vec![1, 2, 3];
        let src: Vec<i32> = vec![10, 20, 30];
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        unsafe {
            reduce_ptr(
                dst.as_mut_ptr() as u64,
                src_bytes,
                3,
                DataType::I32,
                ReduceOp::Sum,
            )
            .unwrap();
        }
        assert_eq!(dst, vec![11, 22, 33]);
    }
}
