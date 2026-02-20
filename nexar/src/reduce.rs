//! Shared element-wise reduction primitives used by collective algorithms
//! and the CPU device adapter.

use crate::error::Result;
use crate::reduce_types::{Bf16, F16, Identity, LeBytes, Reducible};
use crate::types::{DataType, ReduceOp};

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
        DataType::F32 => {
            if unsafe { crate::reduce_simd::reduce_f32_simd(dst, src, count, op) } {
                return Ok(());
            }
            reduce_slice_typed::<f32>(dst, src, count, op);
        }
        DataType::F64 => {
            if unsafe { crate::reduce_simd::reduce_f64_simd(dst, src, count, op) } {
                return Ok(());
            }
            reduce_slice_typed::<f64>(dst, src, count, op);
        }
        DataType::BF16 => {
            if unsafe { crate::reduce_simd::reduce_bf16_simd(dst, src, count, op) } {
                return Ok(());
            }
            reduce_slice_typed::<Bf16>(dst, src, count, op);
        }
        DataType::F16 => {
            if unsafe { crate::reduce_simd::reduce_f16_simd(dst, src, count, op) } {
                return Ok(());
            }
            reduce_slice_typed::<F16>(dst, src, count, op);
        }
        DataType::I32 => reduce_slice_typed::<i32>(dst, src, count, op),
        DataType::I64 => reduce_slice_typed::<i64>(dst, src, count, op),
        DataType::U32 => reduce_slice_typed::<u32>(dst, src, count, op),
        DataType::U64 => reduce_slice_typed::<u64>(dst, src, count, op),
        DataType::I8 => reduce_slice_typed::<i8>(dst, src, count, op),
        DataType::U8 => reduce_slice_typed::<u8>(dst, src, count, op),
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
            DataType::BF16 => reduce_ptr_typed::<Bf16>(dst_ptr, src, count, op),
            DataType::F16 => reduce_ptr_typed::<F16>(dst_ptr, src, count, op),
            DataType::I32 => reduce_ptr_typed::<i32>(dst_ptr, src, count, op),
            DataType::I64 => reduce_ptr_typed::<i64>(dst_ptr, src, count, op),
            DataType::U32 => reduce_ptr_typed::<u32>(dst_ptr, src, count, op),
            DataType::U64 => reduce_ptr_typed::<u64>(dst_ptr, src, count, op),
            DataType::I8 => reduce_ptr_typed::<i8>(dst_ptr, src, count, op),
            DataType::U8 => reduce_ptr_typed::<u8>(dst_ptr, src, count, op),
        }
    }
    Ok(())
}

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
        DataType::BF16 => Ok(identity_slice_typed::<Bf16>(count, op)),
        DataType::F16 => Ok(identity_slice_typed::<F16>(count, op)),
        DataType::I32 => Ok(identity_slice_typed::<i32>(count, op)),
        DataType::I64 => Ok(identity_slice_typed::<i64>(count, op)),
        DataType::U32 => Ok(identity_slice_typed::<u32>(count, op)),
        DataType::U64 => Ok(identity_slice_typed::<u64>(count, op)),
        DataType::I8 => Ok(identity_slice_typed::<i8>(count, op)),
        DataType::U8 => Ok(identity_slice_typed::<u8>(count, op)),
    }
}

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
    fn test_bf16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 3.125, 65504.0, -0.0];
        for &v in &values {
            let bf = Bf16::from_f32(v);
            let back = bf.to_f32();
            // BF16 has ~7-bit mantissa, so ~1% relative error for non-zero.
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                assert!(
                    (back - v).abs() / v.abs() < 0.01,
                    "bf16 roundtrip failed for {v}"
                );
            }
        }
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 3.125, 65504.0, -0.0];
        for &v in &values {
            let h = F16::from_f32(v);
            let back = h.to_f32();
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                assert!(
                    (back - v).abs() / v.abs() < 0.002,
                    "f16 roundtrip failed for {v}: got {back}"
                );
            }
        }
    }

    #[test]
    fn test_reduce_slice_bf16_sum() {
        let a = [Bf16::from_f32(1.0), Bf16::from_f32(2.0)];
        let b = [Bf16::from_f32(10.0), Bf16::from_f32(20.0)];
        let mut dst_buf = vec![0u8; 4];
        let mut src_buf = vec![0u8; 4];
        a[0].write_le(&mut dst_buf[0..2]);
        a[1].write_le(&mut dst_buf[2..4]);
        b[0].write_le(&mut src_buf[0..2]);
        b[1].write_le(&mut src_buf[2..4]);

        reduce_slice(&mut dst_buf, &src_buf, 2, DataType::BF16, ReduceOp::Sum).unwrap();

        let r0 = Bf16::read_le(&dst_buf[0..2]).to_f32();
        let r1 = Bf16::read_le(&dst_buf[2..4]).to_f32();
        assert!((r0 - 11.0).abs() < 0.1);
        assert!((r1 - 22.0).abs() < 0.1);
    }

    #[test]
    fn test_reduce_slice_f16_sum() {
        let a = [F16::from_f32(1.0), F16::from_f32(2.0)];
        let b = [F16::from_f32(10.0), F16::from_f32(20.0)];
        let mut dst_buf = vec![0u8; 4];
        let mut src_buf = vec![0u8; 4];
        a[0].write_le(&mut dst_buf[0..2]);
        a[1].write_le(&mut dst_buf[2..4]);
        b[0].write_le(&mut src_buf[0..2]);
        b[1].write_le(&mut src_buf[2..4]);

        reduce_slice(&mut dst_buf, &src_buf, 2, DataType::F16, ReduceOp::Sum).unwrap();

        let r0 = F16::read_le(&dst_buf[0..2]).to_f32();
        let r1 = F16::read_le(&dst_buf[2..4]).to_f32();
        assert!((r0 - 11.0).abs() < 0.01);
        assert!((r1 - 22.0).abs() < 0.01);
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
