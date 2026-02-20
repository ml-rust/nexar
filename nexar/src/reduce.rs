//! Shared element-wise reduction primitives used by collective algorithms
//! and the CPU device adapter.

use crate::error::Result;
use crate::types::{DataType, ReduceOp};

// ── BF16 / F16 newtypes ────────────────────────────────────────────────

/// Brain floating-point 16: sign(1) + exponent(8) + mantissa(7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bf16(pub u16);

impl Bf16 {
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        // Round-to-nearest-even: add rounding bias then truncate.
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        Bf16((rounded >> 16) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

/// IEEE 754 half-precision float: sign(1) + exponent(5) + mantissa(10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F16(pub u16);

impl F16 {
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        if exponent == 0xFF {
            // Inf / NaN
            let h_mantissa = if mantissa != 0 { 0x0200 } else { 0 };
            return F16((sign | 0x7C00 | h_mantissa) as u16);
        }

        let unbiased = exponent - 127;

        if unbiased > 15 {
            // Overflow → Inf
            return F16((sign | 0x7C00) as u16);
        }

        if unbiased < -24 {
            // Too small → zero
            return F16(sign as u16);
        }

        if unbiased < -14 {
            // Subnormal in f16
            let shift = -1 - unbiased;
            let m = (mantissa | 0x0080_0000) >> (shift + 13);
            return F16((sign | m) as u16);
        }

        // Normal
        let h_exp = ((unbiased + 15) as u32) << 10;
        let h_man = mantissa >> 13;
        // Rounding
        let round_bit = (mantissa >> 12) & 1;
        F16((sign | h_exp | (h_man + round_bit)) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exponent = (h >> 10) & 0x1F;
        let mantissa = h & 0x03FF;

        if exponent == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign); // ±0
            }
            // Subnormal: normalize
            let mut m = mantissa;
            let mut e: i32 = -14 + 127;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            return f32::from_bits(sign | ((e as u32) << 23) | (m << 13));
        }

        if exponent == 31 {
            // Inf / NaN
            let f_man = if mantissa != 0 { 0x0040_0000 } else { 0 };
            return f32::from_bits(sign | 0x7F80_0000 | f_man);
        }

        // Normal: rebias exponent 15→127
        let f_exp = (exponent + 127 - 15) << 23;
        f32::from_bits(sign | f_exp | (mantissa << 13))
    }
}

// ── Reducible trait ────────────────────────────────────────────────────

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

impl Reducible for Bf16 {
    #[inline]
    fn reduce(a: Self, b: Self, op: ReduceOp) -> Self {
        Bf16::from_f32(f32::reduce(a.to_f32(), b.to_f32(), op))
    }
}

impl Reducible for F16 {
    #[inline]
    fn reduce(a: Self, b: Self, op: ReduceOp) -> Self {
        F16::from_f32(f32::reduce(a.to_f32(), b.to_f32(), op))
    }
}

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

impl LeBytes for Bf16 {
    #[inline]
    fn read_le(bytes: &[u8]) -> Self {
        Bf16(u16::from_le_bytes(
            bytes.try_into().expect("slice length matches type size"),
        ))
    }
    #[inline]
    fn write_le(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.0.to_le_bytes());
    }
}

impl LeBytes for F16 {
    #[inline]
    fn read_le(bytes: &[u8]) -> Self {
        F16(u16::from_le_bytes(
            bytes.try_into().expect("slice length matches type size"),
        ))
    }
    #[inline]
    fn write_le(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.0.to_le_bytes());
    }
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

impl Identity for Bf16 {
    #[inline]
    fn identity(op: ReduceOp) -> Self {
        Bf16::from_f32(f32::identity(op))
    }
}

impl Identity for F16 {
    #[inline]
    fn identity(op: ReduceOp) -> Self {
        F16::from_f32(f32::identity(op))
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
