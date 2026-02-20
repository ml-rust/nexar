//! Half-precision float newtypes and reduction trait machinery.
//!
//! Shared by `reduce.rs` (byte-slice dispatch) and `reduce_simd.rs` (SIMD paths).

use crate::types::ReduceOp;

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

// ── LeBytes trait ──────────────────────────────────────────────────────

/// Read/write a value from/to a little-endian byte slice (alignment-safe).
pub(crate) trait LeBytes: Sized {
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

// ── Identity trait ─────────────────────────────────────────────────────

/// Identity element for a reduction operation.
pub(crate) trait Identity: LeBytes + Copy {
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
