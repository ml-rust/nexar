//! SIMD-accelerated reduction dispatch for multiple architectures and data types.

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use crate::types::ReduceOp;

/// Try SIMD-accelerated f32 reduction. Returns `true` if handled.
///
/// # Safety
/// `dst` and `src` must both have at least `count * 4` bytes.
pub(crate) unsafe fn reduce_f32_simd(
    dst: &mut [u8],
    src: &[u8],
    count: usize,
    op: ReduceOp,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { x86_64::reduce_f32_op_avx512(dst, src, count, op) };
            return true;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { x86_64::reduce_f32_op_avx2(dst, src, count, op) };
            return true;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::reduce_f32_op_neon(dst, src, count, op) };
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// Try SIMD-accelerated f64 reduction. Returns `true` if handled.
///
/// # Safety
/// `dst` and `src` must both have at least `count * 8` bytes.
pub(crate) unsafe fn reduce_f64_simd(
    dst: &mut [u8],
    src: &[u8],
    count: usize,
    op: ReduceOp,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { x86_64::reduce_f64_op_avx512(dst, src, count, op) };
            return true;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { x86_64::reduce_f64_op_avx2(dst, src, count, op) };
            return true;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::reduce_f64_op_neon(dst, src, count, op) };
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// Try SIMD-accelerated bf16 reduction. Returns `true` if handled.
///
/// # Safety
/// `dst` and `src` must both have at least `count * 2` bytes.
pub(crate) unsafe fn reduce_bf16_simd(
    dst: &mut [u8],
    src: &[u8],
    count: usize,
    op: ReduceOp,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { x86_64::reduce_bf16_op_avx2(dst, src, count, op) };
            return true;
        }
    }
    // aarch64: scalar fallback (NEON fp16 support inconsistent across chips)
    let _ = (dst, src, count, op);
    false
}

/// Try SIMD-accelerated f16 reduction. Returns `true` if handled.
///
/// # Safety
/// `dst` and `src` must both have at least `count * 2` bytes.
pub(crate) unsafe fn reduce_f16_simd(
    dst: &mut [u8],
    src: &[u8],
    count: usize,
    op: ReduceOp,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            unsafe { x86_64::reduce_f16_op_f16c(dst, src, count, op) };
            return true;
        }
    }
    // aarch64: scalar fallback
    let _ = (dst, src, count, op);
    false
}
