//! AVX2 and AVX-512 SIMD reduction kernels for x86_64.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::types::ReduceOp;

// ── f32 ──────────────────────────────────────────────────────────────

#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn reduce_f32_op_avx512(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    let dp = dst.as_mut_ptr() as *mut f32;
    let sp = src.as_ptr() as *const f32;
    let chunks = count / 16;
    let tail = count % 16;

    for i in 0..chunks {
        let off = i * 16;
        unsafe {
            let a = _mm512_loadu_ps(dp.add(off));
            let b = _mm512_loadu_ps(sp.add(off));
            let r = match op {
                ReduceOp::Sum => _mm512_add_ps(a, b),
                ReduceOp::Prod => _mm512_mul_ps(a, b),
                ReduceOp::Min => _mm512_min_ps(a, b),
                ReduceOp::Max => _mm512_max_ps(a, b),
            };
            _mm512_storeu_ps(dp.add(off), r);
        }
    }

    if tail > 0 {
        unsafe {
            reduce_f32_op_avx2(&mut dst[chunks * 64..], &src[chunks * 64..], tail, op);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn reduce_f32_op_avx2(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    let dp = dst.as_mut_ptr() as *mut f32;
    let sp = src.as_ptr() as *const f32;
    let chunks = count / 8;
    let tail = count % 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let a = _mm256_loadu_ps(dp.add(off));
            let b = _mm256_loadu_ps(sp.add(off));
            let r = match op {
                ReduceOp::Sum => _mm256_add_ps(a, b),
                ReduceOp::Prod => _mm256_mul_ps(a, b),
                ReduceOp::Min => _mm256_min_ps(a, b),
                ReduceOp::Max => _mm256_max_ps(a, b),
            };
            _mm256_storeu_ps(dp.add(off), r);
        }
    }

    let base = chunks * 8;
    for i in 0..tail {
        let idx = base + i;
        unsafe {
            let a = *dp.add(idx);
            let b = *sp.add(idx);
            *dp.add(idx) = scalar_op_f32(a, b, op);
        }
    }
}

// ── f64 ──────────────────────────────────────────────────────────────

#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn reduce_f64_op_avx512(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    let dp = dst.as_mut_ptr() as *mut f64;
    let sp = src.as_ptr() as *const f64;
    let chunks = count / 8;
    let tail = count % 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let a = _mm512_loadu_pd(dp.add(off));
            let b = _mm512_loadu_pd(sp.add(off));
            let r = match op {
                ReduceOp::Sum => _mm512_add_pd(a, b),
                ReduceOp::Prod => _mm512_mul_pd(a, b),
                ReduceOp::Min => _mm512_min_pd(a, b),
                ReduceOp::Max => _mm512_max_pd(a, b),
            };
            _mm512_storeu_pd(dp.add(off), r);
        }
    }

    if tail > 0 {
        unsafe {
            reduce_f64_op_avx2(&mut dst[chunks * 64..], &src[chunks * 64..], tail, op);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn reduce_f64_op_avx2(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    let dp = dst.as_mut_ptr() as *mut f64;
    let sp = src.as_ptr() as *const f64;
    let chunks = count / 4;
    let tail = count % 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let a = _mm256_loadu_pd(dp.add(off));
            let b = _mm256_loadu_pd(sp.add(off));
            let r = match op {
                ReduceOp::Sum => _mm256_add_pd(a, b),
                ReduceOp::Prod => _mm256_mul_pd(a, b),
                ReduceOp::Min => _mm256_min_pd(a, b),
                ReduceOp::Max => _mm256_max_pd(a, b),
            };
            _mm256_storeu_pd(dp.add(off), r);
        }
    }

    let base = chunks * 4;
    for i in 0..tail {
        let idx = base + i;
        unsafe {
            let a = *dp.add(idx);
            let b = *sp.add(idx);
            *dp.add(idx) = scalar_op_f64(a, b, op);
        }
    }
}

// ── bf16 (bit-shift to f32, reduce, shift back) ─────────────────────

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn reduce_bf16_op_avx2(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    let dp = dst.as_mut_ptr() as *mut u16;
    let sp = src.as_ptr() as *const u16;
    let chunks = count / 8;
    let tail = count % 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let a_u16 = _mm_loadu_si128(dp.add(off) as *const __m128i);
            let b_u16 = _mm_loadu_si128(sp.add(off) as *const __m128i);
            let a_i32 = _mm256_cvtepu16_epi32(a_u16);
            let b_i32 = _mm256_cvtepu16_epi32(b_u16);
            let a_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(a_i32, 16));
            let b_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(b_i32, 16));

            let r_f32 = match op {
                ReduceOp::Sum => _mm256_add_ps(a_f32, b_f32),
                ReduceOp::Prod => _mm256_mul_ps(a_f32, b_f32),
                ReduceOp::Min => _mm256_min_ps(a_f32, b_f32),
                ReduceOp::Max => _mm256_max_ps(a_f32, b_f32),
            };

            let r_i32 = _mm256_srli_epi32(_mm256_castps_si256(r_f32), 16);
            let lo = _mm256_castsi256_si128(r_i32);
            let hi = _mm256_extracti128_si256(r_i32, 1);
            let packed = _mm_packus_epi32(lo, hi);
            _mm_storeu_si128(dp.add(off) as *mut __m128i, packed);
        }
    }

    let base = chunks * 8;
    for i in 0..tail {
        let idx = base + i;
        unsafe {
            let a = f32::from_bits((*dp.add(idx) as u32) << 16);
            let b = f32::from_bits((*sp.add(idx) as u32) << 16);
            let r = scalar_op_f32(a, b, op);
            *dp.add(idx) = (r.to_bits() >> 16) as u16;
        }
    }
}

// ── f16 (F16C convert to f32, reduce, convert back) ─────────────────

#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn reduce_f16_op_f16c(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    let dp = dst.as_mut_ptr() as *mut u16;
    let sp = src.as_ptr() as *const u16;
    let chunks = count / 8;
    let tail = count % 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let a_h = _mm_loadu_si128(dp.add(off) as *const __m128i);
            let b_h = _mm_loadu_si128(sp.add(off) as *const __m128i);
            let a_f32 = _mm256_cvtph_ps(a_h);
            let b_f32 = _mm256_cvtph_ps(b_h);

            let r_f32 = match op {
                ReduceOp::Sum => _mm256_add_ps(a_f32, b_f32),
                ReduceOp::Prod => _mm256_mul_ps(a_f32, b_f32),
                ReduceOp::Min => _mm256_min_ps(a_f32, b_f32),
                ReduceOp::Max => _mm256_max_ps(a_f32, b_f32),
            };

            let r_h = _mm256_cvtps_ph(r_f32, 0x00);
            _mm_storeu_si128(dp.add(off) as *mut __m128i, r_h);
        }
    }

    let base = chunks * 8;
    for i in 0..tail {
        let idx = base + i;
        unsafe {
            let a = crate::reduce::F16(*dp.add(idx)).to_f32();
            let b = crate::reduce::F16(*sp.add(idx)).to_f32();
            let r = scalar_op_f32(a, b, op);
            *dp.add(idx) = crate::reduce::F16::from_f32(r).0;
        }
    }
}

// ── Scalar helpers ───────────────────────────────────────────────────

#[inline]
fn scalar_op_f32(a: f32, b: f32, op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum => a + b,
        ReduceOp::Prod => a * b,
        ReduceOp::Min => a.min(b),
        ReduceOp::Max => a.max(b),
    }
}

#[inline]
fn scalar_op_f64(a: f64, b: f64, op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum => a + b,
        ReduceOp::Prod => a * b,
        ReduceOp::Min => a.min(b),
        ReduceOp::Max => a.max(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_op_f32(a: f32, b: f32, op: ReduceOp) -> f32 {
        scalar_op_f32(a, b, op)
    }

    #[test]
    fn test_f32_avx2_all_ops() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Min, ReduceOp::Max] {
            let count = 37;
            let mut dst: Vec<f32> = (0..count).map(|i| i as f32 * 1.5 + 0.1).collect();
            let src: Vec<f32> = (0..count).map(|i| i as f32 * 2.3 + 0.7).collect();
            let expected: Vec<f32> = dst
                .iter()
                .zip(&src)
                .map(|(&a, &b)| apply_op_f32(a, b, op))
                .collect();

            let dst_bytes =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, count * 4) };
            let src_bytes =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, count * 4) };
            unsafe { reduce_f32_op_avx2(dst_bytes, src_bytes, count, op) };

            for (i, (&got, &exp)) in dst.iter().zip(&expected).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "f32 avx2 {op:?} mismatch at {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn test_f64_avx2_all_ops() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Min, ReduceOp::Max] {
            let count = 19;
            let mut dst: Vec<f64> = (0..count).map(|i| i as f64 * 1.5 + 0.1).collect();
            let src: Vec<f64> = (0..count).map(|i| i as f64 * 2.3 + 0.7).collect();
            let expected: Vec<f64> = dst
                .iter()
                .zip(&src)
                .map(|(&a, &b)| scalar_op_f64(a, b, op))
                .collect();

            let dst_bytes =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, count * 8) };
            let src_bytes =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, count * 8) };
            unsafe { reduce_f64_op_avx2(dst_bytes, src_bytes, count, op) };

            for (i, (&got, &exp)) in dst.iter().zip(&expected).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-10,
                    "f64 avx2 {op:?} mismatch at {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn test_bf16_avx2_sum() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let count = 19;
        let a_f32: Vec<f32> = (0..count).map(|i| i as f32 * 1.5).collect();
        let b_f32: Vec<f32> = (0..count).map(|i| i as f32 * 2.0 + 1.0).collect();

        let mut dst: Vec<u16> = a_f32.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
        let src: Vec<u16> = b_f32.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();

        let dst_bytes =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, count * 2) };
        let src_bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, count * 2) };
        unsafe { reduce_bf16_op_avx2(dst_bytes, src_bytes, count, ReduceOp::Sum) };

        for i in 0..count {
            let got = f32::from_bits((dst[i] as u32) << 16);
            let exp = a_f32[i] + b_f32[i];
            assert!(
                (got - exp).abs() < exp.abs() * 0.02 + 0.1,
                "bf16 avx2 sum mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_f16_f16c() {
        if !is_x86_feature_detected!("f16c") {
            return;
        }
        let count = 19;
        let a_vals: Vec<f32> = (0..count).map(|i| i as f32 * 0.5 + 0.1).collect();
        let b_vals: Vec<f32> = (0..count).map(|i| i as f32 * 0.3 + 0.2).collect();

        let mut dst: Vec<u16> = a_vals
            .iter()
            .map(|&v| crate::reduce::F16::from_f32(v).0)
            .collect();
        let src: Vec<u16> = b_vals
            .iter()
            .map(|&v| crate::reduce::F16::from_f32(v).0)
            .collect();

        let dst_bytes =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, count * 2) };
        let src_bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, count * 2) };
        unsafe { reduce_f16_op_f16c(dst_bytes, src_bytes, count, ReduceOp::Sum) };

        for i in 0..count {
            let got = crate::reduce::F16(dst[i]).to_f32();
            let exp = a_vals[i] + b_vals[i];
            assert!(
                (got - exp).abs() < exp.abs() * 0.01 + 0.1,
                "f16 f16c sum mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }
}
