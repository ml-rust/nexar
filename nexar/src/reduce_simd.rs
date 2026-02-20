//! SIMD-accelerated reduction kernels for x86_64.

/// Reduce f32 sum using AVX2 if available, scalar fallback otherwise.
///
/// # Safety
/// `dst` and `src` must both have at least `count * 4` bytes.
pub(crate) unsafe fn reduce_f32_sum_simd(dst: &mut [u8], src: &[u8], count: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { reduce_f32_sum_avx2(dst, src, count) };
            return;
        }
    }
    reduce_f32_sum_scalar(dst, src, count);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_f32_sum_avx2(dst: &mut [u8], src: &[u8], count: usize) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let dst_ptr = dst.as_mut_ptr() as *mut f32;
    let src_ptr = src.as_ptr() as *const f32;

    let simd_count = count / 8;
    let remainder = count % 8;

    for i in 0..simd_count {
        let off = i * 8;
        unsafe {
            let a = _mm256_loadu_ps(dst_ptr.add(off));
            let b = _mm256_loadu_ps(src_ptr.add(off));
            let r = _mm256_add_ps(a, b);
            _mm256_storeu_ps(dst_ptr.add(off), r);
        }
    }

    // Scalar tail
    let tail_start = simd_count * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        unsafe {
            *dst_ptr.add(idx) += *src_ptr.add(idx);
        }
    }
}

fn reduce_f32_sum_scalar(dst: &mut [u8], src: &[u8], count: usize) {
    let dst_ptr = dst.as_mut_ptr() as *mut f32;
    let src_ptr = src.as_ptr() as *const f32;
    for i in 0..count {
        unsafe {
            *dst_ptr.add(i) += *src_ptr.add(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vs_scalar_identical() {
        let count = 37; // non-multiple of 8
        let mut dst_simd: Vec<f32> = (0..count).map(|i| i as f32 * 1.5).collect();
        let src: Vec<f32> = (0..count).map(|i| i as f32 * 2.3 + 0.7).collect();
        let mut dst_scalar = dst_simd.clone();

        let dst_s_bytes =
            unsafe { std::slice::from_raw_parts_mut(dst_simd.as_mut_ptr() as *mut u8, count * 4) };
        let src_bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, count * 4) };
        unsafe { reduce_f32_sum_simd(dst_s_bytes, src_bytes, count) };

        let dst_sc_bytes = unsafe {
            std::slice::from_raw_parts_mut(dst_scalar.as_mut_ptr() as *mut u8, count * 4)
        };
        reduce_f32_sum_scalar(dst_sc_bytes, src_bytes, count);

        assert_eq!(dst_simd, dst_scalar);
    }
}
