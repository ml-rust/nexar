//! NEON SIMD reduction kernels for aarch64.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::types::ReduceOp;

// ── f32 (NEON, 4-wide) ──────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn reduce_f32_op_neon(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    unsafe {
        let dp = dst.as_mut_ptr() as *mut f32;
        let sp = src.as_ptr() as *const f32;
        let chunks = count / 4;
        let tail = count % 4;

        for i in 0..chunks {
            let off = i * 4;
            let a = vld1q_f32(dp.add(off));
            let b = vld1q_f32(sp.add(off));
            let r = match op {
                ReduceOp::Sum => vaddq_f32(a, b),
                ReduceOp::Prod => vmulq_f32(a, b),
                ReduceOp::Min => vminq_f32(a, b),
                ReduceOp::Max => vmaxq_f32(a, b),
            };
            vst1q_f32(dp.add(off), r);
        }

        let base = chunks * 4;
        for i in 0..tail {
            let idx = base + i;
            let a = *dp.add(idx);
            let b = *sp.add(idx);
            *dp.add(idx) = scalar_op_f32(a, b, op);
        }
    }
}

// ── f64 (NEON, 2-wide) ──────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn reduce_f64_op_neon(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    unsafe {
        let dp = dst.as_mut_ptr() as *mut f64;
        let sp = src.as_ptr() as *const f64;
        let chunks = count / 2;
        let tail = count % 2;

        for i in 0..chunks {
            let off = i * 2;
            let a = vld1q_f64(dp.add(off));
            let b = vld1q_f64(sp.add(off));
            let r = match op {
                ReduceOp::Sum => vaddq_f64(a, b),
                ReduceOp::Prod => vmulq_f64(a, b),
                ReduceOp::Min => vminq_f64(a, b),
                ReduceOp::Max => vmaxq_f64(a, b),
            };
            vst1q_f64(dp.add(off), r);
        }

        if tail > 0 {
            let idx = chunks * 2;
            let a = *dp.add(idx);
            let b = *sp.add(idx);
            *dp.add(idx) = scalar_op_f64(a, b, op);
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
