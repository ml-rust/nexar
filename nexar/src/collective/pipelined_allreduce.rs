//! Pipelined ring allreduce for large tensors.
//!
//! When the total transfer size exceeds `PIPELINE_THRESHOLD_BYTES`, the
//! scatter-reduce and allgather phases are segmented so that reduction
//! overlaps with network I/O across pipeline stages.

use crate::client::NexarClient;
use crate::collective::helpers::{
    CollectiveTag, collective_recv_with_tag, collective_send_with_tag,
};
use crate::error::{NexarError, Result};
use crate::reduce::reduce_slice;
use crate::types::{DataType, ReduceOp};

/// Tensors smaller than this use the non-pipelined ring.
pub(crate) const PIPELINE_THRESHOLD_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

/// Segment size for pipeline stages.
const PIPELINE_SEGMENT_BYTES: usize = 2 * 1024 * 1024; // 2 MiB

/// Pack pipeline metadata into a single tag for tagged transport.
///
/// Layout: `[63:48] outer_tag | [47:32] ring_step | [31:16] segment | [15:0] phase`
///
/// - `outer_tag`: base tag from the collective operation
/// - `phase`: 0 = scatter-reduce, 1 = allgather
/// - `step`: current ring step (0 to world-2)
/// - `segment`: pipeline segment index within this step
fn pack_tag(outer_tag: u64, phase: u16, step: u16, segment: u16) -> u64 {
    (outer_tag & 0xFFFF) << 48 | (step as u64) << 32 | (segment as u64) << 16 | (phase as u64)
}

/// Compute `(offset, length)` in elements for a segment within a chunk.
fn segment_range(
    chunk_off: usize,
    chunk_len: usize,
    seg: usize,
    num_segs: usize,
    elem_size: usize,
) -> (usize, usize) {
    debug_assert!(num_segs > 0, "num_segs must be > 0");
    debug_assert!(seg < num_segs, "segment index {seg} >= num_segs {num_segs}");

    let chunk_bytes = chunk_len * elem_size;
    let base_seg = chunk_bytes / num_segs;
    let rem = chunk_bytes % num_segs;

    let seg_byte_off = base_seg * seg + seg.min(rem);
    let seg_byte_len = base_seg + if seg < rem { 1 } else { 0 };

    // Align to element boundaries.
    let byte_off = chunk_off * elem_size + seg_byte_off;
    let aligned_off = byte_off / elem_size;
    let aligned_len = seg_byte_len / elem_size;

    (aligned_off, aligned_len)
}

/// Pipelined ring allreduce.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub(crate) async unsafe fn pipelined_ring_allreduce(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;

    if world <= 1 {
        return Ok(());
    }

    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };
    let mut buf = data;

    let base_chunk = count / world;
    let remainder = count % world;

    let chunk_count = |i: usize| -> usize {
        if i < remainder {
            base_chunk + 1
        } else {
            base_chunk
        }
    };

    let chunk_offsets: Vec<usize> = (0..world)
        .scan(0usize, |acc, i| {
            let off = *acc;
            *acc += chunk_count(i);
            Some(off)
        })
        .collect();

    let next = (rank + 1) % world;
    let prev = (rank + world - 1) % world;

    let outer_tag = tag.unwrap_or(0);

    // Phase 1: Pipelined scatter-reduce (N-1 rounds).
    for step in 0..(world - 1) {
        let send_idx = (rank + world - step) % world;
        let send_off = chunk_offsets[send_idx];
        let send_count = chunk_count(send_idx);
        let send_bytes = send_count * elem_size;

        let recv_idx = (rank + world - step - 1) % world;
        let recv_off = chunk_offsets[recv_idx];
        let recv_count = chunk_count(recv_idx);
        let recv_bytes = recv_count * elem_size;

        let num_segs = recv_bytes.max(send_bytes).div_ceil(PIPELINE_SEGMENT_BYTES);
        let num_segs = num_segs.max(1);

        if num_segs <= 1 {
            // No benefit from pipelining for this chunk.
            let send_byte_off = send_off * elem_size;
            let send_data = buf[send_byte_off..send_byte_off + send_bytes].to_vec();
            let step_tag = Some(pack_tag(outer_tag, 0, step as u16, 0));

            let (sr, rr) = tokio::join!(
                collective_send_with_tag(client, next as u32, &send_data, "allreduce", step_tag),
                collective_recv_with_tag(client, prev as u32, "allreduce", step_tag),
            );
            sr?;
            let received = rr?;

            if received.len() != recv_bytes {
                return Err(NexarError::BufferSizeMismatch {
                    expected: recv_bytes,
                    actual: received.len(),
                });
            }
            let recv_byte_off = recv_off * elem_size;
            let dst_slice = &mut buf[recv_byte_off..recv_byte_off + recv_bytes];
            reduce_slice(dst_slice, &received, recv_count, dtype, op)?;
            continue;
        }

        // --- Pipelined path ---

        // Prime: send+recv segment 0.
        let (s0_off, s0_len) = segment_range(send_off, send_count, 0, num_segs, elem_size);
        let s0_byte_off = s0_off * elem_size;
        let s0_byte_len = s0_len * elem_size;
        let send_data_0 = buf[s0_byte_off..s0_byte_off + s0_byte_len].to_vec();
        let tag_0 = Some(pack_tag(outer_tag, 0, step as u16, 0));

        let (sr, rr) = tokio::join!(
            collective_send_with_tag(client, next as u32, &send_data_0, "allreduce", tag_0),
            collective_recv_with_tag(client, prev as u32, "allreduce", tag_0),
        );
        sr?;
        let mut prev_received = rr?;

        // Pipeline loop: reduce seg[k-1] while sending/receiving seg[k].
        for seg in 1..num_segs {
            // Reduce previous segment.
            let (pr_off, pr_len) =
                segment_range(recv_off, recv_count, seg - 1, num_segs, elem_size);
            let pr_byte_off = pr_off * elem_size;
            let pr_byte_len = pr_len * elem_size;

            if prev_received.len() != pr_byte_len {
                return Err(NexarError::BufferSizeMismatch {
                    expected: pr_byte_len,
                    actual: prev_received.len(),
                });
            }
            reduce_slice(
                &mut buf[pr_byte_off..pr_byte_off + pr_byte_len],
                &prev_received,
                pr_len,
                dtype,
                op,
            )?;

            // Send+recv current segment.
            let (sk_off, sk_len) = segment_range(send_off, send_count, seg, num_segs, elem_size);
            let sk_byte_off = sk_off * elem_size;
            let sk_byte_len = sk_len * elem_size;
            let send_data_k = buf[sk_byte_off..sk_byte_off + sk_byte_len].to_vec();
            let tag_k = Some(pack_tag(outer_tag, 0, step as u16, seg as u16));

            let (sr, rr) = tokio::join!(
                collective_send_with_tag(client, next as u32, &send_data_k, "allreduce", tag_k,),
                collective_recv_with_tag(client, prev as u32, "allreduce", tag_k),
            );
            sr?;
            prev_received = rr?;
        }

        // Drain: reduce final segment.
        let (fr_off, fr_len) =
            segment_range(recv_off, recv_count, num_segs - 1, num_segs, elem_size);
        let fr_byte_off = fr_off * elem_size;
        let fr_byte_len = fr_len * elem_size;

        if prev_received.len() != fr_byte_len {
            return Err(NexarError::BufferSizeMismatch {
                expected: fr_byte_len,
                actual: prev_received.len(),
            });
        }
        reduce_slice(
            &mut buf[fr_byte_off..fr_byte_off + fr_byte_len],
            &prev_received,
            fr_len,
            dtype,
            op,
        )?;
    }

    // Phase 2: Pipelined allgather (N-1 rounds).
    for step in 0..(world - 1) {
        let send_idx = (rank + world + 1 - step) % world;
        let send_off = chunk_offsets[send_idx];
        let send_count_s = chunk_count(send_idx);
        let send_bytes = send_count_s * elem_size;

        let recv_idx = (rank + world - step) % world;
        let recv_off = chunk_offsets[recv_idx];
        let recv_count_r = chunk_count(recv_idx);
        let recv_bytes = recv_count_r * elem_size;

        let num_segs = recv_bytes.max(send_bytes).div_ceil(PIPELINE_SEGMENT_BYTES);
        let num_segs = num_segs.max(1);

        if num_segs <= 1 {
            let send_byte_off = send_off * elem_size;
            let send_data = buf[send_byte_off..send_byte_off + send_bytes].to_vec();
            let step_tag = Some(pack_tag(outer_tag, 1, step as u16, 0));

            let (sr, rr) = tokio::join!(
                collective_send_with_tag(client, next as u32, &send_data, "allreduce", step_tag),
                collective_recv_with_tag(client, prev as u32, "allreduce", step_tag),
            );
            sr?;
            let received = rr?;

            if received.len() != recv_bytes {
                return Err(NexarError::BufferSizeMismatch {
                    expected: recv_bytes,
                    actual: received.len(),
                });
            }
            let recv_byte_off = recv_off * elem_size;
            buf[recv_byte_off..recv_byte_off + recv_bytes].copy_from_slice(&received);
            continue;
        }

        // Pipelined allgather: send/recv segment by segment, copy on arrival.
        for seg in 0..num_segs {
            let (sk_off, sk_len) = segment_range(send_off, send_count_s, seg, num_segs, elem_size);
            let sk_byte_off = sk_off * elem_size;
            let sk_byte_len = sk_len * elem_size;
            let send_data = buf[sk_byte_off..sk_byte_off + sk_byte_len].to_vec();
            let tag_k = Some(pack_tag(outer_tag, 1, step as u16, seg as u16));

            let (sr, rr) = tokio::join!(
                collective_send_with_tag(client, next as u32, &send_data, "allreduce", tag_k),
                collective_recv_with_tag(client, prev as u32, "allreduce", tag_k),
            );
            sr?;
            let received = rr?;

            let (rk_off, rk_len) = segment_range(recv_off, recv_count_r, seg, num_segs, elem_size);
            let rk_byte_off = rk_off * elem_size;
            let rk_byte_len = rk_len * elem_size;

            if received.len() != rk_byte_len {
                return Err(NexarError::BufferSizeMismatch {
                    expected: rk_byte_len,
                    actual: received.len(),
                });
            }
            buf[rk_byte_off..rk_byte_off + rk_byte_len].copy_from_slice(&received);
        }
    }

    unsafe { client.adapter().receive_to_device(&buf, ptr)? };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_tag_roundtrip() {
        let tag = pack_tag(0xABCD, 3, 7, 15);
        assert_eq!((tag >> 48) & 0xFFFF, 0xABCD);
        assert_eq!((tag >> 32) & 0xFFFF, 7);
        assert_eq!((tag >> 16) & 0xFFFF, 15);
        assert_eq!(tag & 0xFFFF, 3);
    }

    #[test]
    fn test_segment_range_even_split() {
        // 1024 elements of 4 bytes = 4096 bytes, 2 segments
        let (off0, len0) = segment_range(0, 1024, 0, 2, 4);
        let (off1, len1) = segment_range(0, 1024, 1, 2, 4);
        assert_eq!(off0, 0);
        assert_eq!(len0, 512);
        assert_eq!(off1, 512);
        assert_eq!(len1, 512);
    }

    #[test]
    fn test_segment_range_single() {
        let (off, len) = segment_range(100, 50, 0, 1, 4);
        assert_eq!(off, 100);
        assert_eq!(len, 50);
    }
}
