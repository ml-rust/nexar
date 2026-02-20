use crate::client::NexarClient;
use crate::collective::helpers::{
    ChunkLayout, CollectiveTag, collective_recv_with_tag, collective_send_with_tag,
};
use crate::error::{NexarError, Result};
use crate::reduce::reduce_slice;
use crate::types::{DataType, ReduceOp};

// Algorithm thresholds are read from client.config().large_msg_bytes
// and client.config().ring_max_world at runtime.

/// Tagged variant for non-blocking collectives.
pub(crate) async unsafe fn ring_allreduce_with_tag(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size() as usize;
    let total_bytes = count * dtype.size_in_bytes();

    let cfg = client.config();
    if total_bytes >= cfg.large_msg_bytes {
        // Large messages: pipelined ring is bandwidth-optimal.
        unsafe {
            crate::collective::pipelined_allreduce::pipelined_ring_allreduce(
                client, ptr, count, dtype, op, tag,
            )
            .await
        }
    } else if world <= cfg.ring_max_world {
        // Small world: ring has lower constant overhead and handles
        // non-power-of-2 world sizes without the excess-rank exchange
        // that halving-doubling requires.
        unsafe { ring_allreduce_impl(client, ptr, count, dtype, op, tag).await }
    } else {
        // Large world (N > 8), sub-8MiB message: halving-doubling is
        // latency-optimal with O(log N) steps vs ring's O(N).
        unsafe { halving_doubling_allreduce(client, ptr, count, dtype, op, tag).await }
    }
}

/// Ring-allreduce: in-place reduce across all ranks.
///
/// Algorithm:
/// 1. Scatter-reduce: N-1 rounds. Each rank sends one chunk to the next rank
///    and receives one chunk from the previous rank, reducing in-place.
/// 2. Allgather: N-1 rounds. Each rank sends its fully-reduced chunk to the
///    next rank and receives from the previous rank.
///
/// After completion, `ptr` on every rank contains the reduced result of
/// all ranks' original data.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
async unsafe fn ring_allreduce_impl(
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

    let layout = ChunkLayout::new(count, world);

    let next = (rank + 1) % world;
    let prev = (rank + world - 1) % world;

    // Phase 1: Scatter-reduce (N-1 rounds).
    for step in 0..(world - 1) {
        let send_idx = (rank + world - step) % world;
        let send_off = layout.offsets[send_idx] * elem_size;
        let send_len = layout.chunk_count(send_idx) * elem_size;

        let recv_idx = (rank + world - step - 1) % world;
        let recv_off = layout.offsets[recv_idx] * elem_size;
        let recv_count = layout.chunk_count(recv_idx);
        let recv_len = recv_count * elem_size;

        // Zero-copy: extract send slice before the join so recv can borrow buf mutably.
        // Send and recv operate on different chunks, so this is safe.
        let send_snapshot = buf[send_off..send_off + send_len].to_vec();

        let (_, received) = tokio::try_join!(
            collective_send_with_tag(client, next as u32, &send_snapshot, "allreduce", tag),
            collective_recv_with_tag(client, prev as u32, "allreduce", tag),
        )?;

        if received.len() != recv_len {
            return Err(NexarError::BufferSizeMismatch {
                expected: recv_len,
                actual: received.len(),
            });
        }
        let dst_slice = &mut buf[recv_off..recv_off + recv_len];
        reduce_slice(dst_slice, &received, recv_count, dtype, op)?;
    }

    // Phase 2: Allgather (N-1 rounds).
    for step in 0..(world - 1) {
        let send_idx = (rank + world + 1 - step) % world;
        let send_off = layout.offsets[send_idx] * elem_size;
        let send_len = layout.chunk_count(send_idx) * elem_size;

        let recv_idx = (rank + world - step) % world;
        let recv_off = layout.offsets[recv_idx] * elem_size;
        let recv_len = layout.chunk_count(recv_idx) * elem_size;

        // In allgather, send chunk is already fully reduced and won't be
        // modified by recv. Still need to copy because tokio::try_join! borrows
        // the future args, and we can't split borrow buf in safe Rust.
        let send_snapshot = buf[send_off..send_off + send_len].to_vec();

        let (_, received) = tokio::try_join!(
            collective_send_with_tag(client, next as u32, &send_snapshot, "allreduce", tag),
            collective_recv_with_tag(client, prev as u32, "allreduce", tag),
        )?;

        if received.len() != recv_len {
            return Err(NexarError::BufferSizeMismatch {
                expected: recv_len,
                actual: received.len(),
            });
        }
        buf[recv_off..recv_off + recv_len].copy_from_slice(&received);
    }

    unsafe { client.adapter().receive_to_device(&buf, ptr)? };

    Ok(())
}

/// Recursive halving-doubling allreduce: 2*log₂(N) communication steps.
///
/// For power-of-2 world sizes, this is straightforward. For non-power-of-2,
/// excess ranks first send their data to a partner in the lower range,
/// reducing to a power-of-2 problem, then receive the result back.
///
/// Phase 1 (Reduce-scatter): log₂(N) rounds where rank i exchanges with
///   rank `i XOR 2^r`. Each side reduces on its respective half.
/// Phase 2 (Allgather): log₂(N) rounds reversing phase 1 to reconstruct
///   the full reduced result on all ranks.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
async unsafe fn halving_doubling_allreduce(
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

    // Find largest power of 2 <= world.
    let p2 = world.next_power_of_two() >> if world.is_power_of_two() { 0 } else { 1 };
    let excess = world - p2;

    // Step 1: Handle non-power-of-2 by reducing excess ranks into the lower p2.
    // Ranks [p2..world) send their data to rank [0..excess).
    // Those "extra" ranks don't participate in the main algorithm.
    let mut virtual_rank: Option<usize> = None; // None means this rank is "extra" (sits out)

    if rank < excess {
        // This rank receives from its excess partner.
        let partner = rank + p2;
        let received = collective_recv_with_tag(client, partner as u32, "allreduce", tag).await?;
        if received.len() != total_bytes {
            return Err(NexarError::BufferSizeMismatch {
                expected: total_bytes,
                actual: received.len(),
            });
        }
        reduce_slice(&mut buf, &received, count, dtype, op)?;
        virtual_rank = Some(rank);
    } else if rank >= p2 {
        // This is an excess rank — send data and wait.
        let partner = rank - p2;
        collective_send_with_tag(client, partner as u32, &buf, "allreduce", tag).await?;
        // Will receive result from partner after the algorithm completes.
    } else {
        // rank in [excess..p2) — participates directly.
        virtual_rank = Some(rank);
    }

    if let Some(vrank) = virtual_rank {
        // Main halving-doubling on p2 ranks.
        let log2 = p2.trailing_zeros() as usize;

        // Phase 1: Reduce-scatter — each round halves the active data range.
        // Track which slice of the count this rank is responsible for.
        let mut slice_start = 0usize;
        let mut slice_len = count;

        for round in 0..log2 {
            let partner_vrank = vrank ^ (1 << round);

            // Map virtual rank back to real rank. For the halving-doubling
            // algorithm, virtual ranks map 1:1 to real ranks in [0..p2).
            let partner_real = partner_vrank;

            let half = slice_len / 2;
            let half_rem = slice_len - half; // upper half gets remainder for odd splits

            // Lower-indexed virtual rank takes the lower half, higher takes upper.
            let (send_start, send_len, keep_start, keep_len) = if vrank < partner_vrank {
                // Send upper half, keep lower half.
                (slice_start + half, half_rem, slice_start, half)
            } else {
                // Send lower half, keep upper half.
                (slice_start, half, slice_start + half, half_rem)
            };

            let send_off = send_start * elem_size;
            let send_bytes = send_len * elem_size;
            let keep_off = keep_start * elem_size;
            let keep_bytes = keep_len * elem_size;

            let send_data = buf[send_off..send_off + send_bytes].to_vec();

            let (_, received) = tokio::try_join!(
                collective_send_with_tag(client, partner_real as u32, &send_data, "allreduce", tag),
                collective_recv_with_tag(client, partner_real as u32, "allreduce", tag),
            )?;

            if received.len() != keep_bytes {
                return Err(NexarError::BufferSizeMismatch {
                    expected: keep_bytes,
                    actual: received.len(),
                });
            }

            // Reduce the received data into our kept portion.
            let dst = &mut buf[keep_off..keep_off + keep_bytes];
            reduce_slice(dst, &received, keep_len, dtype, op)?;

            slice_start = keep_start;
            slice_len = keep_len;
        }

        // Phase 2: Allgather — reverse of reduce-scatter.
        // Reconstruct full buffer by exchanging reduced chunks in reverse order.
        for round in (0..log2).rev() {
            let partner_vrank = vrank ^ (1 << round);
            let partner_real = partner_vrank;

            // We hold [slice_start..slice_start+slice_len]. Partner holds the other half.
            // After this round we double our range.
            let send_off = slice_start * elem_size;
            let send_bytes = slice_len * elem_size;

            let send_data = buf[send_off..send_off + send_bytes].to_vec();

            let (_, received) = tokio::try_join!(
                collective_send_with_tag(client, partner_real as u32, &send_data, "allreduce", tag),
                collective_recv_with_tag(client, partner_real as u32, "allreduce", tag),
            )?;

            // Place received data in the partner's portion.
            let (new_start, new_len) = if vrank < partner_vrank {
                // We have lower half, partner sends upper half.
                let recv_start = slice_start + slice_len;
                let recv_len = received.len() / elem_size;
                let recv_off = recv_start * elem_size;
                buf[recv_off..recv_off + received.len()].copy_from_slice(&received);
                (slice_start, slice_len + recv_len)
            } else {
                // We have upper half, partner sends lower half.
                let recv_len = received.len() / elem_size;
                let recv_start = slice_start - recv_len;
                let recv_off = recv_start * elem_size;
                buf[recv_off..recv_off + received.len()].copy_from_slice(&received);
                (recv_start, recv_len + slice_len)
            };

            slice_start = new_start;
            slice_len = new_len;
        }
    }

    // Step 3: Send results back to excess ranks.
    if rank < excess {
        let partner = rank + p2;
        collective_send_with_tag(client, partner as u32, &buf, "allreduce", tag).await?;
    } else if rank >= p2 {
        let partner = rank - p2;
        let received = collective_recv_with_tag(client, partner as u32, "allreduce", tag).await?;
        if received.len() != total_bytes {
            return Err(NexarError::BufferSizeMismatch {
                expected: total_bytes,
                actual: received.len(),
            });
        }
        buf.copy_from_slice(&received);
    }

    unsafe { client.adapter().receive_to_device(&buf, ptr)? };

    Ok(())
}
