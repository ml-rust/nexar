use crate::client::NexarClient;
use crate::collective::helpers::{ceil_log2, collective_recv, collective_send};
use crate::error::{NexarError, Result};
use crate::reduce::{identity_slice, reduce_slice};
use crate::types::{DataType, ReduceOp};

/// Inclusive prefix scan (prefix sum) across all ranks.
///
/// After completion, rank `i` holds the reduction of ranks 0..=i.
/// Uses the Hillis-Steele parallel scan pattern: in round `r`,
/// rank `i` receives from rank `i - 2^r` (if it exists) and reduces.
/// This takes ceil(log₂(N)) rounds.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn inclusive_scan(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
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

    // Hillis-Steele: ceil(log2(world)) rounds.
    let num_rounds = ceil_log2(world as u32) as usize;

    for round in 0..num_rounds {
        let distance = 1 << round;

        // Ranks that have a source to receive from: rank >= distance.
        // Ranks that need to send: rank + distance < world.
        let should_send = rank + distance < world;
        let should_recv = rank >= distance;
        let source = rank.wrapping_sub(distance); // only valid if should_recv

        match (should_send, should_recv) {
            (true, true) => {
                let dest = rank + distance;
                let send_data = buf.clone();
                let (send_result, recv_result) = tokio::join!(
                    collective_send(client, dest as u32, &send_data, "scan"),
                    collective_recv(client, source as u32, "scan"),
                );
                send_result?;
                let received = recv_result?;
                if received.len() != total_bytes {
                    return Err(NexarError::BufferSizeMismatch {
                        expected: total_bytes,
                        actual: received.len(),
                    });
                }
                reduce_slice(&mut buf, &received, count, dtype, op)?;
            }
            (true, false) => {
                let dest = rank + distance;
                collective_send(client, dest as u32, &buf, "scan").await?;
            }
            (false, true) => {
                let received = collective_recv(client, source as u32, "scan").await?;
                if received.len() != total_bytes {
                    return Err(NexarError::BufferSizeMismatch {
                        expected: total_bytes,
                        actual: received.len(),
                    });
                }
                reduce_slice(&mut buf, &received, count, dtype, op)?;
            }
            (false, false) => {
                // Nothing to do this round.
            }
        }
    }

    unsafe { client.adapter().receive_to_device(&buf, ptr)? };

    Ok(())
}

/// Exclusive prefix scan across all ranks.
///
/// After completion, rank `i` holds the reduction of ranks `0..i` (exclusive).
/// Rank 0 receives the identity element for the given operation.
/// Equivalent to MPI's `MPI_Exscan`.
///
/// Algorithm: save original data, run inclusive scan, then shift results
/// left by one rank. Rank 0 gets the identity element.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn exclusive_scan(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;
    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    if world <= 1 {
        // Single node: fill with identity element.
        let id = identity_slice(count, dtype, op)?;
        unsafe { client.adapter().receive_to_device(&id, ptr)? };
        return Ok(());
    }

    // Run inclusive scan in-place.
    unsafe { inclusive_scan(client, ptr, count, dtype, op).await? };

    // After inclusive scan, rank i holds reduce(0..=i).
    // For exclusive scan, rank i needs reduce(0..i) = inclusive_scan[i-1].
    // So rank i sends its inclusive result to rank i+1.
    // Rank 0 gets the identity element.

    let inclusive_data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };

    let should_send = rank + 1 < world;
    let should_recv = rank > 0;

    match (should_send, should_recv) {
        (true, true) => {
            let (send_res, recv_res) = tokio::join!(
                collective_send(client, (rank + 1) as u32, &inclusive_data, "exscan"),
                collective_recv(client, (rank - 1) as u32, "exscan"),
            );
            send_res?;
            let received = recv_res?;
            if received.len() != total_bytes {
                return Err(NexarError::BufferSizeMismatch {
                    expected: total_bytes,
                    actual: received.len(),
                });
            }
            unsafe { client.adapter().receive_to_device(&received, ptr)? };
        }
        (true, false) => {
            // Rank 0: send inclusive result to rank 1, write identity to own buffer.
            collective_send(client, (rank + 1) as u32, &inclusive_data, "exscan").await?;
            let id = identity_slice(count, dtype, op)?;
            unsafe { client.adapter().receive_to_device(&id, ptr)? };
        }
        (false, true) => {
            // Last rank: recv from prev.
            let received = collective_recv(client, (rank - 1) as u32, "exscan").await?;
            if received.len() != total_bytes {
                return Err(NexarError::BufferSizeMismatch {
                    expected: total_bytes,
                    actual: received.len(),
                });
            }
            unsafe { client.adapter().receive_to_device(&received, ptr)? };
        }
        (false, false) => unreachable!("world > 1"),
    }

    Ok(())
}
