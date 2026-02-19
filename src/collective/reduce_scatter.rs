use crate::client::NexarClient;
use crate::collective::{collective_recv, collective_send};
use crate::error::{NexarError, Result};
use crate::reduce::reduce_slice;
use crate::types::{DataType, ReduceOp};

/// Ring reduce-scatter: reduce across all ranks, each rank gets a different
/// slice of the result.
///
/// Input: each rank has `count * world_size` elements at `send_ptr`.
/// Output: rank `i` gets the reduced chunk `i` (of size `count`) at `recv_ptr`.
///
/// # Safety
/// - `send_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
/// - `recv_ptr` must point to at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn ring_reduce_scatter(
    client: &NexarClient,
    send_ptr: u64,
    recv_ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;

    let elem_size = dtype.size_in_bytes();
    let chunk_bytes = count * elem_size;
    let total_bytes = chunk_bytes * world;

    if world <= 1 {
        // Single node: copy the rank-0 slice.
        let data = unsafe { client.adapter().stage_for_send(send_ptr, chunk_bytes)? };
        unsafe { client.adapter().receive_to_device(&data, recv_ptr)? };
        return Ok(());
    }

    // Read entire send buffer.
    let data = unsafe { client.adapter().stage_for_send(send_ptr, total_bytes)? };
    let mut buf = data;

    let next = (rank + 1) % world;
    let prev = (rank + world - 1) % world;

    // N-1 rounds of scatter-reduce (same as allreduce phase 1).
    for step in 0..(world - 1) {
        let send_idx = (rank + world - step) % world;
        let recv_idx = (rank + world - step - 1) % world;

        let send_off = send_idx * chunk_bytes;
        let recv_off = recv_idx * chunk_bytes;

        let send_data = buf[send_off..send_off + chunk_bytes].to_vec();

        let (send_result, recv_result) = tokio::join!(
            collective_send(client, next as u32, &send_data, "reduce_scatter"),
            collective_recv(client, prev as u32, "reduce_scatter"),
        );
        send_result?;
        let received = recv_result?;

        // Validate received length before reducing.
        if received.len() != chunk_bytes {
            return Err(NexarError::BufferSizeMismatch {
                expected: chunk_bytes,
                actual: received.len(),
            });
        }
        reduce_slice(
            &mut buf[recv_off..recv_off + chunk_bytes],
            &received,
            count,
            dtype,
            op,
        )?;
    }

    // Our result is the chunk at position (rank + 1) % world after scatter-reduce.
    let result_idx = (rank + 1) % world;
    let result_off = result_idx * chunk_bytes;
    let result = &buf[result_off..result_off + chunk_bytes];

    unsafe { client.adapter().receive_to_device(result, recv_ptr)? };

    Ok(())
}
