use crate::client::NexarClient;
use crate::collective::helpers::{CollectiveTag, collective_recv, collective_send, step_tag};
use crate::error::{NexarError, Result};
use crate::types::DataType;

/// Ring allgather: each rank contributes `count` elements, result is all
/// contributions concatenated in rank order.
///
/// Uses N-1 ring rounds where each rank forwards the latest received chunk
/// to its successor.
///
/// # Safety
/// - `send_ptr` must point to at least `count * dtype.size_in_bytes()` bytes.
/// - `recv_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
pub(crate) async unsafe fn ring_allgather(
    client: &NexarClient,
    send_ptr: u64,
    recv_ptr: u64,
    count: usize,
    dtype: DataType,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;

    let elem_size = dtype.size_in_bytes();
    let chunk_bytes = count * elem_size;
    let total_bytes = chunk_bytes * world;

    if world <= 1 {
        // Single node: copy send to recv.
        let data = unsafe { client.adapter().stage_for_send(send_ptr, chunk_bytes)? };
        unsafe { client.adapter().receive_to_device(&data, recv_ptr)? };
        return Ok(());
    }

    // Build the output buffer: place our own data at position `rank`.
    let mut buf = vec![0u8; total_bytes];
    let own_data = unsafe { client.adapter().stage_for_send(send_ptr, chunk_bytes)? };
    buf[rank * chunk_bytes..(rank + 1) * chunk_bytes].copy_from_slice(&own_data);

    let next = (rank + 1) % world;
    let prev = (rank + world - 1) % world;

    // N-1 rounds: each round, send our latest received chunk to next,
    // receive a chunk from prev and place it.
    for step in 0..(world - 1) {
        let send_idx = (rank + world - step) % world;
        let recv_idx = (rank + world - step - 1) % world;

        let send_data = buf[send_idx * chunk_bytes..(send_idx + 1) * chunk_bytes].to_vec();

        let round_tag = step_tag(tag, step);
        let (_, received) = tokio::try_join!(
            collective_send(client, next as u32, &send_data, "allgather", round_tag),
            collective_recv(client, prev as u32, "allgather", round_tag),
        )?;

        if received.len() != chunk_bytes {
            return Err(NexarError::BufferSizeMismatch {
                expected: chunk_bytes,
                actual: received.len(),
            });
        }
        buf[recv_idx * chunk_bytes..(recv_idx + 1) * chunk_bytes].copy_from_slice(&received);
    }

    // Write result back.
    unsafe { client.adapter().receive_to_device(&buf, recv_ptr)? };

    Ok(())
}
