use crate::client::NexarClient;
use crate::collective::helpers::{collective_recv, collective_send};
use crate::error::{NexarError, Result};
use crate::types::DataType;

/// All-to-all: each rank sends a distinct chunk to every other rank.
///
/// Pairwise exchange algorithm: in round `r`, rank communicates with
/// `(rank + r) % world`. This avoids contention since no two pairs
/// overlap in any round.
///
/// Input layout at `send_ptr`: `world_size` consecutive chunks, each of
/// `count` elements. Chunk `i` is destined for rank `i`.
///
/// Output layout at `recv_ptr`: `world_size` consecutive chunks, each of
/// `count` elements. Chunk `i` was received from rank `i`.
///
/// # Safety
/// - `send_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
/// - `recv_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
pub async unsafe fn alltoall(
    client: &NexarClient,
    send_ptr: u64,
    recv_ptr: u64,
    count: usize,
    dtype: DataType,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;
    let elem_size = dtype.size_in_bytes();
    let chunk_bytes = count * elem_size;
    let total_bytes = chunk_bytes * world;

    // Stage entire send buffer.
    let send_buf = unsafe { client.adapter().stage_for_send(send_ptr, total_bytes)? };
    let mut recv_buf = vec![0u8; total_bytes];

    // Round 0: local copy (chunk destined for self).
    let self_off = rank * chunk_bytes;
    recv_buf[self_off..self_off + chunk_bytes]
        .copy_from_slice(&send_buf[self_off..self_off + chunk_bytes]);

    // Rounds 1..world-1: pairwise exchange.
    for step in 1..world {
        let send_to = (rank + step) % world;
        let recv_from = (rank + world - step) % world;

        let send_off = send_to * chunk_bytes;
        let send_data = &send_buf[send_off..send_off + chunk_bytes];

        let (send_result, recv_result) = tokio::join!(
            collective_send(client, send_to as u32, send_data, "alltoall"),
            collective_recv(client, recv_from as u32, "alltoall"),
        );
        send_result?;
        let received = recv_result?;

        if received.len() != chunk_bytes {
            return Err(NexarError::BufferSizeMismatch {
                expected: chunk_bytes,
                actual: received.len(),
            });
        }

        let recv_off = recv_from * chunk_bytes;
        recv_buf[recv_off..recv_off + chunk_bytes].copy_from_slice(&received);
    }

    unsafe { client.adapter().receive_to_device(&recv_buf, recv_ptr)? };

    Ok(())
}
