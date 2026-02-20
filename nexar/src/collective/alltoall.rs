use crate::client::NexarClient;
use crate::collective::helpers::{
    CollectiveTag, collective_recv_with_tag, collective_send_with_tag,
};
use crate::error::{NexarError, Result};
use crate::types::DataType;

/// Tagged variant for non-blocking collectives.
pub(crate) async unsafe fn alltoall_with_tag(
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

    let send_buf = unsafe { client.adapter().stage_for_send(send_ptr, total_bytes)? };
    let mut recv_buf = vec![0u8; total_bytes];

    // Round 0: local copy.
    let self_off = rank * chunk_bytes;
    recv_buf[self_off..self_off + chunk_bytes]
        .copy_from_slice(&send_buf[self_off..self_off + chunk_bytes]);

    // Rounds 1..world-1: pairwise exchange.
    for step in 1..world {
        let send_to = (rank + step) % world;
        let recv_from = (rank + world - step) % world;

        let send_off = send_to * chunk_bytes;
        let send_data = &send_buf[send_off..send_off + chunk_bytes];

        let (_, received) = tokio::try_join!(
            collective_send_with_tag(client, send_to as u32, send_data, "alltoall", tag),
            collective_recv_with_tag(client, recv_from as u32, "alltoall", tag),
        )?;

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
