use crate::client::NexarClient;
use crate::collective::helpers::{
    CollectiveTag, collective_recv_with_tag, collective_send_with_tag,
};
use crate::error::{NexarError, Result};
use crate::reduce::reduce_slice;
use crate::types::{DataType, ReduceOp};

/// Tagged variant for non-blocking collectives.
pub(crate) async unsafe fn ring_reduce_scatter_with_tag(
    client: &NexarClient,
    send_ptr: u64,
    recv_ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;

    let elem_size = dtype.size_in_bytes();
    let chunk_bytes = count * elem_size;
    let total_bytes = chunk_bytes * world;

    if world <= 1 {
        let data = unsafe { client.adapter().stage_for_send(send_ptr, chunk_bytes)? };
        unsafe { client.adapter().receive_to_device(&data, recv_ptr)? };
        return Ok(());
    }

    let data = unsafe { client.adapter().stage_for_send(send_ptr, total_bytes)? };
    let mut buf = data;

    let next = (rank + 1) % world;
    let prev = (rank + world - 1) % world;

    for step in 0..(world - 1) {
        let send_idx = (rank + world - step) % world;
        let recv_idx = (rank + world - step - 1) % world;

        let send_off = send_idx * chunk_bytes;
        let recv_off = recv_idx * chunk_bytes;

        let send_data = buf[send_off..send_off + chunk_bytes].to_vec();

        let (_, received) = tokio::try_join!(
            collective_send_with_tag(client, next as u32, &send_data, "reduce_scatter", tag),
            collective_recv_with_tag(client, prev as u32, "reduce_scatter", tag),
        )?;

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

    let result_idx = (rank + 1) % world;
    let result_off = result_idx * chunk_bytes;
    let result = &buf[result_off..result_off + chunk_bytes];

    unsafe { client.adapter().receive_to_device(result, recv_ptr)? };

    Ok(())
}
