use crate::client::NexarClient;
use crate::collective::helpers::{
    CollectiveTag, collective_recv_with_tag, collective_send_with_tag,
};
use crate::error::Result;
use crate::types::{DataType, Rank};
use futures::future::try_join_all;

/// Tagged variant for non-blocking collectives.
pub(crate) async unsafe fn scatter_with_tag(
    client: &NexarClient,
    send_ptr: u64,
    recv_ptr: u64,
    count: usize,
    dtype: DataType,
    root: Rank,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size();
    let rank = client.rank();
    let elem_size = dtype.size_in_bytes();
    let chunk_bytes = count * elem_size;

    if world <= 1 {
        let data = unsafe { client.adapter().stage_for_send(send_ptr, chunk_bytes)? };
        unsafe { client.adapter().receive_to_device(&data, recv_ptr)? };
        return Ok(());
    }

    if rank == root {
        let total_bytes = chunk_bytes * world as usize;
        let all_data = unsafe { client.adapter().stage_for_send(send_ptr, total_bytes)? };

        let own_start = root as usize * chunk_bytes;
        let own_chunk = &all_data[own_start..own_start + chunk_bytes];
        unsafe { client.adapter().receive_to_device(own_chunk, recv_ptr)? };

        let futs: Vec<_> = (0..world)
            .filter(|&r| r != root)
            .map(|r| {
                let start = r as usize * chunk_bytes;
                let chunk = &all_data[start..start + chunk_bytes];
                collective_send_with_tag(client, r, chunk, "scatter", tag)
            })
            .collect();

        try_join_all(futs).await?;
    } else {
        let received = collective_recv_with_tag(client, root, "scatter", tag).await?;
        if received.len() != chunk_bytes {
            return Err(crate::error::NexarError::BufferSizeMismatch {
                expected: chunk_bytes,
                actual: received.len(),
            });
        }
        unsafe { client.adapter().receive_to_device(&received, recv_ptr)? };
    }

    Ok(())
}
