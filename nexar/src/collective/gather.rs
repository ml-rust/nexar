use crate::client::NexarClient;
use crate::collective::helpers::{CollectiveTag, collective_recv, collective_send};
use crate::error::Result;
use crate::types::{DataType, Rank};
use futures::future::try_join_all;

/// Gather: root collects `count` elements from each rank.
///
/// Non-root ranks send their data to root; root receives from all peers
/// concurrently and places each contribution at the sender's rank offset.
///
/// # Safety
/// - `send_ptr`: at least `count * dtype.size_in_bytes()` bytes on all ranks.
/// - `recv_ptr`: at least `count * world_size * dtype.size_in_bytes()` bytes on root.
pub(crate) async unsafe fn gather(
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
        let own_data = unsafe { client.adapter().stage_for_send(send_ptr, chunk_bytes)? };
        let own_offset = root as u64 * chunk_bytes as u64;
        unsafe {
            client
                .adapter()
                .receive_to_device(&own_data, recv_ptr + own_offset)?
        };

        let futs: Vec<_> = (0..world)
            .filter(|&r| r != root)
            .map(|r| async move {
                let received = collective_recv(client, r, "gather", tag).await?;
                if received.len() != chunk_bytes {
                    return Err(crate::error::NexarError::BufferSizeMismatch {
                        expected: chunk_bytes,
                        actual: received.len(),
                    });
                }
                let offset = r as u64 * chunk_bytes as u64;
                unsafe {
                    client
                        .adapter()
                        .receive_to_device(&received, recv_ptr + offset)?
                };
                Ok(())
            })
            .collect();

        try_join_all(futs).await?;
    } else {
        let data = unsafe { client.adapter().stage_for_send(send_ptr, chunk_bytes)? };
        collective_send(client, root, &data, "gather", tag).await?;
    }

    Ok(())
}
