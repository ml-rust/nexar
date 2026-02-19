use crate::client::NexarClient;
use crate::collective::{collective_recv, collective_send};
use crate::error::Result;
use crate::types::{DataType, Rank};

/// Flat broadcast from `root` to all other ranks.
///
/// Root sends the data to every other rank sequentially. Each non-root rank
/// receives from the root. QUIC multiplexes the sends on separate streams
/// so they proceed concurrently at the transport level.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn tree_broadcast(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    root: Rank,
) -> Result<()> {
    let world = client.world_size();
    let rank = client.rank();

    if world <= 1 {
        return Ok(());
    }

    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    if rank == root {
        // Root: read data and send to every other rank.
        let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };

        for r in 0..world {
            if r != root {
                collective_send(client, r, &data, "broadcast").await?;
            }
        }
    } else {
        // Non-root: receive data from root.
        let received = collective_recv(client, root, "broadcast").await?;
        if received.len() != total_bytes {
            return Err(crate::error::NexarError::BufferSizeMismatch {
                expected: total_bytes,
                actual: received.len(),
            });
        }
        unsafe { client.adapter().receive_to_device(&received, ptr)? };
    }

    Ok(())
}
