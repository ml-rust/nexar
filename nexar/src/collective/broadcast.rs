use crate::client::NexarClient;
use crate::collective::helpers::{collective_recv, collective_send};
use crate::error::Result;
use crate::types::{DataType, Rank};
use futures::future::try_join_all;

/// Threshold: use flat broadcast for small worlds, tree broadcast for larger.
const TREE_BROADCAST_THRESHOLD: u32 = 4;

/// Tree broadcast from `root` to all other ranks (O(log N) rounds).
///
/// Uses a binary tree rooted at the logical root. In each round, ranks that
/// already have data forward it to their children. After `ceil(log2(N))` rounds,
/// all ranks have the data.
///
/// Falls back to `flat_broadcast` for world_size < `TREE_BROADCAST_THRESHOLD`.
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

    if world <= 1 {
        return Ok(());
    }

    if world < TREE_BROADCAST_THRESHOLD {
        return unsafe { flat_broadcast(client, ptr, count, dtype, root).await };
    }

    let rank = client.rank();
    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    // Remap ranks so root becomes logical rank 0.
    let logical = |r: Rank| -> Rank { (r + world - root) % world };
    let physical = |l: Rank| -> Rank { (l + root) % world };
    let my_logical = logical(rank);

    // Binary tree: parent of logical rank L is L/2 (for L > 0).
    // Children of L are 2L+1 and 2L+2 (if < world).
    // In each round r (0-indexed), ranks with logical index < 2^(r+1) send to
    // their children. But it's simpler to just figure out when *this* rank
    // receives and when it sends.

    // This rank receives from its parent (unless it's the root).
    // Then it sends to its children (if any).

    let data = if my_logical == 0 {
        // Root: stage data once.
        unsafe { client.adapter().stage_for_send(ptr, total_bytes)? }
    } else {
        // Non-root: receive from parent.
        let parent_logical = (my_logical - 1) / 2;
        let parent_physical = physical(parent_logical);
        let received = collective_recv(client, parent_physical, "broadcast").await?;
        if received.len() != total_bytes {
            return Err(crate::error::NexarError::BufferSizeMismatch {
                expected: total_bytes,
                actual: received.len(),
            });
        }
        // Write to device memory.
        unsafe { client.adapter().receive_to_device(&received, ptr)? };
        received.to_vec()
    };

    // Send to children concurrently.
    let child_left = 2 * my_logical + 1;
    let child_right = 2 * my_logical + 2;

    let mut futs = Vec::new();
    for child_logical in [child_left, child_right] {
        if child_logical < world {
            let child_phys = physical(child_logical);
            let data_ref = &data;
            futs.push(collective_send(client, child_phys, data_ref, "broadcast"));
        }
    }

    if !futs.is_empty() {
        try_join_all(futs).await?;
    }

    Ok(())
}

/// Flat broadcast: root sends to all other ranks concurrently.
///
/// Simpler than tree broadcast with lower constant overhead, suitable for
/// small world sizes. All sends happen concurrently via `try_join_all`.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
async unsafe fn flat_broadcast(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    root: Rank,
) -> Result<()> {
    let world = client.world_size();
    let rank = client.rank();
    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    if rank == root {
        let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };

        // Send to all other ranks concurrently.
        let futs: Vec<_> = (0..world)
            .filter(|&r| r != root)
            .map(|r| collective_send(client, r, &data, "broadcast"))
            .collect();

        try_join_all(futs).await?;
    } else {
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
