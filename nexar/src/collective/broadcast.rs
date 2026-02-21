use crate::client::NexarClient;
use crate::collective::helpers::{CollectiveTag, collective_recv, collective_send};
use crate::error::Result;
use crate::types::{DataType, Rank};
use futures::future::try_join_all;

/// Threshold: use flat broadcast for small worlds, tree broadcast for larger.
const TREE_BROADCAST_THRESHOLD: u32 = 4;

/// Tree broadcast: root sends data to all other ranks via binary tree.
///
/// For sparse topologies, uses a BFS spanning tree over the neighbor graph
/// to ensure every parent→child link is a direct connection.
///
/// Falls back to flat broadcast (root sends to all directly) for small
/// world sizes (< `TREE_BROADCAST_THRESHOLD`) in full mesh mode.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub(crate) async unsafe fn tree_broadcast(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    root: Rank,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size();

    if world <= 1 {
        return Ok(());
    }

    // For sparse topologies, use BFS spanning tree over neighbor graph.
    if client.is_sparse() {
        return unsafe { sparse_tree_broadcast(client, ptr, count, dtype, root, tag).await };
    }

    if world < TREE_BROADCAST_THRESHOLD {
        return unsafe { flat_broadcast(client, ptr, count, dtype, root, tag).await };
    }

    let rank = client.rank();
    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    // Remap ranks so root becomes logical rank 0.
    let logical = |r: Rank| -> Rank { (r + world - root) % world };
    let physical = |l: Rank| -> Rank { (l + root) % world };
    let my_logical = logical(rank);

    let data = if my_logical == 0 {
        unsafe { client.adapter().stage_for_send(ptr, total_bytes)? }
    } else {
        let parent_logical = (my_logical - 1) / 2;
        let parent_physical = physical(parent_logical);
        let received = collective_recv(client, parent_physical, "broadcast", tag).await?;
        if received.len() != total_bytes {
            return Err(crate::error::NexarError::BufferSizeMismatch {
                expected: total_bytes,
                actual: received.len(),
            });
        }
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
            futs.push(collective_send(
                client,
                child_phys,
                data_ref,
                "broadcast",
                tag,
            ));
        }
    }

    if !futs.is_empty() {
        try_join_all(futs).await?;
    }

    Ok(())
}

/// BFS spanning tree broadcast for sparse topologies.
///
/// Builds a spanning tree over the neighbor graph rooted at `root`, then
/// each node receives from its parent and sends to its children. All edges
/// are direct connections (no relay needed).
async unsafe fn sparse_tree_broadcast(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    root: Rank,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size();
    let rank = client.rank();
    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    let topology = &client.config().topology;
    let tree = crate::cluster::sparse::build_spanning_tree(topology, root, world);

    let data =
        if rank == root {
            unsafe { client.adapter().stage_for_send(ptr, total_bytes)? }
        } else {
            let parent = *tree.parent.get(&rank).ok_or_else(|| {
                crate::error::NexarError::CollectiveFailed {
                    operation: "broadcast",
                    rank,
                    reason: "rank not in spanning tree".to_string(),
                }
            })?;
            let received = collective_recv(client, parent, "broadcast", tag).await?;
            if received.len() != total_bytes {
                return Err(crate::error::NexarError::BufferSizeMismatch {
                    expected: total_bytes,
                    actual: received.len(),
                });
            }
            unsafe { client.adapter().receive_to_device(&received, ptr)? };
            received.to_vec()
        };

    // Send to children concurrently.
    let children = tree.children.get(&rank).cloned().unwrap_or_default();
    if !children.is_empty() {
        let futs: Vec<_> = children
            .iter()
            .map(|&child| collective_send(client, child, &data, "broadcast", tag))
            .collect();
        try_join_all(futs).await?;
    }

    Ok(())
}

/// Flat broadcast: root sends to all other ranks concurrently.
async unsafe fn flat_broadcast(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    root: Rank,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size();
    let rank = client.rank();
    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    if rank == root {
        let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };

        let futs: Vec<_> = (0..world)
            .filter(|&r| r != root)
            .map(|r| collective_send(client, r, &data, "broadcast", tag))
            .collect();

        try_join_all(futs).await?;
    } else {
        let received = collective_recv(client, root, "broadcast", tag).await?;
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
