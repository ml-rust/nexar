use nexar::types::{DataType, Rank};

use crate::comm::HierarchicalComm;
use crate::error::{NcclCommError, Result};

/// Hierarchical broadcast from a global root rank.
///
/// Algorithm:
/// - If root is a lead rank: nexar broadcast to other leads → NCCL broadcast per node.
/// - If root is not a lead: NCCL broadcast to local lead → nexar broadcast → NCCL broadcast.
/// - Single node: NCCL broadcast directly.
///
/// # Safety
/// `ptr` must be a valid GPU device pointer for `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn hierarchical_broadcast(
    comm: &HierarchicalComm,
    ptr: u64,
    count: usize,
    dtype: DataType,
    root: Rank,
) -> Result<()> {
    if root >= comm.world_size() {
        return Err(NcclCommError::InvalidRank {
            rank: root,
            world_size: comm.world_size(),
        });
    }

    let topo = comm.topology();

    // Single-node: NCCL broadcast from root's local rank.
    if comm.is_single_node() {
        let root_local = topo
            .local_ranks
            .iter()
            .position(|&r| r == root)
            .ok_or_else(|| NcclCommError::InvalidRank {
                rank: root,
                world_size: comm.world_size(),
            })?;
        unsafe {
            comm.nccl()
                .broadcast_inplace(ptr, count, dtype, root_local)?;
        }
        comm.synchronize()?;
        return Ok(());
    }

    // Determine if root is a lead rank.
    let root_is_lead = topo.inter_node_leads.contains(&root);
    let root_on_our_node = topo.local_ranks.contains(&root);

    if !root_is_lead && root_on_our_node {
        // Root is not a lead but is on our node. NCCL broadcast from root's local
        // rank to all local GPUs so the lead gets the data.
        let root_local = topo
            .local_ranks
            .iter()
            .position(|&r| r == root)
            .expect("root confirmed in local_ranks");
        unsafe {
            comm.nccl()
                .broadcast_inplace(ptr, count, dtype, root_local)?;
        }
        comm.synchronize()?;
    }

    // Step 2: Inter-node broadcast from the root's node lead to all other leads.
    if comm.is_lead() {
        if let Some(inter) = comm.inter_node() {
            // Find which inter-node rank corresponds to the root's node.
            let root_inter_rank = if root_on_our_node {
                // We are on the root's node — our lead is the inter-node root.
                inter.rank()
            } else {
                // Find which node the root is on by checking rank ranges.
                find_node_for_rank(root, topo, comm.world_size())
            };

            unsafe {
                inter
                    .broadcast(ptr, count, dtype, root_inter_rank)
                    .await
                    .map_err(NcclCommError::Nexar)?;
            }
        }
    }

    // Step 3: NCCL broadcast from lead (local_rank 0) to all local GPUs.
    // On the root's node when root is not lead, step 1 already gave everyone the data
    // and lead sent it inter-node. Now lead broadcasts the (possibly updated) data back.
    // On other nodes, lead received from inter-node and now distributes locally.
    unsafe {
        comm.nccl().broadcast_inplace(ptr, count, dtype, 0)?;
    }
    comm.synchronize()?;

    Ok(())
}

/// Find which inter-node rank (node index) a global rank belongs to.
///
/// Uses the inter_node_leads array to determine rank ranges per node.
fn find_node_for_rank(rank: Rank, topo: &crate::topology::NodeTopology, world_size: u32) -> Rank {
    for (idx, &lead) in topo.inter_node_leads.iter().enumerate() {
        let next_start = if idx + 1 < topo.inter_node_leads.len() {
            topo.inter_node_leads[idx + 1]
        } else {
            world_size
        };
        if rank >= lead && rank < next_start {
            return idx as Rank;
        }
    }
    0
}
