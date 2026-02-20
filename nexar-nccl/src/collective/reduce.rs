use nexar::types::{DataType, Rank, ReduceOp};

use crate::comm::HierarchicalComm;
use crate::error::{NcclCommError, Result};

/// Hierarchical reduce to a single global root rank.
///
/// Algorithm:
/// - NCCL reduce to local lead (local_rank 0).
/// - nexar reduce across leads to the root's node lead.
/// - If root is not the lead, NCCL broadcast from lead to all local GPUs.
///
/// # Safety
/// `ptr` must be a valid GPU device pointer for `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn hierarchical_reduce(
    comm: &HierarchicalComm,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    root: Rank,
) -> Result<()> {
    if root >= comm.world_size() {
        return Err(NcclCommError::InvalidRank {
            rank: root,
            world_size: comm.world_size(),
        });
    }

    let topo = comm.topology();

    // Single-node: NCCL reduce directly.
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
                .reduce_inplace(ptr, count, dtype, op, root_local)?;
        }
        comm.synchronize()?;
        return Ok(());
    }

    // Step 1: NCCL reduce to local lead (local_rank 0) on each node.
    unsafe {
        comm.nccl().reduce_inplace(ptr, count, dtype, op, 0)?;
    }
    comm.synchronize()?;

    // Step 2: Leads run nexar reduce to the root's node lead.
    if comm.is_lead()
        && let Some(inter) = comm.inter_node()
    {
        let root_on_our_node = topo.local_ranks.contains(&root);
        let inter_root = if root_on_our_node {
            inter.rank()
        } else {
            find_node_for_rank(root, topo, comm.world_size())
        };

        unsafe {
            inter
                .reduce(ptr, count, dtype, op, inter_root)
                .await
                .map_err(NcclCommError::Nexar)?;
        }
    }

    // Step 3: If root is not the lead on its node, broadcast from lead to root.
    if topo.local_ranks.contains(&root) && root != topo.lead_rank {
        unsafe {
            comm.nccl().broadcast_inplace(ptr, count, dtype, 0)?;
        }
        comm.synchronize()?;
    }

    Ok(())
}

/// Find which inter-node rank (node index) a global rank belongs to.
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
