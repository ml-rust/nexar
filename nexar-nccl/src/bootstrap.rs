use std::sync::Arc;

use cudarc::driver::CudaStream;
use cudarc::nccl::safe::Id;

use crate::comm::HierarchicalComm;
use crate::error::{NcclCommError, Result};
use crate::group::{NcclGroup, id_from_bytes, id_to_bytes};
use crate::topology::{NodeTopology, discover_topology};
use nexar::NexarClient;

/// Form a hierarchical communicator from a nexar client and CUDA stream.
///
/// This is the main entry point. All ranks must call this collectively.
///
/// Steps:
/// 1. Discover topology (exchange hostnames via nexar allgather).
/// 2. Lead rank generates NCCL unique ID, broadcasts to local ranks via nexar.
/// 3. All local ranks call `ncclCommInitRank` to form the intra-node NCCL group.
/// 4. Lead ranks split off an inter-node nexar sub-communicator.
///
/// # Safety
/// Uses raw pointer operations for the topology discovery and NCCL ID exchange.
pub async unsafe fn form_hierarchical_comm(
    nexar: Arc<NexarClient>,
    stream: Arc<CudaStream>,
) -> Result<HierarchicalComm> {
    // Step 1: Discover topology.
    let topo = unsafe { discover_topology(&nexar).await? };

    // Step 2: Exchange NCCL unique ID among local ranks.
    // The lead rank generates the ID and broadcasts it.
    let nccl_id = exchange_nccl_id(&nexar, &topo).await?;

    // Step 3: Initialize NCCL communicator.
    let nccl = NcclGroup::init(
        stream,
        topo.local_rank_idx,
        topo.local_world_size(),
        nccl_id,
    )?;

    // Step 4: Split nexar communicator for inter-node leads.
    let inter_node = if topo.num_nodes > 1 {
        // Color: leads get color 0, non-leads get color 1.
        // Key: node index for leads (to ensure consistent ordering).
        let color = if topo.is_lead() { 0u32 } else { 1u32 };
        let key = topo.node_idx as u32;
        let split = nexar
            .split(color, key)
            .await
            .map_err(NcclCommError::Nexar)?;
        if topo.is_lead() { Some(split) } else { None }
    } else {
        None
    };

    Ok(HierarchicalComm::new(nexar, inter_node, nccl, topo))
}

/// Exchange the NCCL unique ID among ranks on the same node.
///
/// The lead rank generates the ID and broadcasts it via nexar to all local ranks.
async fn exchange_nccl_id(client: &NexarClient, topo: &NodeTopology) -> Result<Id> {
    // Use a tagged allgather-style exchange: lead rank generates, others receive.
    // We broadcast via nexar send/recv among local ranks.
    const NCCL_ID_SIZE: usize = 128;

    if topo.local_world_size() == 1 {
        // Single GPU on this node — just generate locally.
        return Id::new().map_err(NcclCommError::from);
    }

    let rank = client.rank();

    if topo.is_lead() {
        // Lead generates the ID and sends to all other local ranks.
        let id = Id::new().map_err(NcclCommError::from)?;
        let bytes = id_to_bytes(&id);

        for &local_rank in &topo.local_ranks {
            if local_rank != rank {
                let tag = 0xACC1_0000u32 | (topo.node_idx as u32);
                unsafe {
                    client
                        .send(bytes.as_ptr() as u64, NCCL_ID_SIZE, local_rank, tag)
                        .await
                        .map_err(NcclCommError::Nexar)?;
                }
            }
        }
        Ok(id)
    } else {
        // Non-lead receives the ID from the lead.
        let mut buf = vec![0u8; NCCL_ID_SIZE];
        let tag = 0xACC1_0000u32 | (topo.node_idx as u32);
        unsafe {
            client
                .recv(buf.as_mut_ptr() as u64, NCCL_ID_SIZE, topo.lead_rank, tag)
                .await
                .map_err(NcclCommError::Nexar)?;
        }
        Ok(id_from_bytes(&buf))
    }
}
