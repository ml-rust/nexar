use nexar::NexarClient;
use nexar::types::Rank;

use crate::error::{NcclCommError, Result};

/// Describes the topology of nodes in the cluster from this rank's perspective.
pub struct NodeTopology {
    /// Hostname of this node.
    pub hostname: String,
    /// All global ranks on this node, sorted ascending.
    pub local_ranks: Vec<Rank>,
    /// This rank's index within `local_ranks`.
    pub local_rank_idx: usize,
    /// The lead rank for this node (lowest rank on the node).
    pub lead_rank: Rank,
    /// One lead rank per node, ordered by node index.
    pub inter_node_leads: Vec<Rank>,
    /// This node's index among all nodes.
    pub node_idx: usize,
    /// Total number of nodes.
    pub num_nodes: usize,
}

impl NodeTopology {
    /// True if this rank is the lead for its node.
    pub fn is_lead(&self) -> bool {
        self.local_ranks[self.local_rank_idx] == self.lead_rank
    }

    /// Number of local GPUs on this node.
    pub fn local_world_size(&self) -> usize {
        self.local_ranks.len()
    }

    /// True if the entire cluster is a single node.
    pub fn is_single_node(&self) -> bool {
        self.num_nodes == 1
    }
}

/// Max hostname buffer size (256 bytes, null-padded).
const HOSTNAME_BUF_SIZE: usize = 256;

/// Discover cluster topology by exchanging hostnames via nexar allgather.
///
/// All ranks must call this collectively.
///
/// # Safety
/// Uses raw pointer operations for the allgather exchange.
pub async unsafe fn discover_topology(client: &NexarClient) -> Result<NodeTopology> {
    let hostname = gethostname::gethostname().to_string_lossy().into_owned();

    let world = client.world_size() as usize;
    let rank = client.rank();

    // Pad hostname to fixed-size buffer.
    let mut send_buf = [0u8; HOSTNAME_BUF_SIZE];
    let hostname_bytes = hostname.as_bytes();
    let copy_len = hostname_bytes.len().min(HOSTNAME_BUF_SIZE);
    send_buf[..copy_len].copy_from_slice(&hostname_bytes[..copy_len]);

    // Allgather hostnames from all ranks.
    let mut recv_buf = vec![0u8; HOSTNAME_BUF_SIZE * world];

    unsafe {
        client
            .all_gather(
                send_buf.as_ptr() as u64,
                recv_buf.as_mut_ptr() as u64,
                HOSTNAME_BUF_SIZE,
                nexar::types::DataType::U8,
            )
            .await
            .map_err(NcclCommError::Nexar)?;
    }

    // Parse hostnames and group ranks by hostname.
    let mut rank_hostnames: Vec<(Rank, String)> = Vec::with_capacity(world);
    for r in 0..world {
        let off = r * HOSTNAME_BUF_SIZE;
        let buf = &recv_buf[off..off + HOSTNAME_BUF_SIZE];
        let end = buf
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(HOSTNAME_BUF_SIZE);
        let name = String::from_utf8_lossy(&buf[..end]).into_owned();
        rank_hostnames.push((r as Rank, name));
    }

    // Collect unique hostnames in order of first appearance.
    let mut seen_hosts: indexmap::IndexSet<String> = indexmap::IndexSet::new();
    for (_, name) in &rank_hostnames {
        seen_hosts.insert(name.clone());
    }

    // Find which node this rank belongs to.
    let node_idx = seen_hosts
        .get_index_of(&hostname)
        .ok_or_else(|| NcclCommError::Topology {
            reason: "own hostname not found in gathered data".into(),
        })?;

    // Gather local ranks (same hostname) and inter-node leads.
    let mut local_ranks: Vec<Rank> = rank_hostnames
        .iter()
        .filter(|(_, h)| h == &hostname)
        .map(|(r, _)| *r)
        .collect();
    local_ranks.sort();

    let local_rank_idx =
        local_ranks
            .iter()
            .position(|&r| r == rank)
            .ok_or_else(|| NcclCommError::Topology {
                reason: "own rank not in local_ranks".into(),
            })?;

    let lead_rank = local_ranks[0];

    // Build inter-node leads: for each unique hostname, the lowest rank.
    let inter_node_leads: Vec<Rank> = seen_hosts
        .iter()
        .map(|host| {
            rank_hostnames
                .iter()
                .filter(|(_, h)| h == host)
                .map(|(r, _)| *r)
                .min()
                .ok_or_else(|| NcclCommError::Topology {
                    reason: format!("no ranks found for host {host}"),
                })
        })
        .collect::<Result<Vec<Rank>>>()?;

    Ok(NodeTopology {
        hostname,
        local_ranks,
        local_rank_idx,
        lead_rank,
        inter_node_leads,
        node_idx,
        num_nodes: seen_hosts.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_topology_is_lead() {
        let topo = NodeTopology {
            hostname: "node0".into(),
            local_ranks: vec![0, 1, 2, 3],
            local_rank_idx: 0,
            lead_rank: 0,
            inter_node_leads: vec![0, 4],
            node_idx: 0,
            num_nodes: 2,
        };
        assert!(topo.is_lead());
        assert!(!topo.is_single_node());
        assert_eq!(topo.local_world_size(), 4);
    }

    #[test]
    fn test_node_topology_non_lead() {
        let topo = NodeTopology {
            hostname: "node0".into(),
            local_ranks: vec![0, 1, 2, 3],
            local_rank_idx: 2,
            lead_rank: 0,
            inter_node_leads: vec![0, 4],
            node_idx: 0,
            num_nodes: 2,
        };
        assert!(!topo.is_lead());
    }

    #[test]
    fn test_single_node() {
        let topo = NodeTopology {
            hostname: "node0".into(),
            local_ranks: vec![0, 1],
            local_rank_idx: 0,
            lead_rank: 0,
            inter_node_leads: vec![0],
            node_idx: 0,
            num_nodes: 1,
        };
        assert!(topo.is_single_node());
    }
}
