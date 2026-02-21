use crate::types::Rank;
use std::collections::{HashMap, HashSet, VecDeque};

/// Topology strategy for the peer mesh.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum TopologyStrategy {
    /// Every node connects to every other node. O(N^2) connections.
    #[default]
    FullMesh,
    /// Each node connects to K nearest neighbors on a ring. O(K*N) connections.
    KRegular { degree: usize },
    /// Each node connects to peers differing in exactly one bit. Requires power-of-2 world_size.
    Hypercube,
}

/// Routing table for a single rank: direct neighbors and next-hop routing for non-neighbors.
#[derive(Debug, Clone)]
pub struct RoutingTable {
    /// Directly connected peers.
    pub neighbors: HashSet<Rank>,
    /// For non-neighbor destinations: dest -> next_hop.
    pub next_hop: HashMap<Rank, Rank>,
}

impl RoutingTable {
    /// Get the next hop to reach `dest`. Returns `dest` itself if it's a direct neighbor.
    pub fn route(&self, dest: Rank) -> Option<Rank> {
        if self.neighbors.contains(&dest) {
            Some(dest)
        } else {
            self.next_hop.get(&dest).copied()
        }
    }

    /// Returns true if `dest` is a direct neighbor.
    pub fn is_neighbor(&self, dest: Rank) -> bool {
        self.neighbors.contains(&dest)
    }
}

/// BFS spanning tree over the neighbor graph.
pub struct SpanningTree {
    /// node -> parent rank (root has no entry).
    pub parent: HashMap<Rank, Rank>,
    /// node -> list of children.
    pub children: HashMap<Rank, Vec<Rank>>,
}

/// Compute which peers a rank should directly connect to.
pub fn build_neighbors(strategy: &TopologyStrategy, rank: Rank, world_size: u32) -> HashSet<Rank> {
    match strategy {
        TopologyStrategy::FullMesh => (0..world_size).filter(|&r| r != rank).collect(),
        TopologyStrategy::KRegular { degree } => {
            let half = (*degree / 2) as u32;
            let mut neighbors = HashSet::new();
            for d in 1..=half {
                neighbors.insert((rank + d) % world_size);
                neighbors.insert((rank + world_size - d) % world_size);
            }
            neighbors
        }
        TopologyStrategy::Hypercube => {
            assert!(
                world_size.is_power_of_two(),
                "Hypercube requires power-of-2 world_size, got {world_size}"
            );
            let bits = world_size.trailing_zeros();
            (0..bits).map(|b| rank ^ (1 << b)).collect()
        }
    }
}

/// Shortest distance on a ring of `world_size` nodes.
fn ring_distance(from: Rank, to: Rank, world_size: u32) -> u32 {
    let cw = (to + world_size - from) % world_size;
    let ccw = (from + world_size - to) % world_size;
    cw.min(ccw)
}

/// Build a complete routing table for a rank.
pub fn build_routing_table(
    strategy: &TopologyStrategy,
    rank: Rank,
    world_size: u32,
) -> RoutingTable {
    let neighbors = build_neighbors(strategy, rank, world_size);

    if matches!(strategy, TopologyStrategy::FullMesh) {
        return RoutingTable {
            neighbors,
            next_hop: HashMap::new(),
        };
    }

    let mut next_hop = HashMap::new();

    for dest in 0..world_size {
        if dest == rank || neighbors.contains(&dest) {
            continue;
        }

        let hop = match strategy {
            TopologyStrategy::FullMesh => unreachable!(),
            TopologyStrategy::KRegular { .. } => {
                // Greedy ring routing: pick neighbor closest to dest on the ring.
                *neighbors
                    .iter()
                    .min_by_key(|&&n| ring_distance(n, dest, world_size))
                    .expect("KRegular always has neighbors")
            }
            TopologyStrategy::Hypercube => {
                // Bit-fixing: flip the highest differing bit toward target.
                let diff = rank ^ dest;
                let highest_bit = 31 - diff.leading_zeros();
                rank ^ (1 << highest_bit)
            }
        };

        next_hop.insert(dest, hop);
    }

    RoutingTable {
        neighbors,
        next_hop,
    }
}

/// For ring collectives, return a Hamiltonian cycle through the neighbor graph.
///
/// For KRegular with degree >= 2, the natural ring 0→1→2→...→N-1→0 works
/// because ±1 are always direct neighbors.
pub fn optimal_ring_order(world_size: u32) -> Vec<Rank> {
    (0..world_size).collect()
}

/// Build a BFS spanning tree from `root` over the neighbor graph.
///
/// Every edge in the tree is a direct connection between neighbors.
/// Used by broadcast/reduce to avoid relay overhead.
pub fn build_spanning_tree(
    strategy: &TopologyStrategy,
    root: Rank,
    world_size: u32,
) -> SpanningTree {
    let mut parent = HashMap::new();
    let mut children: HashMap<Rank, Vec<Rank>> = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(root);
    queue.push_back(root);
    children.insert(root, Vec::new());

    while let Some(node) = queue.pop_front() {
        let node_neighbors = build_neighbors(strategy, node, world_size);
        let mut node_children: Vec<Rank> = node_neighbors
            .into_iter()
            .filter(|n| visited.insert(*n))
            .collect();
        node_children.sort();
        for &child in &node_children {
            parent.insert(child, node);
            children.insert(child, Vec::new());
            queue.push_back(child);
        }
        // Safe: node is in children because it either came from root insertion or was inserted in previous iteration
        children
            .get_mut(&node)
            .expect("BFS tree invariant: all visited nodes have children entry")
            .extend(node_children);
    }

    SpanningTree { parent, children }
}

/// Parse a topology strategy from a string.
///
/// Formats: "full_mesh", "k_regular:8", "hypercube"
pub fn parse_topology(s: &str) -> Option<TopologyStrategy> {
    let s = s.trim().to_lowercase();
    if s == "full_mesh" {
        Some(TopologyStrategy::FullMesh)
    } else if s == "hypercube" {
        Some(TopologyStrategy::Hypercube)
    } else if let Some(rest) = s.strip_prefix("k_regular:") {
        rest.parse::<usize>().ok().map(|degree| {
            assert!(
                degree >= 2 && degree % 2 == 0,
                "KRegular degree must be even and >= 2"
            );
            TopologyStrategy::KRegular { degree }
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_mesh_neighbors() {
        let n = build_neighbors(&TopologyStrategy::FullMesh, 0, 4);
        assert_eq!(n, HashSet::from([1, 2, 3]));
    }

    #[test]
    fn test_kregular_neighbors() {
        // degree=4 means ±1, ±2
        let n = build_neighbors(&TopologyStrategy::KRegular { degree: 4 }, 0, 8);
        assert_eq!(n, HashSet::from([1, 2, 6, 7]));
    }

    #[test]
    fn test_kregular_neighbors_wrap() {
        let n = build_neighbors(&TopologyStrategy::KRegular { degree: 4 }, 7, 8);
        assert_eq!(n, HashSet::from([5, 6, 0, 1]));
    }

    #[test]
    fn test_hypercube_neighbors() {
        // world_size=8: 3 bits, rank 0 connects to 1,2,4
        let n = build_neighbors(&TopologyStrategy::Hypercube, 0, 8);
        assert_eq!(n, HashSet::from([1, 2, 4]));

        let n = build_neighbors(&TopologyStrategy::Hypercube, 5, 8);
        // 5 = 101, neighbors: 100=4, 111=7, 001=1
        assert_eq!(n, HashSet::from([4, 7, 1]));
    }

    #[test]
    fn test_kregular_routing() {
        let rt = build_routing_table(&TopologyStrategy::KRegular { degree: 4 }, 0, 16);
        // Neighbors of 0: {14, 15, 1, 2}
        assert_eq!(rt.neighbors, HashSet::from([14, 15, 1, 2]));
        // Route to 5: closest neighbor is 2 (distance 3 to 5)
        let hop = rt.route(5).unwrap();
        assert!(rt.neighbors.contains(&hop));
    }

    #[test]
    fn test_hypercube_routing() {
        let rt = build_routing_table(&TopologyStrategy::Hypercube, 0, 8);
        // Route to 7 (111): highest differing bit is 2, hop = 0 ^ 4 = 4
        assert_eq!(rt.route(7), Some(4));
        // Route to 3 (011): highest differing bit is 1, hop = 0 ^ 2 = 2
        assert_eq!(rt.route(3), Some(2));
    }

    #[test]
    fn test_full_mesh_no_next_hop() {
        let rt = build_routing_table(&TopologyStrategy::FullMesh, 0, 4);
        assert!(rt.next_hop.is_empty());
        assert_eq!(rt.route(3), Some(3));
    }

    #[test]
    fn test_spanning_tree() {
        let tree = build_spanning_tree(&TopologyStrategy::KRegular { degree: 4 }, 0, 8);
        // All nodes should be in the tree
        assert_eq!(tree.children.len(), 8);
        // Root has no parent
        assert!(!tree.parent.contains_key(&0));
        // Every non-root has a parent
        for r in 1..8 {
            assert!(tree.parent.contains_key(&r));
        }
    }

    #[test]
    fn test_parse_topology() {
        assert_eq!(
            parse_topology("full_mesh"),
            Some(TopologyStrategy::FullMesh)
        );
        assert_eq!(
            parse_topology("hypercube"),
            Some(TopologyStrategy::Hypercube)
        );
        assert_eq!(
            parse_topology("k_regular:8"),
            Some(TopologyStrategy::KRegular { degree: 8 })
        );
        assert_eq!(parse_topology("invalid"), None);
    }

    #[test]
    fn test_ring_distance() {
        assert_eq!(ring_distance(0, 3, 8), 3);
        assert_eq!(ring_distance(0, 5, 8), 3); // 8-5=3 < 5
        assert_eq!(ring_distance(7, 0, 8), 1);
    }
}
