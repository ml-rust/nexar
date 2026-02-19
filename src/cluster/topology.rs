use crate::types::Rank;
use std::collections::HashMap;
use std::time::Instant;

/// Tracks the set of peers in the cluster and their health status.
#[derive(Debug)]
pub struct ClusterMap {
    /// Peer addresses indexed by rank.
    peers: HashMap<Rank, PeerInfo>,
    /// Monotonically increasing epoch for topology changes.
    epoch: u64,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub addr: String,
    pub last_heartbeat: Instant,
    pub alive: bool,
}

impl ClusterMap {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            epoch: 0,
        }
    }

    /// Register a peer. Bumps the epoch.
    pub fn add_peer(&mut self, rank: Rank, addr: String) {
        self.peers.insert(
            rank,
            PeerInfo {
                addr,
                last_heartbeat: Instant::now(),
                alive: true,
            },
        );
        self.epoch += 1;
    }

    /// Mark a peer as dead. Bumps the epoch.
    pub fn remove_peer(&mut self, rank: Rank) -> bool {
        if let Some(info) = self.peers.get_mut(&rank) {
            if info.alive {
                info.alive = false;
                self.epoch += 1;
                return true;
            }
        }
        false
    }

    /// Update heartbeat timestamp for a peer.
    pub fn heartbeat(&mut self, rank: Rank) {
        if let Some(info) = self.peers.get_mut(&rank) {
            info.last_heartbeat = Instant::now();
        }
    }

    /// Get the list of alive peers as `(rank, addr)` pairs.
    pub fn alive_peers(&self) -> Vec<(Rank, String)> {
        self.peers
            .iter()
            .filter(|(_, info)| info.alive)
            .map(|(&rank, info)| (rank, info.addr.clone()))
            .collect()
    }

    /// Get a specific peer's info.
    pub fn get(&self, rank: Rank) -> Option<&PeerInfo> {
        self.peers.get(&rank)
    }

    /// Check for peers whose heartbeats are older than `timeout` and mark them dead.
    /// Returns the list of newly-dead ranks.
    pub fn check_timeouts(&mut self, timeout: std::time::Duration) -> Vec<Rank> {
        let now = Instant::now();
        let mut dead = Vec::new();
        for (&rank, info) in &self.peers {
            if info.alive && now.duration_since(info.last_heartbeat) > timeout {
                dead.push(rank);
            }
        }
        for &rank in &dead {
            self.remove_peer(rank);
        }
        dead
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn num_alive(&self) -> usize {
        self.peers.values().filter(|p| p.alive).count()
    }
}

impl Default for ClusterMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_add_and_list_peers() {
        let mut map = ClusterMap::new();
        map.add_peer(0, "127.0.0.1:5000".into());
        map.add_peer(1, "127.0.0.1:5001".into());
        assert_eq!(map.num_alive(), 2);
        assert_eq!(map.epoch(), 2);
    }

    #[test]
    fn test_remove_peer() {
        let mut map = ClusterMap::new();
        map.add_peer(0, "a".into());
        map.add_peer(1, "b".into());
        assert!(map.remove_peer(1));
        assert_eq!(map.num_alive(), 1);
        // Removing again is a no-op.
        assert!(!map.remove_peer(1));
    }

    #[test]
    fn test_heartbeat() {
        let mut map = ClusterMap::new();
        map.add_peer(0, "a".into());
        let t1 = map.get(0).unwrap().last_heartbeat;
        std::thread::sleep(Duration::from_millis(10));
        map.heartbeat(0);
        let t2 = map.get(0).unwrap().last_heartbeat;
        assert!(t2 > t1);
    }

    #[test]
    fn test_check_timeouts() {
        let mut map = ClusterMap::new();
        map.add_peer(0, "a".into());
        // Immediately check with very short timeout shouldn't detect anything
        // because we just added it.
        let dead = map.check_timeouts(Duration::from_secs(60));
        assert!(dead.is_empty());
    }
}
