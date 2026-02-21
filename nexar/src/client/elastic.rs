use crate::error::{NexarError, Result};
use crate::rpc::registry::RpcRegistry;
use crate::transport::PeerConnection;
use crate::types::Rank;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, RwLock};

use super::NexarClient;
use super::async_client::RawRecvSource;

impl NexarClient {
    /// Rebuild adding new peer connections (elastic grow).
    ///
    /// New ranks are appended after the existing world (no remapping of existing ranks).
    /// Returns a new `NexarClient` with `world_size += new_peers.len()`.
    ///
    /// The new client shares the parent's QUIC connections and routers. It uses
    /// per-comm_id raw channels so its collectives don't interfere with the parent.
    pub async fn rebuild_adding(
        &self,
        new_peers: Vec<(Rank, Arc<PeerConnection>)>,
    ) -> Result<NexarClient> {
        if new_peers.is_empty() {
            return Err(NexarError::Elastic("no new peers to add".into()));
        }

        let new_world_size = self.world_size + new_peers.len() as u32;

        // Generate a deterministic comm_id from the existing comm_id + joined ranks.
        let rebuild_gen = self.split_generation.fetch_add(1, Ordering::Relaxed);
        let new_comm_id = {
            let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
            for b in self.comm_id.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for b in rebuild_gen.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for &(rank, _) in &new_peers {
                for b in rank.to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            if h == 0 { 1 } else { h }
        };

        // Merge existing peers + new peers.
        let mut merged_peers = HashMap::new();
        let mut comm_receivers = HashMap::new();

        // Keep existing peers (keyed by their current rank, which stays the same).
        for (&peer_rank, peer) in &self.peers {
            merged_peers.insert(peer_rank, Arc::clone(peer));

            if let Some(router) = self.routers.get(&peer_rank) {
                let rx = router.register_comm(new_comm_id).await;
                comm_receivers.insert(peer_rank, Mutex::new(rx));
            }
        }

        // Add new peers.
        for (rank, peer) in new_peers {
            merged_peers.insert(rank, peer);
            // New peers don't have routers yet in the parent — their router
            // must be spawned separately by the ElasticManager before calling this.
        }

        // Rank map: identity mapping (no remapping for grow).
        let mut rank_map = HashMap::new();
        for r in 0..new_world_size {
            if r != self.rank {
                rank_map.insert(r, r);
            }
        }

        Ok(NexarClient {
            rank: self.rank,
            world_size: new_world_size,
            comm_id: new_comm_id,
            peers: merged_peers,
            routers: HashMap::new(),
            raw_recv: RawRecvSource::Comm(comm_receivers),
            _router_handles: Vec::new(),
            adapter: Arc::clone(&self.adapter),
            _pool: Arc::clone(&self._pool),
            barrier_epoch: AtomicU64::new(0),
            rpc_registry: Arc::new(RwLock::new(RpcRegistry::new())),
            rpc_req_id: AtomicU64::new(0),
            split_generation: AtomicU64::new(0),
            rank_map,
            collective_tag: AtomicU64::new(1),
            tagged_receivers: Mutex::new(HashMap::new()),
            config: Arc::clone(&self.config),
            failure_tx: Arc::clone(&self.failure_tx),
            failure_rx: self.failure_rx.clone(),
            _monitor_handle: None,
        })
    }
}
