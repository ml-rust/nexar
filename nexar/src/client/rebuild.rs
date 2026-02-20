use crate::error::{NexarError, Result};
use crate::rpc::registry::RpcRegistry;
use crate::types::Rank;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, RwLock};

use super::NexarClient;
use super::async_client::RawRecvSource;

impl NexarClient {
    /// Rebuild the communicator excluding dead ranks.
    ///
    /// This is a **local** operation: each surviving rank independently computes
    /// the new rank mapping from the agreed `dead_ranks` list. All survivors must
    /// call this with the **same** `dead_ranks` to get a consistent view.
    ///
    /// Returns a new `NexarClient` with contiguous ranks `[0, survivors)`,
    /// `world_size = survivors`, and the relative order of surviving ranks preserved.
    ///
    /// The new client shares the parent's QUIC connections and routers. It uses
    /// per-comm_id raw channels so its collectives don't interfere with the parent.
    pub async fn rebuild_excluding(&self, dead_ranks: &[Rank]) -> Result<NexarClient> {
        debug_assert!(
            !dead_ranks.contains(&self.rank),
            "a dead rank should not call rebuild_excluding"
        );

        // Compute surviving ranks in order.
        let mut survivors: Vec<Rank> = (0..self.world_size)
            .filter(|r| !dead_ranks.contains(r))
            .collect();
        survivors.sort();

        let new_world_size = survivors.len() as u32;
        let new_rank =
            survivors
                .iter()
                .position(|&r| r == self.rank)
                .ok_or(NexarError::CollectiveFailed {
                    operation: "rebuild_excluding",
                    rank: self.rank,
                    reason: "rank not found among survivors".into(),
                })? as Rank;

        // Generate a deterministic comm_id from the dead set so all survivors agree.
        // Hash dead ranks in sorted order so all survivors compute the same comm_id
        // regardless of the order dead_ranks was constructed.
        let mut sorted_dead: Vec<Rank> = dead_ranks.to_vec();
        sorted_dead.sort();

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
            for &dr in &sorted_dead {
                for b in dr.to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            if h == 0 { 1 } else { h }
        };

        // Build rank_map and peer subset.
        let mut rank_map = HashMap::new();
        let mut new_peers = HashMap::new();
        let mut comm_receivers = HashMap::new();

        for (new_r, &orig_rank) in survivors.iter().enumerate() {
            let new_r = new_r as Rank;
            rank_map.insert(new_r, orig_rank);

            if orig_rank != self.rank {
                let peer = self.peer(orig_rank)?;
                new_peers.insert(new_r, Arc::clone(peer));

                if let Some(router) = self.routers.get(&orig_rank) {
                    let rx = router.register_comm(new_comm_id).await;
                    comm_receivers.insert(new_r, Mutex::new(rx));
                }
            }
        }

        Ok(NexarClient {
            rank: new_rank,
            world_size: new_world_size,
            comm_id: new_comm_id,
            peers: new_peers,
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
