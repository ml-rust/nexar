use crate::cluster::sparse::recompute_routing_table;
use crate::error::{NexarError, Result};
use crate::rpc::registry::RpcRegistry;
use crate::transport::PeerConnection;
use crate::transport::router::PeerRouter;
use crate::types::Rank;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, RwLock};

use super::NexarClient;
use super::async_client::RawRecvSource;
use super::hash::fnv1a_comm_id;

impl NexarClient {
    /// Rebuild adding new peer connections (elastic grow).
    ///
    /// New ranks are appended after the existing world (no remapping of existing ranks).
    /// Returns a new `NexarClient` with `world_size += new_peers.len()`.
    ///
    /// The new client shares the parent's QUIC connections and routers. It uses
    /// per-comm_id raw channels so its collectives don't interfere with the parent.
    ///
    /// For sparse topologies, relay infrastructure is inherited from the parent for
    /// existing peers and set up for new peers automatically.
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
            let mut parts: Vec<Vec<u8>> = vec![
                self.comm_id.to_le_bytes().to_vec(),
                rebuild_gen.to_le_bytes().to_vec(),
            ];
            for &(rank, _) in &new_peers {
                parts.push(rank.to_le_bytes().to_vec());
            }
            fnv1a_comm_id(&parts)
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

        // Add new peers and spawn routers for them.
        let mut new_routers = HashMap::new();
        let mut new_router_handles = Vec::new();
        for (rank, peer) in new_peers {
            let conn_clone = peer.conn.clone();
            let (router, handle) = PeerRouter::spawn(rank, conn_clone, Arc::clone(&self._pool));
            let rx = router.register_comm(new_comm_id).await;
            comm_receivers.insert(rank, Mutex::new(rx));
            new_routers.insert(rank, router);
            new_router_handles.push(handle);
            merged_peers.insert(rank, peer);
        }

        // Rank map: identity mapping (no remapping for grow).
        let mut rank_map = HashMap::new();
        for r in 0..new_world_size {
            if r != self.rank {
                rank_map.insert(r, r);
            }
        }

        let routing_table =
            recompute_routing_table(&self.config.topology, self.rank, new_world_size);

        // Inherit parent's relay deliveries for existing peers. For sparse topologies,
        // new peer routers also need relay listeners set up below.
        let relay_deliveries = self.relay_deliveries.clone();

        let mut client = NexarClient {
            rank: self.rank,
            world_size: new_world_size,
            comm_id: new_comm_id,
            peers: merged_peers,
            routers: new_routers,
            raw_recv: RawRecvSource::Comm(comm_receivers),
            _router_handles: new_router_handles,
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
            routing_table: routing_table.clone(),
            relay_deliveries,
            _relay_handles: Vec::new(),
            _endpoints: Vec::new(),
        };

        // Set up relay listeners for new peer routers in sparse topologies.
        if let Some(rt) = routing_table {
            client.setup_relay(rt).await;
        }

        Ok(client)
    }
}
