use crate::error::{NexarError, Result};
use crate::rpc::registry::RpcRegistry;
use crate::types::Rank;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

use super::NexarClient;
use super::hash::fnv1a_comm_id;

impl NexarClient {
    /// Split this communicator into sub-groups.
    ///
    /// All ranks must call `split` with the same arguments simultaneously.
    /// Ranks with the same `color` end up in the same sub-communicator.
    /// Within each group, ranks are ordered by `key` (ties broken by original rank).
    ///
    /// The returned client has new rank/world_size within the sub-group and uses
    /// a unique `comm_id` for its raw stream traffic, so collectives on the
    /// sub-communicator don't interfere with the parent or other sub-groups.
    ///
    /// The parent client's routers demux raw streams by `comm_id`, so the parent
    /// must remain alive for the duration of the split client's use.
    pub async fn split(&self, color: u32, key: u32) -> Result<NexarClient> {
        let world = self.world_size as usize;
        let rank = self.rank;

        // Step 1: Exchange (color, key, split_generation) tuples with all peers.
        // Encode as 16 bytes: [color: u32 LE][key: u32 LE][generation: u64 LE].
        let current_gen = self.split_generation.load(Ordering::Relaxed);
        let mut my_info = [0u8; 16];
        my_info[..4].copy_from_slice(&color.to_le_bytes());
        my_info[4..8].copy_from_slice(&key.to_le_bytes());
        my_info[8..16].copy_from_slice(&current_gen.to_le_bytes());

        // AllGather the info from all ranks.
        let mut all_info = vec![0u8; 16 * world];
        all_info[rank as usize * 16..(rank as usize + 1) * 16].copy_from_slice(&my_info);

        // Use the existing allgather collective. We pass raw pointers to our
        // stack-allocated buffers.
        let send_ptr = my_info.as_ptr() as u64;
        let recv_ptr = all_info.as_mut_ptr() as u64;
        let tag = self.next_collective_tag();
        unsafe {
            crate::collective::ring_allgather(
                self,
                send_ptr,
                recv_ptr,
                16, // 16 bytes per rank
                crate::types::DataType::U8,
                Some(tag),
            )
            .await?;
        }

        // Step 2: Parse all (color, key) tuples and validate generation consistency.
        let mut entries: Vec<(Rank, u32, u32)> = Vec::with_capacity(world);
        for r in 0..world {
            let off = r * 16;
            let c = u32::from_le_bytes(
                all_info[off..off + 4]
                    .try_into()
                    .map_err(|_| NexarError::DecodeFailed("split color bytes".into()))?,
            );
            let k = u32::from_le_bytes(
                all_info[off + 4..off + 8]
                    .try_into()
                    .map_err(|_| NexarError::DecodeFailed("split key bytes".into()))?,
            );
            let peer_gen = u64::from_le_bytes(
                all_info[off + 8..off + 16]
                    .try_into()
                    .map_err(|_| NexarError::DecodeFailed("split generation bytes".into()))?,
            );
            if peer_gen != current_gen {
                return Err(NexarError::CollectiveFailed {
                    operation: "split",
                    rank: r as Rank,
                    reason: format!(
                        "split_generation mismatch: local={current_gen}, rank {r}={peer_gen}"
                    ),
                });
            }
            entries.push((r as Rank, c, k));
        }

        // Step 3: Find our group (same color), sort by (key, original_rank).
        let my_color = color;
        let mut group: Vec<(Rank, u32)> = entries
            .iter()
            .filter(|&&(_, c, _)| c == my_color)
            .map(|&(r, _, k)| (r, k))
            .collect();
        group.sort_by_key(|&(orig_rank, k)| (k, orig_rank));

        let new_world_size = group.len() as u32;
        let new_rank =
            group
                .iter()
                .position(|&(r, _)| r == rank)
                .ok_or(NexarError::CollectiveFailed {
                    operation: "split",
                    rank,
                    reason: "rank not found in its own color group".into(),
                })? as Rank;

        // Step 4: Generate a deterministic comm_id agreed upon by all ranks.
        // All ranks in this communicator advance split_generation in lockstep
        // (split is collective). Combine parent comm_id, generation, and color
        // to produce a unique comm_id per split group.
        let split_gen = self.split_generation.fetch_add(1, Ordering::Relaxed);
        let new_comm_id = fnv1a_comm_id([
            &self.comm_id.to_le_bytes()[..],
            &split_gen.to_le_bytes(),
            &my_color.to_le_bytes(),
        ]);

        // Step 5: Build rank_map (new_rank -> global rank) and peer subset.
        let mut rank_map = HashMap::new();
        let mut new_peers = HashMap::new();
        let mut routers = HashMap::new();

        for (new_r, &(orig_rank, _)) in group.iter().enumerate() {
            let new_r = new_r as Rank;
            let global = self.resolve_rank(orig_rank);
            rank_map.insert(new_r, global);

            if orig_rank != rank {
                // Share the parent's PeerConnection (keyed by the parent's rank).
                let peer = self.peer(orig_rank)?;
                new_peers.insert(new_r, Arc::clone(peer));
                let router = self
                    .routers
                    .get(&global)
                    .ok_or(NexarError::UnknownPeer { rank: orig_rank })?;
                routers.insert(global, Arc::clone(router));
            }
        }

        // Step 6: Build the split client. It shares the parent's routers.
        // Every receive key carries the comm_id, so the two never take each
        // other's messages.

        Ok(NexarClient {
            rank: new_rank,
            world_size: new_world_size,
            comm_id: new_comm_id,
            peers: new_peers,
            routers,
            _router_handles: Vec::new(),
            adapter: Arc::clone(&self.adapter),
            _pool: Arc::clone(&self._pool),
            barrier_epoch: AtomicU64::new(0),
            rpc_registry: Arc::new(RwLock::new(RpcRegistry::new())),
            rpc_req_id: AtomicU64::new(0),
            split_generation: AtomicU64::new(0),
            rank_map,
            collective_tag: AtomicU64::new(1),
            config: Arc::clone(&self.config),
            failure_tx: Arc::clone(&self.failure_tx),
            failure_rx: self.failure_rx.clone(),
            _monitor_handle: None, // Split clients share the parent's monitor
            routing_table: self.routing_table.clone(),
            relay_deliveries: self.relay_deliveries.clone(),
            _relay_handles: Vec::new(), // Split clients share the parent's relay
            mesh: Arc::clone(&self.mesh),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::client::bootstrap::{local_pair, shutdown_pair};

    #[tokio::test]
    async fn test_split_and_parent_tags_stay_apart_over_quic() {
        let (a, b) = local_pair(16).await;
        let (sa, sb) = tokio::join!(a.split(0, 0), b.split(0, 1));
        let (sa, sb) = (sa.unwrap(), sb.unwrap());
        // Same tag on both communicators; the split one is sent first.
        sa.send_bytes_tagged(1, 7, b"split").await.unwrap();
        a.send_bytes_tagged(1, 7, b"parent").await.unwrap();
        assert_eq!(&b.recv_bytes_tagged(0, 7).await.unwrap()[..], b"parent");
        assert_eq!(&sb.recv_bytes_tagged(0, 7).await.unwrap()[..], b"split");
        // Barriers of both communicators, interleaved, without the sidecar.
        let (r1, r2, r3, r4) = tokio::join!(sa.barrier(), a.barrier(), b.barrier(), sb.barrier());
        for r in [r1, r2, r3, r4] {
            r.unwrap();
        }
        drop((sa, sb));
        shutdown_pair(a, b).await;
    }
}
