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

        // Step 1: Exchange (color, key) tuples with all peers.
        // Encode as 8 bytes: [color: u32 LE][key: u32 LE].
        let mut my_info = [0u8; 8];
        my_info[..4].copy_from_slice(&color.to_le_bytes());
        my_info[4..].copy_from_slice(&key.to_le_bytes());

        // AllGather the info from all ranks.
        let mut all_info = vec![0u8; 8 * world];
        all_info[rank as usize * 8..(rank as usize + 1) * 8].copy_from_slice(&my_info);

        // Use the existing allgather collective. We pass raw pointers to our
        // stack-allocated buffers.
        let send_ptr = my_info.as_ptr() as u64;
        let recv_ptr = all_info.as_mut_ptr() as u64;
        unsafe {
            crate::collective::ring_allgather(
                self,
                send_ptr,
                recv_ptr,
                8, // 8 bytes per rank
                crate::types::DataType::U8,
            )
            .await?;
        }

        // Step 2: Parse all (color, key) tuples.
        let mut entries: Vec<(Rank, u32, u32)> = Vec::with_capacity(world);
        for r in 0..world {
            let off = r * 8;
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
        let new_rank = group
            .iter()
            .position(|&(r, _)| r == rank)
            .expect("rank must be in its own color group") as Rank;

        // Step 4: Generate a deterministic comm_id agreed upon by all ranks.
        // All ranks in this communicator advance split_generation in lockstep
        // (split is collective). Combine parent comm_id, generation, and color
        // to produce a unique comm_id per split group.
        let split_gen = self.split_generation.fetch_add(1, Ordering::Relaxed);
        let new_comm_id = {
            // Hash (parent_comm_id, generation, color) to produce a non-zero u32.
            let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
            for b in self.comm_id.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for b in split_gen.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for b in my_color.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            // Fold to u32, ensure non-zero (0 is reserved for root comm).
            let id = ((h >> 32) ^ h) as u32;
            if id == 0 { 1 } else { id }
        };

        // Step 5: Build rank_map (new_rank -> original_rank) and peer subset.
        let mut rank_map = HashMap::new();
        let mut new_peers = HashMap::new();
        let mut comm_receivers = HashMap::new();

        for (new_r, &(orig_rank, _)) in group.iter().enumerate() {
            let new_r = new_r as Rank;
            rank_map.insert(new_r, orig_rank);

            if orig_rank != rank {
                // Share the parent's PeerConnection (keyed by original rank).
                let peer = self.peer(orig_rank)?;
                new_peers.insert(new_r, Arc::clone(peer));

                // Register a per-comm_id channel on the parent's router for this peer.
                let original_rank_key = orig_rank;
                let router =
                    self.routers
                        .get(&original_rank_key)
                        .ok_or(NexarError::UnknownPeer {
                            rank: original_rank_key,
                        })?;
                let rx = router.register_comm(new_comm_id).await;
                comm_receivers.insert(new_r, Mutex::new(rx));
            }
        }

        // Step 6: Build the split client. It shares the parent's routers
        // but uses comm-specific raw channels.
        // Note: The split client doesn't own routers or router handles — it
        // borrows the parent's routers indirectly through the registered comm channels.
        // Control/data/RPC lanes are still on the parent's routers.
        // Split clients don't have their own routers — they use comm channels for
        // collective raw data. Barrier uses comm_id in the message to avoid cross-talk.

        Ok(NexarClient {
            rank: new_rank,
            world_size: new_world_size,
            comm_id: new_comm_id,
            peers: new_peers,
            routers: HashMap::new(), // Split clients don't own routers
            raw_recv: RawRecvSource::Comm(comm_receivers),
            _router_handles: Vec::new(),
            adapter: Arc::clone(&self.adapter),
            _pool: Arc::clone(&self._pool),
            barrier_epoch: AtomicU64::new(0),
            rpc_registry: Arc::new(RwLock::new(RpcRegistry::new())),
            rpc_req_id: AtomicU64::new(0),
            split_generation: AtomicU64::new(0),
            rank_map,
        })
    }
}
