//! Point-to-point messaging, RPC, relay, and scatter-gather I/O.

use crate::cluster::sparse::RoutingTable;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::transport::PeerRouter;
use crate::transport::buffer_pool::PooledBuf;
use crate::transport::lane::{ControlKey, LaneKey};
use crate::transport::relay::RelayDeliveries;
use crate::types::{Priority, Rank};
use std::collections::HashMap;
use std::sync::Arc;

use super::NexarClient;

impl NexarClient {
    /// Call a remote function on the target rank and wait for the response.
    pub async fn rpc(&self, target: Rank, fn_id: u16, args: &[u8]) -> Result<Vec<u8>> {
        self.rpc_with_timeout(target, fn_id, args, self.config.rpc_timeout)
            .await
    }

    /// Call a remote function with a custom timeout.
    pub async fn rpc_with_timeout(
        &self,
        target: Rank,
        fn_id: u16,
        args: &[u8],
        timeout: std::time::Duration,
    ) -> Result<Vec<u8>> {
        let req_id = self
            .rpc_req_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let router = self.router(target)?;

        let rx = router.register_rpc_waiter(req_id).await;

        let peer = self.peer(target)?;
        let request = NexarMessage::Rpc {
            req_id,
            fn_id,
            payload: args.to_vec(),
        };
        if let Err(e) = peer.send_message(&request, Priority::Realtime).await {
            router.remove_rpc_waiter(req_id).await;
            return Err(e);
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(msg)) => match msg {
                NexarMessage::RpcResponse { payload, .. } => Ok(payload),
                other => Err(NexarError::RpcFailed {
                    rank: target,
                    reason: format!("expected RpcResponse, got {other:?}"),
                }),
            },
            Ok(Err(_)) => Err(NexarError::PeerDisconnected { rank: target }),
            Err(_) => {
                router.remove_rpc_waiter(req_id).await;
                Err(NexarError::RpcFailed {
                    rank: target,
                    reason: format!(
                        "RPC fn_id={fn_id} timed out after {}ms",
                        timeout.as_millis()
                    ),
                })
            }
        }
    }

    /// Receive the control message with key `key` from a direct peer.
    pub(crate) async fn recv_control(&self, src: Rank, key: ControlKey) -> Result<NexarMessage> {
        self.router(src)?.recv_control(src, key).await
    }

    /// Receive the next RPC request from the rpc_requests lane for a given peer.
    pub async fn recv_rpc_request(&self, src: Rank) -> Result<NexarMessage> {
        self.router(src)?.recv_rpc_request(src).await
    }

    /// The router that receives from `src` (a rank of this communicator).
    pub(super) fn router(&self, src: Rank) -> Result<&PeerRouter> {
        self.routers
            .get(&self.resolve_rank(src))
            .map(|r| r.as_ref())
            .ok_or(NexarError::UnknownPeer { rank: src })
    }

    /// Send tagged data to a specific rank.
    ///
    /// Messages of one `(dest, tag)` arrive in send order. A non-neighbor in a
    /// sparse topology receives through relay hops, without that order.
    ///
    /// # Safety
    /// `data_ptr` must be valid for `size` bytes.
    pub async unsafe fn send(
        &self,
        data_ptr: u64,
        size: usize,
        dest: Rank,
        tag: u32,
    ) -> Result<()> {
        self.check_rank(dest)?;
        let data = unsafe { self.adapter.stage_for_send(data_ptr, size)? };
        self.send_data(dest, tag, data).await
    }

    /// Receive data with tag `tag` from `src`.
    ///
    /// Waits for the next message from `src` with this tag, in send order.
    /// Messages with other tags stay queued for their own receivers.
    ///
    /// # Safety
    /// `buf_ptr` must be valid for `size` bytes.
    pub async unsafe fn recv(&self, buf_ptr: u64, size: usize, src: Rank, tag: u32) -> Result<()> {
        self.check_rank(src)?;
        let payload = self.recv_data(src, tag).await?;
        if payload.len() != size {
            return Err(NexarError::BufferSizeMismatch {
                expected: size,
                actual: payload.len(),
            });
        }
        unsafe { self.adapter.receive_to_device(&payload, buf_ptr) }
    }

    fn check_rank(&self, rank: Rank) -> Result<()> {
        if rank >= self.world_size {
            return Err(NexarError::InvalidRank {
                rank,
                world_size: self.world_size,
            });
        }
        Ok(())
    }

    /// Send staged point-to-point data: the data lane to a direct peer, a
    /// relayed `Data` message to a non-neighbor.
    async fn send_data(&self, dest: Rank, tag: u32, payload: Vec<u8>) -> Result<()> {
        if let Some(peer) = self.peers.get(&dest) {
            let lane = LaneKey::Data {
                comm_id: self.comm_id,
                tag,
            };
            return peer.send_lane(lane, &payload).await;
        }
        let msg = NexarMessage::Data {
            tag,
            src_rank: self.rank,
            payload,
        };
        self.send_message_to(dest, &msg, Priority::Bulk).await
    }

    /// Receive the next point-to-point payload from `src` on `tag`.
    async fn recv_data(&self, src: Rank, tag: u32) -> Result<PooledBuf> {
        if !self.has_direct_peer(src)
            && let Some(deliveries) = &self.relay_deliveries
        {
            let payload = deliveries.recv_data(self.resolve_rank(src), tag).await?;
            return Ok(PooledBuf::from_vec(payload, Arc::clone(&self._pool)));
        }
        let lane = LaneKey::Data {
            comm_id: self.comm_id,
            tag,
        };
        self.router(src)?.recv_lane(src, lane).await
    }

    /// Send non-contiguous data (scatter-gather) to a specific rank.
    ///
    /// Gathers the regions into a contiguous buffer and sends as a single
    /// message on the same lane as [`send`](Self::send).
    ///
    /// # Safety
    /// Each region's `ptr` must be valid for its `len` bytes.
    pub async unsafe fn send_iov(
        &self,
        regions: &[crate::types::IoVec],
        dest: Rank,
        tag: u32,
    ) -> Result<()> {
        self.check_rank(dest)?;
        let data = unsafe { self.adapter.stage_for_send_iov(regions)? };
        self.send_data(dest, tag, data).await
    }

    /// Receive data and scatter into non-contiguous regions.
    ///
    /// # Safety
    /// Each region's `ptr` must be valid for its `len` bytes.
    /// The total size of all regions must match the received data size.
    pub async unsafe fn recv_iov(
        &self,
        regions: &[crate::types::IoVec],
        src: Rank,
        tag: u32,
    ) -> Result<()> {
        self.check_rank(src)?;
        let expected: usize = regions.iter().map(|r| r.len).sum();
        let payload = self.recv_data(src, tag).await?;
        if payload.len() != expected {
            return Err(NexarError::BufferSizeMismatch {
                expected,
                actual: payload.len(),
            });
        }
        unsafe { self.adapter.receive_to_device_iov(&payload, regions) }
    }

    /// Send a framed message to any rank, using relay if needed.
    ///
    /// For full mesh or direct neighbors, sends directly.
    /// For sparse topology non-neighbors, wraps in Relay and sends to next_hop.
    pub async fn send_message_to(
        &self,
        dest: Rank,
        msg: &NexarMessage,
        priority: Priority,
    ) -> Result<()> {
        if let Some(peer) = self.peers.get(&dest) {
            peer.send_message(msg, priority).await
        } else if let Some(ref rt) = self.routing_table {
            let route = crate::transport::relay::RelayRoute {
                my_rank: self.rank,
                peers: &self.peers,
                routing_table: rt,
                strategy: &self.config.topology,
                world_size: self.world_size,
            };
            crate::transport::relay::send_or_relay_message(route, dest, msg, priority).await
        } else {
            Err(NexarError::UnknownPeer { rank: dest })
        }
    }

    /// Receive the control message with key `key` from any rank, using
    /// relay delivery if needed.
    pub(crate) async fn recv_control_from(
        &self,
        src: Rank,
        key: ControlKey,
    ) -> Result<NexarMessage> {
        let original_src = self.resolve_rank(src);
        if let Some(router) = self.routers.get(&original_src) {
            // Direct peer: use router.
            router.recv_control(src, key).await
        } else if let Some(ref deliveries) = self.relay_deliveries {
            // Non-neighbor: receive via relay.
            deliveries.recv_control(original_src, key).await
        } else {
            Err(NexarError::UnknownPeer { rank: src })
        }
    }

    /// Set up the relay infrastructure for sparse topology.
    ///
    /// Must be called after construction when using sparse topology.
    /// Takes relay receivers from the routers and starts relay listener tasks.
    pub(crate) async fn setup_relay(&mut self, routing_table: Arc<RoutingTable>) {
        // A rebuilt client keeps the parent's deliveries: the parent's relay
        // listeners still deliver there.
        let deliveries = self
            .relay_deliveries
            .clone()
            .unwrap_or_else(|| Arc::new(RelayDeliveries::new()));
        self.relay_deliveries = Some(Arc::clone(&deliveries));
        self.routing_table = Some(Arc::clone(&routing_table));

        // Take relay receivers from each router.
        let mut relay_receivers = HashMap::new();
        for (&peer_rank, router) in &self.routers {
            if let Some(rx) = router.take_relay_rx().await {
                relay_receivers.insert(peer_rank, rx);
            }
        }

        // Convert peers to Arc<HashMap> for the relay listener.
        let peers_arc = Arc::new(self.peers.clone());

        let topology = crate::transport::relay::RelayTopology {
            my_rank: self.rank,
            peers: peers_arc,
            routing_table,
            strategy: self.config.topology.clone(),
            world_size: self.world_size,
        };
        let handles = crate::transport::relay::start_relay_listeners(
            topology,
            relay_receivers,
            deliveries,
            Arc::clone(&self._pool),
        );
        self._relay_handles = handles;
    }
}

#[cfg(test)]
mod tests {
    use crate::client::NexarClient;
    use crate::client::bootstrap::{local_pair, shutdown_pair};
    use std::time::Duration;

    async fn send_u32(c: &NexarClient, dest: u32, tag: u32, value: u32) {
        let data = value.to_le_bytes();
        unsafe { c.send(data.as_ptr() as u64, 4, dest, tag).await.unwrap() };
    }

    async fn recv_u32(c: &NexarClient, src: u32, tag: u32) -> u32 {
        let mut out = [0u8; 4];
        unsafe { c.recv(out.as_mut_ptr() as u64, 4, src, tag).await.unwrap() };
        u32::from_le_bytes(out)
    }

    #[tokio::test]
    async fn test_recv_skips_messages_of_other_tags_without_consuming() {
        let (a, b) = local_pair(16).await;
        send_u32(&a, 1, 1, 100).await;
        send_u32(&a, 1, 2, 200).await;
        send_u32(&a, 1, 1, 101).await;
        // Tag 2 first: the two tag-1 messages stay queued for their receiver.
        assert_eq!(recv_u32(&b, 0, 2).await, 200);
        assert_eq!(recv_u32(&b, 0, 1).await, 100);
        assert_eq!(recv_u32(&b, 0, 1).await, 101);
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_cancelled_recv_consumes_nothing() {
        let (a, b) = local_pair(16).await;
        let early = tokio::time::timeout(Duration::from_millis(100), recv_u32(&b, 0, 5)).await;
        assert!(early.is_err());
        send_u32(&a, 1, 5, 55).await;
        assert_eq!(recv_u32(&b, 0, 5).await, 55);
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_data_lane_keeps_send_order() {
        let (a, b) = local_pair(4).await;
        let count = 64u32;
        let sender = async {
            for i in 0..count {
                send_u32(&a, 1, 9, i).await;
            }
        };
        let receiver = async {
            for want in 0..count {
                assert_eq!(recv_u32(&b, 0, 9).await, want);
            }
        };
        tokio::join!(sender, receiver);
        shutdown_pair(a, b).await;
    }
}
