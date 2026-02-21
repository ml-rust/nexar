//! Point-to-point messaging, RPC, relay, and scatter-gather I/O.

use crate::cluster::sparse::RoutingTable;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
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

        // For split clients, resolve to the original rank for router lookup.
        let original_target = self.resolve_rank(target);
        let router = self
            .routers
            .get(&original_target)
            .ok_or(NexarError::UnknownPeer { rank: target })?;

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

    /// Receive the next message from the control lane for a given peer.
    pub(crate) async fn recv_control(&self, src: Rank) -> Result<NexarMessage> {
        let original_src = self.resolve_rank(src);
        let router = self
            .routers
            .get(&original_src)
            .ok_or(NexarError::UnknownPeer { rank: src })?;
        router.recv_control(original_src).await
    }

    /// Receive the next RPC request from the rpc_requests lane for a given peer.
    pub async fn recv_rpc_request(&self, src: Rank) -> Result<NexarMessage> {
        let original_src = self.resolve_rank(src);
        let router = self
            .routers
            .get(&original_src)
            .ok_or(NexarError::UnknownPeer { rank: src })?;
        router.recv_rpc_request(original_src).await
    }

    /// Receive the next data message from the data lane for a given peer.
    pub(super) async fn recv_data_message(&self, src: Rank) -> Result<NexarMessage> {
        let original_src = self.resolve_rank(src);
        let router = self
            .routers
            .get(&original_src)
            .ok_or(NexarError::UnknownPeer { rank: src })?;
        router.recv_data(original_src).await
    }

    /// Send non-contiguous data (scatter-gather) to a specific rank.
    ///
    /// Gathers the regions into a contiguous buffer and sends as a single message.
    ///
    /// # Safety
    /// Each region's `ptr` must be valid for its `len` bytes.
    pub async unsafe fn send_iov(
        &self,
        regions: &[crate::types::IoVec],
        dest: Rank,
        tag: u32,
    ) -> Result<()> {
        if dest >= self.world_size {
            return Err(NexarError::InvalidRank {
                rank: dest,
                world_size: self.world_size,
            });
        }
        let data = unsafe { self.adapter.stage_for_send_iov(regions)? };
        let peer = self.peer(dest)?;
        let msg = NexarMessage::Data {
            tag,
            src_rank: self.rank,
            payload: data,
        };
        peer.send_message(&msg, Priority::Bulk).await
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
        if src >= self.world_size {
            return Err(NexarError::InvalidRank {
                rank: src,
                world_size: self.world_size,
            });
        }

        let expected: usize = regions.iter().map(|r| r.len).sum();
        let msg = self.recv_data_message(src).await?;

        match msg {
            NexarMessage::Data {
                tag: recv_tag,
                payload,
                ..
            } => {
                if recv_tag != tag {
                    return Err(NexarError::DecodeFailed(format!(
                        "tag mismatch: expected {tag}, got {recv_tag}"
                    )));
                }
                if payload.len() != expected {
                    return Err(NexarError::BufferSizeMismatch {
                        expected,
                        actual: payload.len(),
                    });
                }
                unsafe { self.adapter.receive_to_device_iov(&payload, regions)? };
                Ok(())
            }
            other => Err(NexarError::DecodeFailed(format!(
                "expected Data message, got {other:?}"
            ))),
        }
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
            crate::transport::relay::send_or_relay_message(
                self.rank,
                &self.peers,
                rt,
                dest,
                msg,
                priority,
            )
            .await
        } else {
            Err(NexarError::UnknownPeer { rank: dest })
        }
    }

    /// Receive a control message from any rank, using relay delivery if needed.
    pub(crate) async fn recv_control_from(&self, src: Rank) -> Result<NexarMessage> {
        let original_src = self.resolve_rank(src);
        if self.routers.contains_key(&original_src) {
            // Direct peer: use router.
            let router = self.routers.get(&original_src).unwrap();
            router.recv_control(original_src).await
        } else if let Some(ref deliveries) = self.relay_deliveries {
            // Non-neighbor: receive via relay.
            deliveries.recv_control(original_src).await
        } else {
            Err(NexarError::UnknownPeer { rank: src })
        }
    }

    /// Set up the relay infrastructure for sparse topology.
    ///
    /// Must be called after construction when using sparse topology.
    /// Takes relay receivers from the routers and starts relay listener tasks.
    pub(crate) async fn setup_relay(&mut self, routing_table: Arc<RoutingTable>) {
        let deliveries = Arc::new(RelayDeliveries::new());
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

        let handles = crate::transport::relay::start_relay_listeners(
            self.rank,
            peers_arc,
            routing_table,
            relay_receivers,
            deliveries,
            Arc::clone(&self._pool),
        );
        self._relay_handles = handles;
    }
}
