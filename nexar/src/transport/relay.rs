use crate::cluster::sparse::{RoutingTable, TopologyStrategy, find_alternative_hop};
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::transport::PeerConnection;
use crate::transport::buffer_pool::PooledBuf;
use crate::transport::lane::ControlKey;
use crate::transport::mailbox::KeyedMailbox;
use crate::types::{Priority, Rank};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};

/// Per-key queue depth for relay delivery.
const RELAY_QUEUE_DEPTH: usize = 256;

/// Most relayed messages that wait at once for room in their key's queue.
const MAX_PENDING_DELIVERIES: usize = 1024;

/// Manages relay delivery queues for messages arriving via intermediate hops.
///
/// When a node receives a relayed message destined for itself, the relay listener
/// deposits it here. Consumers (collectives, barrier, point-to-point) read from
/// these queues as if the message came directly from the source.
///
/// Each queue is keyed by source rank and receive key, so a receiver takes
/// only messages of its own key. Relayed messages cross several hops on
/// separate streams: no order holds between them, not even within one key.
pub struct RelayDeliveries {
    control: KeyedMailbox<(Rank, ControlKey), NexarMessage>,
    tagged: KeyedMailbox<(Rank, u64), PooledBuf>,
    data: KeyedMailbox<(Rank, u32), Vec<u8>>,
}

impl Default for RelayDeliveries {
    fn default() -> Self {
        Self {
            control: KeyedMailbox::new(RELAY_QUEUE_DEPTH),
            tagged: KeyedMailbox::new(RELAY_QUEUE_DEPTH),
            data: KeyedMailbox::new(RELAY_QUEUE_DEPTH),
        }
    }
}

impl RelayDeliveries {
    pub fn new() -> Self {
        Self::default()
    }

    /// Receive a relayed control message with key `key` from `src`.
    pub async fn recv_control(&self, src: Rank, key: ControlKey) -> Result<NexarMessage> {
        self.control
            .recv(&(src, key))
            .await
            .ok_or(NexarError::PeerDisconnected { rank: src })
    }

    /// Receive relayed tagged data from `src` on `tag`.
    pub async fn recv_tagged(&self, src: Rank, tag: u64) -> Result<PooledBuf> {
        self.tagged
            .recv(&(src, tag))
            .await
            .ok_or(NexarError::PeerDisconnected { rank: src })
    }

    /// Receive the payload of a relayed point-to-point `Data` message from
    /// `src` on `tag`.
    pub async fn recv_data(&self, src: Rank, tag: u32) -> Result<Vec<u8>> {
        self.data
            .recv(&(src, tag))
            .await
            .ok_or(NexarError::PeerDisconnected { rank: src })
    }

    /// Deliver a relayed framed message: `Data` to its tag, control
    /// messages to their key. Waits while that key's queue is full.
    pub async fn deliver_control(&self, src: Rank, msg: NexarMessage) {
        if let NexarMessage::Data { tag, payload, .. } = msg {
            let _ = self.data.send((src, tag), payload).await;
            return;
        }
        match ControlKey::of(&msg) {
            Some(key) => {
                let _ = self.control.send((src, key), msg).await;
            }
            None => tracing::warn!(src, "relay: no receiver for {msg:?}, dropping"),
        }
    }

    /// Deliver relayed tagged data. Waits while that tag's queue is full.
    pub async fn deliver_tagged(&self, src: Rank, tag: u64, data: PooledBuf) {
        let _ = self.tagged.send((src, tag), data).await;
    }
}

/// The routing context of one rank: its rank, its direct peers, and its routes.
///
/// Every relay send and forward needs all of them, so they travel together.
#[derive(Clone, Copy)]
pub struct RelayRoute<'a> {
    /// This rank.
    pub my_rank: Rank,
    /// Direct peers of this rank.
    pub peers: &'a HashMap<Rank, Arc<PeerConnection>>,
    /// Next hop of every destination that is not a direct peer.
    pub routing_table: &'a RoutingTable,
    /// The topology, to pick an alternative hop.
    pub strategy: &'a TopologyStrategy,
    /// Number of ranks in the cluster.
    pub world_size: u32,
}

impl RelayRoute<'_> {
    /// The next hop toward `dest`, a rank that is not a direct peer.
    fn next_hop(self, dest: Rank) -> Result<Rank> {
        self.routing_table
            .next_hop
            .get(&dest)
            .copied()
            .ok_or(NexarError::UnknownPeer { rank: dest })
    }

    /// Send `relay` to `hop`. On error, retry once on `hop`, then try an
    /// alternative hop toward `dest`.
    ///
    /// Returns the alternative hop that took the message, if one did. Returns
    /// the first error when every attempt fails.
    async fn send_with_failover(
        self,
        hop: Rank,
        dest: Rank,
        relay: &NexarMessage,
        priority: Priority,
    ) -> Result<Option<Rank>> {
        let Err(first_err) = try_send_relay(self.peers, hop, relay, priority).await else {
            return Ok(None);
        };
        // Retry once on the same hop (transient QUIC errors).
        if try_send_relay(self.peers, hop, relay, priority)
            .await
            .is_ok()
        {
            return Ok(None);
        }
        if let Some(alt) =
            find_alternative_hop(self.strategy, self.my_rank, dest, hop, self.world_size)
            && try_send_relay(self.peers, alt, relay, priority)
                .await
                .is_ok()
        {
            return Ok(Some(alt));
        }
        Err(first_err)
    }
}

/// The owned form of [`RelayRoute`], for relay listener tasks.
#[derive(Clone)]
pub struct RelayTopology {
    /// This rank.
    pub my_rank: Rank,
    /// Direct peers of this rank.
    pub peers: Arc<HashMap<Rank, Arc<PeerConnection>>>,
    /// Next hop of every destination that is not a direct peer.
    pub routing_table: Arc<RoutingTable>,
    /// The topology, to pick an alternative hop.
    pub strategy: TopologyStrategy,
    /// Number of ranks in the cluster.
    pub world_size: u32,
}

impl RelayTopology {
    /// Borrow this topology as a [`RelayRoute`].
    pub fn route(&self) -> RelayRoute<'_> {
        RelayRoute {
            my_rank: self.my_rank,
            peers: &self.peers,
            routing_table: &self.routing_table,
            strategy: &self.strategy,
            world_size: self.world_size,
        }
    }
}

/// Send a message to a destination rank, using relay if not a direct neighbor.
///
/// If `dest` is a direct peer, sends normally. Otherwise wraps in a Relay message
/// and sends to the next hop. On failure, retries once on the same hop, then
/// attempts an alternative hop before returning an error.
pub async fn send_or_relay_message(
    route: RelayRoute<'_>,
    dest: Rank,
    msg: &NexarMessage,
    priority: Priority,
) -> Result<()> {
    if let Some(peer) = route.peers.get(&dest) {
        return peer.send_message(msg, priority).await;
    }
    let next = route.next_hop(dest)?;
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(msg)
        .map_err(|e| NexarError::EncodeFailed(e.to_string()))?;
    let relay = NexarMessage::Relay {
        src_rank: route.my_rank,
        final_dest: dest,
        tag: 0,
        payload: payload.to_vec(),
    };
    route
        .send_with_failover(next, dest, &relay, priority)
        .await
        .map(|_| ())
}

/// Send tagged data to a destination, using relay if not a direct neighbor.
///
/// On failure, retries once on the same hop, then attempts an alternative hop.
pub async fn send_or_relay_tagged(
    route: RelayRoute<'_>,
    dest: Rank,
    tag: u64,
    data: &[u8],
) -> Result<()> {
    if let Some(peer) = route.peers.get(&dest) {
        return peer.send_raw_tagged_best_effort(tag, data).await;
    }
    let next = route.next_hop(dest)?;
    let relay = NexarMessage::Relay {
        src_rank: route.my_rank,
        final_dest: dest,
        tag,
        payload: data.to_vec(),
    };
    route
        .send_with_failover(next, dest, &relay, Priority::Bulk)
        .await
        .map(|_| ())
}

/// Try to send a relay message to a specific hop.
async fn try_send_relay(
    peers: &HashMap<Rank, Arc<PeerConnection>>,
    hop: Rank,
    msg: &NexarMessage,
    priority: Priority,
) -> Result<()> {
    let peer = peers
        .get(&hop)
        .ok_or(NexarError::UnknownPeer { rank: hop })?;
    peer.send_message(msg, priority).await
}

/// Start relay listener tasks for all neighbor routers.
///
/// For each direct neighbor, spawns a background task that reads Relay messages
/// from that neighbor's relay lane. If the message is for this node, it's delivered
/// locally. Otherwise, it's forwarded to the next hop.
pub fn start_relay_listeners(
    topology: RelayTopology,
    relay_receivers: HashMap<Rank, mpsc::Receiver<NexarMessage>>,
    deliveries: Arc<RelayDeliveries>,
    pool: Arc<crate::transport::buffer_pool::BufferPool>,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::new();
    // Deliveries wait for room on tasks of their own, so one full key does
    // not hold up the other keys from the same neighbor.
    let pending = Arc::new(Semaphore::new(MAX_PENDING_DELIVERIES));
    let my_rank = topology.my_rank;

    for (neighbor_rank, mut relay_rx) in relay_receivers {
        let topology = topology.clone();
        let deliveries = Arc::clone(&deliveries);
        let pool = Arc::clone(&pool);
        let pending = Arc::clone(&pending);

        handles.push(tokio::spawn(async move {
            while let Some(msg) = relay_rx.recv().await {
                if let NexarMessage::Relay {
                    src_rank,
                    final_dest,
                    tag,
                    ref payload,
                } = msg
                {
                    if final_dest == my_rank {
                        let Ok(permit) = Arc::clone(&pending).acquire_owned().await else {
                            return;
                        };
                        let deliveries = Arc::clone(&deliveries);
                        if tag == 0 {
                            match rkyv::from_bytes::<NexarMessage, rkyv::rancor::Error>(payload) {
                                Ok(inner) => {
                                    tokio::spawn(async move {
                                        deliveries.deliver_control(src_rank, inner).await;
                                        drop(permit);
                                    });
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        src_rank,
                                        "relay: failed to deserialize control message: {e}"
                                    );
                                }
                            }
                        } else {
                            let buf = PooledBuf::from_vec(payload.clone(), Arc::clone(&pool));
                            tokio::spawn(async move {
                                deliveries.deliver_tagged(src_rank, tag, buf).await;
                                drop(permit);
                            });
                        }
                    } else {
                        relay_forward(topology.route(), neighbor_rank, final_dest, &msg).await;
                    }
                }
            }
        }));
    }

    handles
}

/// Forward a relay message to the next hop with retry + alternative routing.
async fn relay_forward(route: RelayRoute<'_>, from: Rank, final_dest: Rank, msg: &NexarMessage) {
    let Some(hop) = route.routing_table.route(final_dest) else {
        tracing::error!(dest = final_dest, "relay: no route to destination");
        return;
    };
    match route
        .send_with_failover(hop, final_dest, msg, Priority::Bulk)
        .await
    {
        Ok(None) => {}
        Ok(Some(alt)) => tracing::info!(
            from,
            failed_hop = hop,
            alt_hop = alt,
            dest = final_dest,
            "relay: forwarded via alternative hop"
        ),
        Err(e) => tracing::error!(
            from,
            via = hop,
            dest = final_dest,
            "relay: forward failed after retry and alternative hop: {e}"
        ),
    }
}
