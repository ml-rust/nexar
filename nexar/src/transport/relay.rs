use crate::cluster::sparse::{RoutingTable, TopologyStrategy, find_alternative_hop};
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::transport::PeerConnection;
use crate::transport::buffer_pool::PooledBuf;
use crate::types::{Priority, Rank};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

/// Channel capacity for relay delivery.
const RELAY_CHANNEL_CAPACITY: usize = 256;

/// Manages relay delivery channels for messages arriving via intermediate hops.
///
/// When a node receives a relayed message destined for itself, the relay listener
/// deposits it here. Consumers (collectives, barrier, point-to-point) read from
/// these channels as if the message came directly from the source.
pub struct RelayDeliveries {
    /// Control messages delivered via relay: src_rank -> (sender, cached_receiver).
    control: Mutex<HashMap<Rank, ControlEntry>>,
    /// Tagged data delivered via relay: (src_rank, tag) -> (sender, cached_receiver).
    tagged: Mutex<HashMap<(Rank, u64), TaggedEntry>>,
}

struct ControlEntry {
    tx: mpsc::Sender<NexarMessage>,
    rx: Arc<Mutex<mpsc::Receiver<NexarMessage>>>,
}

struct TaggedEntry {
    tx: mpsc::Sender<PooledBuf>,
    rx: Arc<Mutex<mpsc::Receiver<PooledBuf>>>,
}

impl Default for RelayDeliveries {
    fn default() -> Self {
        Self {
            control: Mutex::new(HashMap::new()),
            tagged: Mutex::new(HashMap::new()),
        }
    }
}

impl RelayDeliveries {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a control delivery sender for a source rank.
    async fn control_tx(&self, src: Rank) -> mpsc::Sender<NexarMessage> {
        let mut map = self.control.lock().await;
        map.entry(src)
            .or_insert_with(|| {
                let (tx, rx) = mpsc::channel(RELAY_CHANNEL_CAPACITY);
                ControlEntry {
                    tx,
                    rx: Arc::new(Mutex::new(rx)),
                }
            })
            .tx
            .clone()
    }

    /// Receive a control message delivered via relay from a source rank.
    ///
    /// The channel is persistent: multiple calls for the same source rank
    /// will receive successive messages from the same channel.
    pub async fn recv_control(&self, src: Rank) -> Result<NexarMessage> {
        let rx = {
            let mut map = self.control.lock().await;
            let entry = map.entry(src).or_insert_with(|| {
                let (tx, rx) = mpsc::channel(RELAY_CHANNEL_CAPACITY);
                ControlEntry {
                    tx,
                    rx: Arc::new(Mutex::new(rx)),
                }
            });
            Arc::clone(&entry.rx)
        };
        rx.lock()
            .await
            .recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank: src })
    }

    /// Get or create a tagged delivery sender for a (src_rank, tag) pair.
    async fn tagged_tx(&self, src: Rank, tag: u64) -> mpsc::Sender<PooledBuf> {
        let mut map = self.tagged.lock().await;
        map.entry((src, tag))
            .or_insert_with(|| {
                let (tx, rx) = mpsc::channel(RELAY_CHANNEL_CAPACITY);
                TaggedEntry {
                    tx,
                    rx: Arc::new(Mutex::new(rx)),
                }
            })
            .tx
            .clone()
    }

    /// Receive tagged data delivered via relay.
    ///
    /// The channel is persistent: multiple calls for the same (src, tag) pair
    /// will receive successive messages from the same channel.
    pub async fn recv_tagged(&self, src: Rank, tag: u64) -> Result<PooledBuf> {
        let rx = {
            let mut map = self.tagged.lock().await;
            let entry = map.entry((src, tag)).or_insert_with(|| {
                let (tx, rx) = mpsc::channel(RELAY_CHANNEL_CAPACITY);
                TaggedEntry {
                    tx,
                    rx: Arc::new(Mutex::new(rx)),
                }
            });
            Arc::clone(&entry.rx)
        };
        rx.lock()
            .await
            .recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank: src })
    }

    /// Deliver a control message that arrived via relay.
    pub async fn deliver_control(&self, src: Rank, msg: NexarMessage) {
        let tx = self.control_tx(src).await;
        let _ = tx.send(msg).await;
    }

    /// Deliver tagged data that arrived via relay.
    pub async fn deliver_tagged(&self, src: Rank, tag: u64, data: PooledBuf) {
        let tx = self.tagged_tx(src, tag).await;
        let _ = tx.send(data).await;
    }
}

/// Send a message to a destination rank, using relay if not a direct neighbor.
///
/// If `dest` is a direct peer, sends normally. Otherwise wraps in a Relay message
/// and sends to the next hop. On failure, retries once on the same hop, then
/// attempts an alternative hop before returning an error.
#[allow(clippy::too_many_arguments)]
pub async fn send_or_relay_message(
    my_rank: Rank,
    peers: &HashMap<Rank, Arc<PeerConnection>>,
    routing_table: &RoutingTable,
    strategy: &TopologyStrategy,
    world_size: u32,
    dest: Rank,
    msg: &NexarMessage,
    priority: Priority,
) -> Result<()> {
    if let Some(peer) = peers.get(&dest) {
        peer.send_message(msg, priority).await
    } else {
        let &next = routing_table
            .next_hop
            .get(&dest)
            .ok_or(NexarError::UnknownPeer { rank: dest })?;
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(msg)
            .map_err(|e| NexarError::EncodeFailed(e.to_string()))?;
        let relay = NexarMessage::Relay {
            src_rank: my_rank,
            final_dest: dest,
            tag: 0,
            payload: payload.to_vec(),
        };

        if let Err(first_err) = try_send_relay(peers, next, &relay, priority).await {
            // Retry once on the same hop (transient QUIC errors).
            if try_send_relay(peers, next, &relay, priority).await.is_ok() {
                return Ok(());
            }
            // Try alternative hop.
            if let Some(alt) = find_alternative_hop(strategy, my_rank, dest, next, world_size)
                && try_send_relay(peers, alt, &relay, priority).await.is_ok()
            {
                return Ok(());
            }
            return Err(first_err);
        }
        Ok(())
    }
}

/// Send tagged data to a destination, using relay if not a direct neighbor.
///
/// On failure, retries once on the same hop, then attempts an alternative hop.
#[allow(clippy::too_many_arguments)]
pub async fn send_or_relay_tagged(
    my_rank: Rank,
    peers: &HashMap<Rank, Arc<PeerConnection>>,
    routing_table: &RoutingTable,
    strategy: &TopologyStrategy,
    world_size: u32,
    dest: Rank,
    tag: u64,
    data: &[u8],
) -> Result<()> {
    if let Some(peer) = peers.get(&dest) {
        peer.send_raw_tagged_best_effort(tag, data).await
    } else {
        let &next = routing_table
            .next_hop
            .get(&dest)
            .ok_or(NexarError::UnknownPeer { rank: dest })?;
        let relay = NexarMessage::Relay {
            src_rank: my_rank,
            final_dest: dest,
            tag,
            payload: data.to_vec(),
        };

        if let Err(first_err) = try_send_relay(peers, next, &relay, Priority::Bulk).await {
            if try_send_relay(peers, next, &relay, Priority::Bulk)
                .await
                .is_ok()
            {
                return Ok(());
            }
            if let Some(alt) = find_alternative_hop(strategy, my_rank, dest, next, world_size)
                && try_send_relay(peers, alt, &relay, Priority::Bulk)
                    .await
                    .is_ok()
            {
                return Ok(());
            }
            return Err(first_err);
        }
        Ok(())
    }
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
#[allow(clippy::too_many_arguments)]
pub fn start_relay_listeners(
    my_rank: Rank,
    peers: Arc<HashMap<Rank, Arc<PeerConnection>>>,
    routing_table: Arc<RoutingTable>,
    strategy: TopologyStrategy,
    world_size: u32,
    relay_receivers: HashMap<Rank, mpsc::Receiver<NexarMessage>>,
    deliveries: Arc<RelayDeliveries>,
    pool: Arc<crate::transport::buffer_pool::BufferPool>,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::new();

    for (neighbor_rank, mut relay_rx) in relay_receivers {
        let peers = Arc::clone(&peers);
        let rt = Arc::clone(&routing_table);
        let deliveries = Arc::clone(&deliveries);
        let pool = Arc::clone(&pool);
        let strat = strategy.clone();

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
                        if tag == 0 {
                            match rkyv::from_bytes::<NexarMessage, rkyv::rancor::Error>(payload) {
                                Ok(inner) => {
                                    deliveries.deliver_control(src_rank, inner).await;
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
                            deliveries.deliver_tagged(src_rank, tag, buf).await;
                        }
                    } else {
                        relay_forward(
                            my_rank,
                            neighbor_rank,
                            final_dest,
                            &msg,
                            &peers,
                            &rt,
                            &strat,
                            world_size,
                        )
                        .await;
                    }
                }
            }
        }));
    }

    handles
}

/// Forward a relay message to the next hop with retry + alternative routing.
#[allow(clippy::too_many_arguments)]
async fn relay_forward(
    my_rank: Rank,
    from: Rank,
    final_dest: Rank,
    msg: &NexarMessage,
    peers: &HashMap<Rank, Arc<PeerConnection>>,
    rt: &RoutingTable,
    strategy: &TopologyStrategy,
    world_size: u32,
) {
    let Some(hop) = rt.route(final_dest) else {
        tracing::error!(dest = final_dest, "relay: no route to destination");
        return;
    };

    // First attempt.
    if try_send_relay(peers, hop, msg, Priority::Bulk)
        .await
        .is_ok()
    {
        return;
    }

    // Retry once on the same hop (transient error).
    if try_send_relay(peers, hop, msg, Priority::Bulk)
        .await
        .is_ok()
    {
        return;
    }

    // Try alternative hop.
    if let Some(alt) = find_alternative_hop(strategy, my_rank, final_dest, hop, world_size)
        && try_send_relay(peers, alt, msg, Priority::Bulk)
            .await
            .is_ok()
    {
        tracing::info!(
            from,
            failed_hop = hop,
            alt_hop = alt,
            dest = final_dest,
            "relay: forwarded via alternative hop"
        );
        return;
    }

    tracing::error!(
        from,
        via = hop,
        dest = final_dest,
        "relay: forward failed after retry and alternative hop"
    );
}
