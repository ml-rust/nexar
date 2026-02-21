use crate::cluster::sparse::RoutingTable;
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
/// and sends to the next hop.
pub async fn send_or_relay_message(
    my_rank: Rank,
    peers: &HashMap<Rank, Arc<PeerConnection>>,
    routing_table: &RoutingTable,
    dest: Rank,
    msg: &NexarMessage,
    priority: Priority,
) -> Result<()> {
    if let Some(peer) = peers.get(&dest) {
        peer.send_message(msg, priority).await
    } else {
        let next = routing_table
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
        let peer = peers
            .get(next)
            .ok_or(NexarError::UnknownPeer { rank: *next })?;
        peer.send_message(&relay, priority).await
    }
}

/// Send tagged data to a destination, using relay if not a direct neighbor.
pub async fn send_or_relay_tagged(
    my_rank: Rank,
    peers: &HashMap<Rank, Arc<PeerConnection>>,
    routing_table: &RoutingTable,
    dest: Rank,
    tag: u64,
    data: &[u8],
) -> Result<()> {
    if let Some(peer) = peers.get(&dest) {
        peer.send_raw_tagged_best_effort(tag, data).await
    } else {
        let next = routing_table
            .next_hop
            .get(&dest)
            .ok_or(NexarError::UnknownPeer { rank: dest })?;
        let relay = NexarMessage::Relay {
            src_rank: my_rank,
            final_dest: dest,
            tag,
            payload: data.to_vec(),
        };
        let peer = peers
            .get(next)
            .ok_or(NexarError::UnknownPeer { rank: *next })?;
        peer.send_message(&relay, Priority::Bulk).await
    }
}

/// Start relay listener tasks for all neighbor routers.
///
/// For each direct neighbor, spawns a background task that reads Relay messages
/// from that neighbor's relay lane. If the message is for this node, it's delivered
/// locally. Otherwise, it's forwarded to the next hop.
pub fn start_relay_listeners(
    my_rank: Rank,
    peers: Arc<HashMap<Rank, Arc<PeerConnection>>>,
    routing_table: Arc<RoutingTable>,
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
                        let next = rt.route(final_dest);
                        if let Some(hop) = next {
                            if let Some(peer) = peers.get(&hop) {
                                if let Err(e) = peer.send_message(&msg, Priority::Bulk).await {
                                    tracing::warn!(
                                        from = neighbor_rank,
                                        via = hop,
                                        dest = final_dest,
                                        "relay: forward failed: {e}"
                                    );
                                }
                            } else {
                                tracing::warn!(
                                    dest = final_dest,
                                    hop,
                                    "relay: next hop not in peers"
                                );
                            }
                        } else {
                            tracing::warn!(dest = final_dest, "relay: no route to destination");
                        }
                    }
                }
            }
        }));
    }

    handles
}
