use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::decode_message;
use crate::transport::buffer_pool::{BufferPool, PooledBuf};
use crate::transport::connection::{
    STREAM_TAG_FRAMED, STREAM_TAG_RAW, STREAM_TAG_RAW_COMM, STREAM_TAG_RAW_TAGGED,
};
use crate::types::Rank;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore, mpsc, oneshot};

/// Channel capacity per message lane.
const LANE_CAPACITY: usize = 256;

/// Maximum number of concurrent in-flight stream handler tasks per peer.
const MAX_CONCURRENT_STREAMS: usize = 512;

/// Maximum framed message size accepted by the router (4 GiB).
const MAX_MESSAGE_SIZE: u64 = 4 * 1024 * 1024 * 1024;

/// A demultiplexer that runs a single receive loop on a QUIC connection and
/// routes incoming streams to typed channels by stream type tag and message variant.
///
/// # Lanes
///
/// - **`rpc_requests`** (`Rpc` variants)
/// - **`rpc_responses`** — keyed by `req_id`
/// - **`control`** (Barrier, BarrierAck, Heartbeat, etc.)
/// - **`data`** (`Data` variants)
/// - **`raw`** — raw byte streams (default comm_id 0)
/// - **`raw_comms`** — per-comm_id raw byte streams (for split communicators)
pub struct PeerRouter {
    pub rpc_requests: Mutex<mpsc::Receiver<NexarMessage>>,
    rpc_waiters: Arc<Mutex<HashMap<u64, oneshot::Sender<NexarMessage>>>>,
    pub control: Mutex<mpsc::Receiver<NexarMessage>>,
    pub data: Mutex<mpsc::Receiver<NexarMessage>>,
    pub raw: Mutex<mpsc::Receiver<PooledBuf>>,
    /// Per-comm_id raw channels for split communicators.
    raw_comms: Arc<Mutex<HashMap<u64, CommChannel>>>,
    /// Per-tag raw channels for concurrent collectives.
    /// Channels are lazily created when tagged data arrives or when
    /// `register_tag` is called, whichever comes first.
    tagged: Arc<Mutex<HashMap<u64, TaggedChannel>>>,
}

/// A communicator channel. Lazily created when either a message arrives or
/// `register_comm` is called. Both sides (router sender and client receiver)
/// get the same underlying channel.
struct CommChannel {
    tx: mpsc::Sender<PooledBuf>,
    /// The receiver, stored here until claimed by `register_comm`.
    /// Once claimed, this is `None`.
    rx: Option<mpsc::Receiver<PooledBuf>>,
}

/// A tagged channel. Lazily created when either a message arrives or
/// `register_tag` is called. Both sides (router sender and client receiver)
/// get the same underlying channel.
struct TaggedChannel {
    tx: mpsc::Sender<PooledBuf>,
    /// The receiver, stored here until claimed by `register_tag`.
    /// Once claimed, this is `None`.
    rx: Option<mpsc::Receiver<PooledBuf>>,
}

/// Senders held by the background receive loop. Cloned into per-stream tasks.
#[derive(Clone)]
struct RouterSenders {
    rank: Rank,
    rpc_requests: mpsc::Sender<NexarMessage>,
    rpc_waiters: Arc<Mutex<HashMap<u64, oneshot::Sender<NexarMessage>>>>,
    control: mpsc::Sender<NexarMessage>,
    data: mpsc::Sender<NexarMessage>,
    raw: mpsc::Sender<PooledBuf>,
    raw_comms: Arc<Mutex<HashMap<u64, CommChannel>>>,
    tagged: Arc<Mutex<HashMap<u64, TaggedChannel>>>,
    pool: Arc<BufferPool>,
}

impl PeerRouter {
    /// Spawn a background receive loop for `conn` and return the router.
    pub fn spawn(
        rank: Rank,
        conn: quinn::Connection,
        pool: Arc<BufferPool>,
    ) -> (Self, tokio::task::JoinHandle<Result<()>>) {
        let (rpc_req_tx, rpc_req_rx) = mpsc::channel(LANE_CAPACITY);
        let (ctrl_tx, ctrl_rx) = mpsc::channel(LANE_CAPACITY);
        let (data_tx, data_rx) = mpsc::channel(LANE_CAPACITY);
        let (raw_tx, raw_rx) = mpsc::channel(LANE_CAPACITY);

        let rpc_waiters: Arc<Mutex<HashMap<u64, oneshot::Sender<NexarMessage>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let raw_comms: Arc<Mutex<HashMap<u64, CommChannel>>> = Arc::new(Mutex::new(HashMap::new()));
        let tagged: Arc<Mutex<HashMap<u64, TaggedChannel>>> = Arc::new(Mutex::new(HashMap::new()));

        let senders = RouterSenders {
            rank,
            rpc_requests: rpc_req_tx,
            rpc_waiters: Arc::clone(&rpc_waiters),
            control: ctrl_tx,
            data: data_tx,
            raw: raw_tx,
            raw_comms: Arc::clone(&raw_comms),
            tagged: Arc::clone(&tagged),
            pool,
        };

        let handle = tokio::spawn(accept_loop(conn, senders));

        let router = Self {
            rpc_requests: Mutex::new(rpc_req_rx),
            rpc_waiters,
            control: Mutex::new(ctrl_rx),
            data: Mutex::new(data_rx),
            raw: Mutex::new(raw_rx),
            raw_comms,
            tagged,
        };

        (router, handle)
    }

    /// Register a oneshot waiter for a specific RPC `req_id`.
    pub async fn register_rpc_waiter(&self, req_id: u64) -> oneshot::Receiver<NexarMessage> {
        let (tx, rx) = oneshot::channel();
        self.rpc_waiters.lock().await.insert(req_id, tx);
        rx
    }

    /// Remove a previously registered waiter.
    pub async fn remove_rpc_waiter(&self, req_id: u64) {
        self.rpc_waiters.lock().await.remove(&req_id);
    }

    /// Register a communicator channel and return the receiver.
    /// Called during `split()` to set up per-comm_id routing.
    ///
    /// If messages for this `comm_id` arrived before registration (lazy creation),
    /// the existing channel's receiver is returned so those messages aren't lost.
    pub async fn register_comm(&self, comm_id: u64) -> mpsc::Receiver<PooledBuf> {
        let mut comms = self.raw_comms.lock().await;
        if let Some(ch) = comms.get_mut(&comm_id)
            && let Some(rx) = ch.rx.take()
        {
            return rx;
        }
        let (tx, rx) = mpsc::channel(LANE_CAPACITY);
        comms.insert(comm_id, CommChannel { tx, rx: None });
        rx
    }

    /// Receive the next message from the control lane.
    pub async fn recv_control(&self, rank: Rank) -> Result<NexarMessage> {
        self.control
            .lock()
            .await
            .recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank })
    }

    /// Receive the next RPC request from the rpc_requests lane.
    pub async fn recv_rpc_request(&self, rank: Rank) -> Result<NexarMessage> {
        self.rpc_requests
            .lock()
            .await
            .recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank })
    }

    /// Receive the next data message from the data lane.
    pub async fn recv_data(&self, rank: Rank) -> Result<NexarMessage> {
        self.data
            .lock()
            .await
            .recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank })
    }

    /// Register a tagged channel and return the receiver.
    ///
    /// If the channel was already created (by a message arriving first),
    /// the existing receiver is returned. Otherwise, a new channel is created.
    pub async fn register_tag(&self, tag: u64) -> mpsc::Receiver<PooledBuf> {
        let mut tags = self.tagged.lock().await;
        if let Some(ch) = tags.get_mut(&tag) {
            // Channel exists (created by router when a message arrived).
            // Take the receiver if we haven't already.
            if let Some(rx) = ch.rx.take() {
                return rx;
            }
            // Receiver already claimed — create a fresh channel.
            // This shouldn't normally happen.
        }
        // Create a new channel.
        let (tx, rx) = mpsc::channel(LANE_CAPACITY);
        tags.insert(tag, TaggedChannel { tx, rx: None });
        rx
    }

    /// Remove a previously registered tag channel.
    pub async fn remove_tag(&self, tag: u64) {
        self.tagged.lock().await.remove(&tag);
    }

    /// Receive raw bytes from the raw lane (default comm_id 0).
    pub async fn recv_raw(&self, rank: Rank) -> Result<PooledBuf> {
        self.raw
            .lock()
            .await
            .recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank })
    }
}

/// The accept loop: accepts incoming QUIC uni streams and spawns a task per stream.
async fn accept_loop(conn: quinn::Connection, tx: RouterSenders) -> Result<()> {
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_STREAMS));

    loop {
        let stream = match conn.accept_uni().await {
            Ok(s) => s,
            Err(_) => {
                tx.rpc_waiters.lock().await.clear();
                return Ok(());
            }
        };

        let Ok(permit) = Arc::clone(&semaphore).acquire_owned().await else {
            return Ok(()); // Semaphore closed, exit gracefully
        };

        let tx = tx.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_stream(stream, &tx).await {
                tracing::error!(
                    rank = tx.rank,
                    "router: local receiver dropped, messages will be lost: {e}"
                );
            }
            drop(permit);
        });
    }
}

/// Read a single stream (tag + payload) and dispatch to the correct lane.
///
/// Returns `Err` if a channel receiver has been dropped, indicating the local
/// consumer is gone and further messages for this peer will be lost.
async fn handle_stream(mut stream: quinn::RecvStream, tx: &RouterSenders) -> Result<()> {
    let mut tag_buf = [0u8; 1];
    if stream.read_exact(&mut tag_buf).await.is_err() {
        tracing::warn!(
            rank = tx.rank,
            "router: failed to read stream tag, skipping stream"
        );
        return Ok(());
    }

    match tag_buf[0] {
        STREAM_TAG_FRAMED => {
            let msg = match read_framed(&mut stream, tx.rank, &tx.pool).await {
                Some(m) => m,
                None => return Ok(()),
            };
            dispatch_framed(msg, tx).await?;
        }
        STREAM_TAG_RAW => {
            let buf = match read_raw(&mut stream, tx.rank, &tx.pool).await {
                Some(b) => b,
                None => return Ok(()),
            };
            if tx.raw.send(buf).await.is_err() {
                return Err(NexarError::PeerDisconnected { rank: tx.rank });
            }
        }
        STREAM_TAG_RAW_TAGGED => {
            // Read 8-byte tag, then length-prefixed payload.
            let mut tag_bytes = [0u8; 8];
            if stream.read_exact(&mut tag_bytes).await.is_err() {
                tracing::warn!(rank = tx.rank, "router: failed to read tagged tag");
                return Ok(());
            }
            let tag = u64::from_le_bytes(tag_bytes);

            let buf = match read_raw(&mut stream, tx.rank, &tx.pool).await {
                Some(b) => b,
                None => return Ok(()),
            };

            let mut tags = tx.tagged.lock().await;
            // Get or create the channel for this tag.
            let ch = tags.entry(tag).or_insert_with(|| {
                let (tx, rx) = mpsc::channel(LANE_CAPACITY);
                TaggedChannel { tx, rx: Some(rx) }
            });
            if ch.tx.send(buf).await.is_err() {
                return Err(NexarError::PeerDisconnected { rank: tx.rank });
            }
        }
        STREAM_TAG_RAW_COMM => {
            // Read 8-byte comm_id, then length-prefixed payload.
            let mut comm_id_buf = [0u8; 8];
            if stream.read_exact(&mut comm_id_buf).await.is_err() {
                tracing::warn!(rank = tx.rank, "router: failed to read comm_id");
                return Ok(());
            }
            let comm_id = u64::from_le_bytes(comm_id_buf);

            let buf = match read_raw(&mut stream, tx.rank, &tx.pool).await {
                Some(b) => b,
                None => return Ok(()),
            };

            let mut comms = tx.raw_comms.lock().await;
            let ch = comms.entry(comm_id).or_insert_with(|| {
                let (tx, rx) = mpsc::channel(LANE_CAPACITY);
                CommChannel { tx, rx: Some(rx) }
            });
            if ch.tx.send(buf).await.is_err() {
                return Err(NexarError::PeerDisconnected { rank: tx.rank });
            }
        }
        other => {
            tracing::warn!(
                rank = tx.rank,
                "router: unknown stream tag 0x{:02x}, skipping stream",
                other
            );
        }
    }
    Ok(())
}

/// Route a decoded framed message to the correct lane.
///
/// Returns `Err` if the target channel's receiver has been dropped.
async fn dispatch_framed(msg: NexarMessage, tx: &RouterSenders) -> Result<()> {
    match msg {
        NexarMessage::Rpc { .. } => {
            if tx.rpc_requests.send(msg).await.is_err() {
                return Err(NexarError::PeerDisconnected { rank: tx.rank });
            }
        }
        NexarMessage::RpcResponse { req_id, .. } => {
            let mut waiters = tx.rpc_waiters.lock().await;
            if let Some(waiter) = waiters.remove(&req_id) {
                let _ = waiter.send(msg);
            } else {
                tracing::warn!(
                    rank = tx.rank,
                    req_id,
                    "router: RpcResponse with no registered waiter, discarding"
                );
            }
        }
        NexarMessage::Barrier { .. }
        | NexarMessage::BarrierAck { .. }
        | NexarMessage::Heartbeat { .. }
        | NexarMessage::NodeJoined { .. }
        | NexarMessage::NodeLeft { .. }
        | NexarMessage::Hello { .. }
        | NexarMessage::Welcome { .. }
        | NexarMessage::RdmaEndpoint { .. }
        | NexarMessage::SplitRequest { .. }
        | NexarMessage::RecoveryVote { .. }
        | NexarMessage::RecoveryAgreement { .. }
        | NexarMessage::ElasticCheckpoint { .. }
        | NexarMessage::ElasticCheckpointAck { .. } => {
            if tx.control.send(msg).await.is_err() {
                return Err(NexarError::PeerDisconnected { rank: tx.rank });
            }
        }
        NexarMessage::Data { .. } => {
            if tx.data.send(msg).await.is_err() {
                return Err(NexarError::PeerDisconnected { rank: tx.rank });
            }
        }
    }
    Ok(())
}

/// Read a length-prefixed payload from a stream into a pooled buffer.
async fn read_length_prefixed(
    stream: &mut quinn::RecvStream,
    rank: Rank,
    pool: &Arc<BufferPool>,
    label: &str,
) -> Option<PooledBuf> {
    let mut len_buf = [0u8; 8];
    if let Err(e) = stream.read_exact(&mut len_buf).await {
        tracing::warn!(rank, "router: {label} length read failed: {e}");
        return None;
    }
    let len = u64::from_le_bytes(len_buf);
    if len > MAX_MESSAGE_SIZE {
        tracing::warn!(
            rank,
            "router: {label} message too large ({len} bytes), skipping"
        );
        return None;
    }
    let mut buf = pool.checkout(len as usize);
    if let Err(e) = stream.read_exact(&mut buf).await {
        tracing::warn!(rank, "router: {label} payload read failed: {e}");
        return None;
    }
    Some(buf)
}

/// Read a framed message from a stream (after the tag byte has been consumed).
async fn read_framed(
    stream: &mut quinn::RecvStream,
    rank: Rank,
    pool: &Arc<BufferPool>,
) -> Option<NexarMessage> {
    let buf = read_length_prefixed(stream, rank, pool, "framed").await?;
    match decode_message(&buf) {
        Ok((_, msg)) => Some(msg),
        Err(e) => {
            tracing::warn!(rank, "router: framed decode failed: {e}");
            None
        }
    }
}

/// Read raw bytes from a stream (after the tag byte has been consumed).
async fn read_raw(
    stream: &mut quinn::RecvStream,
    rank: Rank,
    pool: &Arc<BufferPool>,
) -> Option<PooledBuf> {
    read_length_prefixed(stream, rank, pool, "raw").await
}
