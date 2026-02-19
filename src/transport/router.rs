use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::decode_message;
use crate::transport::connection::{STREAM_TAG_FRAMED, STREAM_TAG_RAW};
use crate::types::Rank;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore, mpsc, oneshot};

/// Channel capacity per message lane.
const LANE_CAPACITY: usize = 256;

/// Maximum number of concurrent in-flight stream handler tasks per peer.
/// Prevents a noisy/malicious peer from exhausting runtime resources by
/// opening streams faster than they can be processed.
const MAX_CONCURRENT_STREAMS: usize = 512;

/// Maximum framed message size accepted by the router (4 GiB).
const MAX_MESSAGE_SIZE: u64 = 4 * 1024 * 1024 * 1024;

/// A demultiplexer that runs a single receive loop on a QUIC connection and
/// routes incoming streams to typed channels by stream type tag and message variant.
///
/// Without this, multiple consumers (RPC serve loop, barrier, collectives)
/// calling `recv_message` or `recv_raw` on the same peer would race for QUIC
/// streams, stealing messages meant for other subsystems.
///
/// Every QUIC uni stream opened by the remote peer begins with a 1-byte tag:
/// - `0x01` (STREAM_TAG_FRAMED): a framed `NexarMessage`; routed by variant.
/// - `0x02` (STREAM_TAG_RAW): raw bulk bytes; routed to the `raw` channel.
///
/// # Lanes
///
/// Each lane's receiver is wrapped in its own `Mutex` so that concurrent
/// consumers can wait on different lanes for the same peer without blocking
/// each other.
///
/// - **`rpc_requests`** (`Rpc` variants) — consumed by the RPC dispatcher serve loop.
/// - **`rpc_responses`** — keyed by `req_id`; each `rpc()` call registers a
///   oneshot channel for its specific request, so concurrent RPCs to the same
///   peer never steal each other's responses.
/// - **`control`** (Barrier, BarrierAck, Heartbeat, NodeJoined, NodeLeft, Hello, Welcome) —
///   consumed by barrier logic and health monitors.
/// - **`data`** (`Data` variants) — consumed by point-to-point `recv`.
/// - **`raw`** — raw byte streams (bulk tensor transfers).
///
/// # No Cross-Lane Head-of-Line Blocking
///
/// Each accepted QUIC stream is read and dispatched in its own spawned task.
/// If one lane's channel is full, that task's `.send().await` blocks only that
/// task — the main `accept_uni` loop and all other lanes remain unblocked.
/// Messages are never dropped due to backpressure.
pub struct PeerRouter {
    pub rpc_requests: Mutex<mpsc::Receiver<NexarMessage>>,
    rpc_waiters: Arc<Mutex<HashMap<u64, oneshot::Sender<NexarMessage>>>>,
    pub control: Mutex<mpsc::Receiver<NexarMessage>>,
    pub data: Mutex<mpsc::Receiver<NexarMessage>>,
    pub raw: Mutex<mpsc::Receiver<Vec<u8>>>,
}

/// Senders held by the background receive loop. Cloned into per-stream tasks.
#[derive(Clone)]
struct RouterSenders {
    rank: Rank,
    rpc_requests: mpsc::Sender<NexarMessage>,
    rpc_waiters: Arc<Mutex<HashMap<u64, oneshot::Sender<NexarMessage>>>>,
    control: mpsc::Sender<NexarMessage>,
    data: mpsc::Sender<NexarMessage>,
    raw: mpsc::Sender<Vec<u8>>,
}

impl PeerRouter {
    /// Spawn a background receive loop for `conn` and return the router.
    ///
    /// Takes the raw `quinn::Connection` directly to avoid lifetime coupling
    /// with `PeerConnection`. Only the receive side of the connection is used here;
    /// the `PeerConnection` (wrapping the same `quinn::Connection` clone) handles
    /// sending independently.
    ///
    /// The returned `JoinHandle` resolves when the connection closes or an
    /// unrecoverable transport error occurs. Callers should monitor it to
    /// detect peer disconnection.
    pub fn spawn(
        rank: Rank,
        conn: quinn::Connection,
    ) -> (Self, tokio::task::JoinHandle<Result<()>>) {
        let (rpc_req_tx, rpc_req_rx) = mpsc::channel(LANE_CAPACITY);
        let (ctrl_tx, ctrl_rx) = mpsc::channel(LANE_CAPACITY);
        let (data_tx, data_rx) = mpsc::channel(LANE_CAPACITY);
        let (raw_tx, raw_rx) = mpsc::channel(LANE_CAPACITY);

        let rpc_waiters: Arc<Mutex<HashMap<u64, oneshot::Sender<NexarMessage>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let senders = RouterSenders {
            rank,
            rpc_requests: rpc_req_tx,
            rpc_waiters: Arc::clone(&rpc_waiters),
            control: ctrl_tx,
            data: data_tx,
            raw: raw_tx,
        };

        let handle = tokio::spawn(accept_loop(conn, senders));

        let router = Self {
            rpc_requests: Mutex::new(rpc_req_rx),
            rpc_waiters,
            control: Mutex::new(ctrl_rx),
            data: Mutex::new(data_rx),
            raw: Mutex::new(raw_rx),
        };

        (router, handle)
    }

    /// Register a oneshot waiter for a specific RPC `req_id` and return the
    /// receiver. When the router receives an `RpcResponse` with this `req_id`,
    /// it delivers it directly to this waiter — no queue contention.
    pub async fn register_rpc_waiter(&self, req_id: u64) -> oneshot::Receiver<NexarMessage> {
        let (tx, rx) = oneshot::channel();
        self.rpc_waiters.lock().await.insert(req_id, tx);
        rx
    }

    /// Remove a previously registered waiter (e.g. on send failure cleanup).
    pub async fn remove_rpc_waiter(&self, req_id: u64) {
        self.rpc_waiters.lock().await.remove(&req_id);
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

    /// Receive raw bytes from the raw lane.
    pub async fn recv_raw(&self, rank: Rank) -> Result<Vec<u8>> {
        self.raw
            .lock()
            .await
            .recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank })
    }
}

/// The accept loop: accepts incoming QUIC uni streams and spawns a task per
/// stream to read and dispatch the message. This ensures that a blocked lane
/// (full channel) does not prevent other streams from being accepted and
/// dispatched — no cross-lane head-of-line blocking, and no message drops.
///
/// A semaphore caps concurrent in-flight tasks to `MAX_CONCURRENT_STREAMS`,
/// preventing a noisy peer from exhausting runtime resources.
///
/// On exit (connection closed), all pending RPC waiters are drained so that
/// `rpc()` callers blocked on `rx.await` receive `PeerDisconnected` instead
/// of hanging forever.
async fn accept_loop(conn: quinn::Connection, tx: RouterSenders) -> Result<()> {
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_STREAMS));

    loop {
        let stream = match conn.accept_uni().await {
            Ok(s) => s,
            Err(_) => {
                // Connection closed — drain all pending RPC waiters so their
                // oneshot::Senders are dropped, unblocking any rpc() callers.
                tx.rpc_waiters.lock().await.clear();
                return Ok(());
            }
        };

        // Acquire a permit before spawning. If all permits are taken, this
        // awaits until a previous stream task completes, providing backpressure.
        let permit = Arc::clone(&semaphore)
            .acquire_owned()
            .await
            .expect("semaphore is never closed");

        let tx = tx.clone();
        tokio::spawn(async move {
            handle_stream(stream, &tx).await;
            drop(permit); // release the concurrency slot
        });
    }
}

/// Read a single stream (tag + payload) and dispatch to the correct lane.
/// Runs in its own task so `.send().await` backpressure is isolated.
async fn handle_stream(mut stream: quinn::RecvStream, tx: &RouterSenders) {
    // Read the 1-byte stream type tag.
    let mut tag_buf = [0u8; 1];
    if stream.read_exact(&mut tag_buf).await.is_err() {
        tracing::warn!(
            rank = tx.rank,
            "router: failed to read stream tag, skipping stream"
        );
        return;
    }

    match tag_buf[0] {
        STREAM_TAG_FRAMED => {
            let msg = match read_framed(&mut stream, tx.rank).await {
                Some(m) => m,
                None => return,
            };
            dispatch_framed(msg, tx).await;
        }
        STREAM_TAG_RAW => {
            let buf = match read_raw(&mut stream, tx.rank).await {
                Some(b) => b,
                None => return,
            };
            // .send().await blocks only this task if the raw lane is full.
            if tx.raw.send(buf).await.is_err() {
                tracing::warn!(rank = tx.rank, "router: raw receiver dropped");
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
}

/// Route a decoded framed message to the correct lane.
/// `.send().await` blocks only this task if the target lane is full.
async fn dispatch_framed(msg: NexarMessage, tx: &RouterSenders) {
    match msg {
        NexarMessage::Rpc { .. } => {
            if tx.rpc_requests.send(msg).await.is_err() {
                tracing::warn!(rank = tx.rank, "router: rpc_requests receiver dropped");
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
        | NexarMessage::Welcome { .. } => {
            if tx.control.send(msg).await.is_err() {
                tracing::warn!(rank = tx.rank, "router: control receiver dropped");
            }
        }
        NexarMessage::Data { .. } => {
            if tx.data.send(msg).await.is_err() {
                tracing::warn!(rank = tx.rank, "router: data receiver dropped");
            }
        }
    }
}

/// Read a framed message from a stream (after the tag byte has been consumed).
/// Returns `None` on any read/decode error (logged and skipped).
async fn read_framed(stream: &mut quinn::RecvStream, rank: Rank) -> Option<NexarMessage> {
    let mut len_buf = [0u8; 8];
    if let Err(e) = stream.read_exact(&mut len_buf).await {
        tracing::warn!(rank, "router: framed length read failed: {e}");
        return None;
    }
    let len = u64::from_le_bytes(len_buf);
    if len > MAX_MESSAGE_SIZE {
        tracing::warn!(
            rank,
            "router: framed message too large ({len} bytes), skipping"
        );
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    if let Err(e) = stream.read_exact(&mut buf).await {
        tracing::warn!(rank, "router: framed payload read failed: {e}");
        return None;
    }
    match decode_message(&buf) {
        Ok((_, msg)) => Some(msg),
        Err(e) => {
            tracing::warn!(rank, "router: framed decode failed: {e}");
            None
        }
    }
}

/// Read raw bytes from a stream (after the tag byte has been consumed).
/// Returns `None` on any read error (logged and skipped).
async fn read_raw(stream: &mut quinn::RecvStream, rank: Rank) -> Option<Vec<u8>> {
    let mut len_buf = [0u8; 8];
    if let Err(e) = stream.read_exact(&mut len_buf).await {
        tracing::warn!(rank, "router: raw length read failed: {e}");
        return None;
    }
    let len = u64::from_le_bytes(len_buf);
    if len > MAX_MESSAGE_SIZE {
        tracing::warn!(
            rank,
            "router: raw message too large ({len} bytes), skipping"
        );
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    if let Err(e) = stream.read_exact(&mut buf).await {
        tracing::warn!(rank, "router: raw payload read failed: {e}");
        return None;
    }
    Some(buf)
}
