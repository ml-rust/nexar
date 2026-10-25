use super::byte_budget::ByteBudget;
use super::lane::{
    ControlKey, LaneKey, MAX_SKIPS, STREAM_TAG_FRAMED, STREAM_TAG_SEQ, STREAM_TAG_SKIP, SeqHeader,
};
use super::mailbox::KeyedMailbox;
use super::seq_inbox::SeqInbox;
use crate::config::{DEFAULT_MAX_FRAME_BYTES, DEFAULT_RECV_INFLIGHT_BYTES, NexarConfig};
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::decode_message;
use crate::transport::buffer_pool::{BufferPool, PooledBuf};
use crate::types::Rank;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore, SemaphorePermit, mpsc, oneshot};
use tokio::task::JoinSet;

/// Default per-key receive queue depth. See [`NexarConfig::recv_queue_depth`](crate::NexarConfig::recv_queue_depth).
pub const DEFAULT_RECV_QUEUE_DEPTH: usize = 256;

/// Maximum number of payloads read at once per peer.
const MAX_CONCURRENT_READS: usize = 512;

/// Stop code of a stream the router refuses to read.
const REFUSED_CODE: u32 = 0x4e58_0011;

/// Receive limits of one peer's router.
#[derive(Debug, Clone, Copy)]
pub struct RouterLimits {
    /// Messages each receive key queues. See [`NexarConfig::recv_queue_depth`].
    pub depth: usize,
    /// Largest message accepted. See [`NexarConfig::max_frame_bytes`].
    pub max_message_bytes: usize,
    /// Payload bytes read at once. See [`NexarConfig::recv_inflight_bytes`].
    pub inflight_bytes: usize,
}

impl Default for RouterLimits {
    fn default() -> Self {
        Self {
            depth: DEFAULT_RECV_QUEUE_DEPTH,
            max_message_bytes: DEFAULT_MAX_FRAME_BYTES,
            inflight_bytes: DEFAULT_RECV_INFLIGHT_BYTES,
        }
    }
}

impl From<&NexarConfig> for RouterLimits {
    fn from(c: &NexarConfig) -> Self {
        Self {
            depth: c.recv_queue_depth,
            max_message_bytes: c.max_frame_bytes,
            inflight_bytes: c.recv_inflight_bytes,
        }
    }
}

type RpcWaiters = Arc<Mutex<HashMap<u64, oneshot::Sender<NexarMessage>>>>;

/// A demultiplexer that runs one receive loop on a QUIC connection and
/// routes each incoming stream to the receiver of its key.
///
/// # Receive keys
///
/// - **Byte lanes** ([`LaneKey`]): raw, tagged, and point-to-point data
///   messages, per communicator. FIFO per lane: messages leave in the order
///   the sending connection numbered them.
/// - **Control** ([`ControlKey`]): barrier, recovery, and elastic messages.
///   No order within a key. Received heartbeats are dropped.
/// - **RPC requests**, and **RPC responses** keyed by `req_id`.
/// - **Relay** messages for sparse topology forwarding.
///
/// A receive for one key never takes or fails on a message of another key.
/// Each key queues at most `depth` messages. A message past that waits
/// unread in its QUIC stream, and QUIC flow control slows its sender. Other
/// keys go on while the peer's open streams stay under the QUIC concurrent
/// stream limit.
///
/// A message longer than [`RouterLimits::max_message_bytes`] is refused: the
/// router stops its stream without allocating, and the sender's write fails.
/// The payload bytes read at once stay under [`RouterLimits::inflight_bytes`].
///
/// When the connection closes, receivers first drain complete messages,
/// then get [`NexarError::PeerDisconnected`].
pub struct PeerRouter {
    lanes: Arc<SeqInbox>,
    control: Arc<KeyedMailbox<ControlKey, NexarMessage>>,
    rpc_requests: Mutex<mpsc::Receiver<NexarMessage>>,
    rpc_waiters: RpcWaiters,
    relay: Mutex<Option<mpsc::Receiver<NexarMessage>>>,
}

/// Everything a stream handler needs. Shared by all handlers of one peer.
struct RouterCtx {
    rank: Rank,
    conn: quinn::Connection,
    pool: Arc<BufferPool>,
    reads: Semaphore,
    budget: ByteBudget,
    max_message_bytes: usize,
    lanes: Arc<SeqInbox>,
    control: Arc<KeyedMailbox<ControlKey, NexarMessage>>,
    rpc_requests: mpsc::Sender<NexarMessage>,
    rpc_waiters: RpcWaiters,
    relay: mpsc::Sender<NexarMessage>,
}

impl PeerRouter {
    /// Spawn a background receive loop for `conn` and return the router,
    /// with the default [`RouterLimits`].
    pub fn spawn(
        rank: Rank,
        conn: quinn::Connection,
        pool: Arc<BufferPool>,
    ) -> (Self, tokio::task::JoinHandle<Result<()>>) {
        Self::spawn_with_limits(rank, conn, pool, RouterLimits::default())
    }

    /// Like [`spawn`](Self::spawn), with `limits`. A depth below 1 counts as 1.
    pub fn spawn_with_limits(
        rank: Rank,
        conn: quinn::Connection,
        pool: Arc<BufferPool>,
        limits: RouterLimits,
    ) -> (Self, tokio::task::JoinHandle<Result<()>>) {
        let depth = limits.depth.max(1);
        let (rpc_tx, rpc_rx) = mpsc::channel(depth);
        let (relay_tx, relay_rx) = mpsc::channel(depth);
        let rpc_waiters: RpcWaiters = Arc::new(Mutex::new(HashMap::new()));
        let lanes = Arc::new(SeqInbox::new(depth));
        let control = Arc::new(KeyedMailbox::new(depth));

        let ctx = Arc::new(RouterCtx {
            rank,
            conn: conn.clone(),
            pool,
            reads: Semaphore::new(MAX_CONCURRENT_READS),
            budget: ByteBudget::new(limits.inflight_bytes),
            max_message_bytes: limits.max_message_bytes,
            lanes: Arc::clone(&lanes),
            control: Arc::clone(&control),
            rpc_requests: rpc_tx,
            rpc_waiters: Arc::clone(&rpc_waiters),
            relay: relay_tx,
        });
        let handle = tokio::spawn(accept_loop(conn, ctx));

        let router = Self {
            lanes,
            control,
            rpc_requests: Mutex::new(rpc_rx),
            rpc_waiters,
            relay: Mutex::new(Some(relay_rx)),
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

    /// Receive the next message of `lane` from this router's peer `rank`, in send order.
    ///
    /// Messages of other lanes stay queued. After the connection closes and
    /// `lane` has no complete message left, returns
    /// [`NexarError::PeerDisconnected`] naming `rank`. Cancel-safe.
    pub async fn recv_lane(&self, rank: Rank, lane: LaneKey) -> Result<PooledBuf> {
        self.lanes
            .recv(lane)
            .await
            .ok_or(NexarError::PeerDisconnected { rank })
    }

    /// Receive a control message with key `key` from this router's peer `rank`.
    ///
    /// `rank` names the peer in [`NexarError::PeerDisconnected`]. Cancel-safe.
    pub async fn recv_control(&self, rank: Rank, key: ControlKey) -> Result<NexarMessage> {
        self.control
            .recv(&key)
            .await
            .ok_or(NexarError::PeerDisconnected { rank })
    }

    /// Receive the next RPC request.
    pub async fn recv_rpc_request(&self, rank: Rank) -> Result<NexarMessage> {
        self.rpc_requests
            .lock()
            .await
            .recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank })
    }

    /// Take the relay receiver out of this router (for the relay listener
    /// task). Returns `None` once taken.
    pub async fn take_relay_rx(&self) -> Option<mpsc::Receiver<NexarMessage>> {
        self.relay.lock().await.take()
    }
}

/// Accept incoming uni streams and handle each on its own task.
///
/// When the connection ends, waits for every handler, then closes the
/// receive queues, so a receiver sees every message that arrived whole.
async fn accept_loop(conn: quinn::Connection, ctx: Arc<RouterCtx>) -> Result<()> {
    let mut handlers = JoinSet::new();
    loop {
        tokio::select! {
            accepted = conn.accept_uni() => match accepted {
                Ok(stream) => {
                    handlers.spawn(handle_stream(stream, Arc::clone(&ctx)));
                }
                Err(_) => break,
            },
            Some(_) = handlers.join_next(), if !handlers.is_empty() => {}
        }
    }
    while handlers.join_next().await.is_some() {}
    ctx.lanes.close();
    ctx.control.close();
    ctx.rpc_waiters.lock().await.clear();
    Ok(())
}

/// Read one stream and hand its message to the receiver of its key.
async fn handle_stream(mut stream: quinn::RecvStream, ctx: Arc<RouterCtx>) {
    let rank = ctx.rank;
    let mut tag_buf = [0u8; 1];
    if stream.read_exact(&mut tag_buf).await.is_err() {
        tracing::debug!(rank, "router: stream ended before its type byte");
        return;
    }
    match tag_buf[0] {
        STREAM_TAG_SEQ => handle_lane(stream, &ctx).await,
        STREAM_TAG_SKIP => handle_skip(stream, &ctx).await,
        STREAM_TAG_FRAMED => {
            if let Some(msg) = read_framed(&mut stream, &ctx).await {
                dispatch_framed(msg, &ctx).await;
            }
        }
        other => {
            tracing::warn!(
                rank,
                "router: unknown stream tag 0x{other:02x}, skipping stream"
            );
        }
    }
}

/// A sequenced byte message: header, admission into its lane, then payload.
async fn handle_lane(mut stream: quinn::RecvStream, ctx: &RouterCtx) {
    let rank = ctx.rank;
    let mut head = [0u8; SeqHeader::SIZE];
    if let Err(e) = stream.read_exact(&mut head).await {
        // The sender sees the failure too and sends a skip notice.
        tracing::debug!(rank, "router: lane header read failed: {e}");
        return;
    }
    let header = SeqHeader::decode(&head);
    let Some(lane) = header.lane else {
        tracing::warn!(
            rank,
            seq = header.seq,
            "router: unknown lane kind, skipping stream"
        );
        ctx.lanes.skip(&[header.seq]);
        return;
    };
    let Some(mut pending) = ctx.lanes.announce(lane, header.seq) else {
        return; // the sender abandoned this message
    };
    if refuse_oversized(&mut stream, header.len, ctx) {
        return;
    }
    // Wait for room in the lane with the payload still unread.
    tokio::select! {
        () = pending.admitted() => {}
        _ = ctx.conn.closed() => return,
    }
    let Some((_budget, _read)) = reserve_read(ctx, header.len).await else {
        return;
    };
    let mut buf = ctx.pool.checkout(header.len as usize);
    match stream.read_exact(&mut buf).await {
        Ok(()) => pending.fill(buf),
        Err(e) => tracing::debug!(rank, ?lane, "router: lane payload read failed: {e}"),
    }
}

/// A skip notice: sequence numbers whose messages never complete.
async fn handle_skip(mut stream: quinn::RecvStream, ctx: &RouterCtx) {
    let mut count = [0u8; 4];
    if stream.read_exact(&mut count).await.is_err() {
        return;
    }
    let count = u32::from_le_bytes(count);
    if count > MAX_SKIPS {
        tracing::warn!(
            rank = ctx.rank,
            count,
            "router: oversized skip notice, ignoring"
        );
        return;
    }
    let mut raw = vec![0u8; count as usize * 8];
    if stream.read_exact(&mut raw).await.is_err() {
        return;
    }
    let seqs: Vec<u64> = raw
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    ctx.lanes.skip(&seqs);
}

/// Route a decoded framed message to its receiver. Waits for room in that
/// receiver's queue only, and gives up when the connection closes.
async fn dispatch_framed(msg: NexarMessage, ctx: &RouterCtx) {
    let rank = ctx.rank;
    let delivered = match msg {
        NexarMessage::Rpc { .. } => tokio::select! {
            r = ctx.rpc_requests.send(msg) => r.is_ok(),
            _ = ctx.conn.closed() => false,
        },
        NexarMessage::RpcResponse { req_id, .. } => {
            match ctx.rpc_waiters.lock().await.remove(&req_id) {
                Some(waiter) => {
                    let _ = waiter.send(msg);
                }
                None => tracing::warn!(rank, req_id, "router: RpcResponse with no waiter"),
            }
            true
        }
        NexarMessage::Relay { .. } => tokio::select! {
            r = ctx.relay.send(msg) => r.is_ok(),
            _ = ctx.conn.closed() => false,
        },
        // Liveness comes from send success, so a received heartbeat carries no
        // information.
        NexarMessage::Heartbeat { .. } => true,
        other => match ControlKey::of(&other) {
            Some(key) => tokio::select! {
                r = ctx.control.send(key, other) => r.is_ok(),
                _ = ctx.conn.closed() => false,
            },
            None => {
                tracing::warn!(rank, "router: no mesh receiver for {other:?}, dropping");
                true
            }
        },
    };
    if !delivered {
        tracing::debug!(rank, "router: receiver gone, message dropped");
    }
}

/// Stop `stream` and return `true` when a message of `len` bytes exceeds the
/// limit. Nothing is allocated for it.
fn refuse_oversized(stream: &mut quinn::RecvStream, len: u64, ctx: &RouterCtx) -> bool {
    if len <= ctx.max_message_bytes as u64 {
        return false;
    }
    let err = NexarError::FrameTooLarge {
        len,
        max: ctx.max_message_bytes,
    };
    tracing::warn!(rank = ctx.rank, "router: {err}, stopping its stream");
    let _ = stream.stop(quinn::VarInt::from_u32(REFUSED_CODE));
    true
}

/// Wait for room in the byte budget and for a read slot. Hold both while the
/// payload is read. `None`: the router is shutting down.
async fn reserve_read(
    ctx: &RouterCtx,
    len: u64,
) -> Option<(SemaphorePermit<'_>, SemaphorePermit<'_>)> {
    let budget = ctx.budget.reserve(len).await?;
    let slot = ctx.reads.acquire().await.ok()?;
    Some((budget, slot))
}

/// Read a length-prefixed framed message (after the type byte).
async fn read_framed(stream: &mut quinn::RecvStream, ctx: &RouterCtx) -> Option<NexarMessage> {
    let rank = ctx.rank;
    let mut len_buf = [0u8; 8];
    if let Err(e) = stream.read_exact(&mut len_buf).await {
        tracing::warn!(rank, "router: framed length read failed: {e}");
        return None;
    }
    let len = u64::from_le_bytes(len_buf);
    if refuse_oversized(stream, len, ctx) {
        return None;
    }
    let _held = reserve_read(ctx, len).await?;
    let mut buf = ctx.pool.checkout(len as usize);
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
