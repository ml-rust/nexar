use crate::device::DeviceAdapter;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::rpc::RpcDispatcher;
use crate::rpc::registry::{RpcHandler, RpcRegistry};
use crate::transport::PeerConnection;
use crate::transport::buffer_pool::{BufferPool, PoolProfile, PooledBuf};
use crate::transport::router::PeerRouter;
use crate::types::{Priority, Rank};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, RwLock};

/// Abstraction over raw byte receive channels.
/// Default clients use the router's raw lane; split clients use per-comm_id channels.
pub(super) enum RawRecvSource {
    /// Default communicator: recv from router's raw lane.
    Router,
    /// Split communicator: recv from per-comm_id channels.
    Comm(HashMap<Rank, Mutex<tokio::sync::mpsc::Receiver<PooledBuf>>>),
}

/// The main async API for nexar distributed communication.
///
/// Holds peer connections (for sending) and per-peer routers (for receiving).
/// The routers run as background tasks that demultiplex incoming QUIC streams
/// into typed channels, preventing races between consumers (RPC, barrier,
/// collectives) that would otherwise steal each other's messages.
///
/// # Example
///
/// ```no_run
/// use nexar::client::NexarClient;
/// use nexar::device::CpuAdapter;
/// use std::sync::Arc;
///
/// # async fn example() -> nexar::error::Result<()> {
/// let adapter = Arc::new(CpuAdapter::new());
/// let clients = NexarClient::bootstrap_local(4, adapter).await?;
///
/// // Each client has a unique rank in [0, world_size).
/// assert_eq!(clients[0].rank(), 0);
/// assert_eq!(clients[0].world_size(), 4);
/// # Ok(())
/// # }
/// ```
pub struct NexarClient {
    pub(super) rank: Rank,
    pub(super) world_size: u32,
    /// Communicator ID. 0 = default (root) communicator.
    pub(super) comm_id: u32,
    /// Sending side: one `PeerConnection` per remote rank.
    /// For split clients, this contains only the peers in this comm group,
    /// keyed by the NEW rank within the group.
    pub(crate) peers: HashMap<Rank, Arc<PeerConnection>>,
    /// Receiving side: one `PeerRouter` per remote rank.
    /// For split clients, this is a reference to the parent's routers
    /// (keyed by the ORIGINAL rank). The split client uses comm-specific
    /// raw channels instead of the router's default raw lane.
    pub(super) routers: HashMap<Rank, PeerRouter>,
    /// How this client receives raw bytes. Default clients use Router,
    /// split clients use Comm with per-comm_id channels.
    pub(super) raw_recv: RawRecvSource,
    /// Background tasks; kept alive for the lifetime of this client.
    pub(super) _router_handles: Vec<tokio::task::JoinHandle<Result<()>>>,
    pub(super) adapter: Arc<dyn DeviceAdapter>,
    /// Shared buffer pool for router read buffers.
    pub(super) _pool: Arc<BufferPool>,
    pub(super) barrier_epoch: AtomicU64,
    pub(super) rpc_registry: Arc<RwLock<RpcRegistry>>,
    pub(super) rpc_req_id: AtomicU64,
    /// Per-client split generation counter. All ranks in a communicator advance
    /// this in lockstep (since `split()` is called collectively), so it can be
    /// used to derive deterministic comm_ids.
    pub(super) split_generation: AtomicU64,
    /// Global rank mapping: new_rank -> original_rank (for split clients).
    /// Empty for the root communicator.
    pub(super) rank_map: HashMap<Rank, Rank>,
}

impl NexarClient {
    /// Create a client from pre-established peer connections.
    pub fn new(
        rank: Rank,
        world_size: u32,
        peers: HashMap<Rank, PeerConnection>,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Self {
        Self::new_with_profile(rank, world_size, peers, adapter, PoolProfile::Training)
    }

    /// Create a client with a specific buffer pool profile.
    pub fn new_with_profile(
        rank: Rank,
        world_size: u32,
        peers: HashMap<Rank, PeerConnection>,
        adapter: Arc<dyn DeviceAdapter>,
        profile: PoolProfile,
    ) -> Self {
        let pool = BufferPool::with_profile(profile);
        let mut peer_arcs: HashMap<Rank, Arc<PeerConnection>> = HashMap::new();
        let mut routers: HashMap<Rank, PeerRouter> = HashMap::new();
        let mut handles = Vec::new();

        for (peer_rank, peer_conn) in peers {
            let conn_clone = peer_conn.conn.clone();
            let (router, handle) = PeerRouter::spawn(peer_rank, conn_clone, Arc::clone(&pool));
            peer_arcs.insert(peer_rank, Arc::new(peer_conn));
            routers.insert(peer_rank, router);
            handles.push(handle);
        }

        Self {
            rank,
            world_size,
            comm_id: 0,
            peers: peer_arcs,
            routers,
            raw_recv: RawRecvSource::Router,
            _router_handles: handles,
            adapter,
            _pool: pool,
            barrier_epoch: AtomicU64::new(0),
            rpc_registry: Arc::new(RwLock::new(RpcRegistry::new())),
            rpc_req_id: AtomicU64::new(0),
            split_generation: AtomicU64::new(0),
            rank_map: HashMap::new(),
        }
    }

    /// Get the next barrier epoch (per-client counter).
    pub(crate) fn next_barrier_epoch(&self) -> u64 {
        self.barrier_epoch.fetch_add(1, Ordering::Relaxed)
    }

    /// Register an RPC handler for a function ID.
    pub async fn register_rpc(&self, fn_id: u16, handler: RpcHandler) {
        let mut reg = self.rpc_registry.write().await;
        reg.register(fn_id, handler);
    }

    /// Default RPC timeout.
    const RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

    /// Call a remote function on the target rank and wait for the response.
    pub async fn rpc(&self, target: Rank, fn_id: u16, args: &[u8]) -> Result<Vec<u8>> {
        self.rpc_with_timeout(target, fn_id, args, Self::RPC_TIMEOUT)
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
        let req_id = self.rpc_req_id.fetch_add(1, Ordering::Relaxed);

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

    /// Get a reference to the RPC dispatcher for this client.
    pub fn rpc_dispatcher(&self) -> RpcDispatcher {
        RpcDispatcher::new(Arc::clone(&self.rpc_registry))
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
    async fn recv_data_message(&self, src: Rank) -> Result<NexarMessage> {
        let original_src = self.resolve_rank(src);
        let router = self
            .routers
            .get(&original_src)
            .ok_or(NexarError::UnknownPeer { rank: src })?;
        router.recv_data(original_src).await
    }

    /// This client's rank within its communicator group (0-indexed).
    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Total number of ranks in the communicator group.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// The communicator ID (0 = root communicator).
    pub fn comm_id(&self) -> u32 {
        self.comm_id
    }

    /// Reference to the device adapter used for memory staging.
    pub fn adapter(&self) -> &dyn DeviceAdapter {
        self.adapter.as_ref()
    }

    /// Get a reference to a peer connection (for sending).
    pub fn peer(&self, rank: Rank) -> Result<&Arc<PeerConnection>> {
        self.peers
            .get(&rank)
            .ok_or(NexarError::UnknownPeer { rank })
    }

    /// Resolve a rank in this communicator to the original (global) rank.
    /// For the root communicator, this is identity.
    fn resolve_rank(&self, rank: Rank) -> Rank {
        self.rank_map.get(&rank).copied().unwrap_or(rank)
    }

    /// Send tagged data to a specific rank.
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
        if dest >= self.world_size {
            return Err(NexarError::InvalidRank {
                rank: dest,
                world_size: self.world_size,
            });
        }

        let data = unsafe { self.adapter.stage_for_send(data_ptr, size)? };
        let peer = self.peer(dest)?;

        let msg = NexarMessage::Data {
            tag,
            src_rank: self.rank,
            payload: data,
        };
        peer.send_message(&msg, Priority::Bulk).await
    }

    /// Receive tagged data from a specific rank.
    ///
    /// # Safety
    /// `buf_ptr` must be valid for `size` bytes.
    pub async unsafe fn recv(&self, buf_ptr: u64, size: usize, src: Rank, tag: u32) -> Result<()> {
        if src >= self.world_size {
            return Err(NexarError::InvalidRank {
                rank: src,
                world_size: self.world_size,
            });
        }

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
                if payload.len() != size {
                    return Err(NexarError::BufferSizeMismatch {
                        expected: size,
                        actual: payload.len(),
                    });
                }
                unsafe { self.adapter.receive_to_device(&payload, buf_ptr)? };
                Ok(())
            }
            other => Err(NexarError::DecodeFailed(format!(
                "expected Data message, got {other:?}"
            ))),
        }
    }

    /// Send raw bytes to a peer.
    ///
    /// Uses comm-aware send for split communicators. Always uses QUIC transport.
    /// For bulk data in collectives, prefer `send_bytes_best_effort` which
    /// auto-selects RDMA when available.
    pub async fn send_bytes(&self, dest: Rank, data: &[u8]) -> Result<()> {
        let peer = self.peer(dest)?;
        if self.comm_id == 0 {
            peer.send_raw(data).await
        } else {
            peer.send_raw_comm(self.comm_id, data).await
        }
    }

    /// Send raw bytes using the best available transport (RDMA if available, QUIC fallback).
    /// For split communicators, always uses QUIC (comm-id routing required).
    pub(crate) async fn send_bytes_best_effort(&self, dest: Rank, data: &[u8]) -> Result<()> {
        let peer = self.peer(dest)?;
        if self.comm_id == 0 {
            peer.send_raw_best_effort(data).await
        } else {
            // Split communicators need comm_id tagging, QUIC only.
            peer.send_raw_comm(self.comm_id, data).await
        }
    }

    /// Receive raw bytes from a peer.
    ///
    /// Uses comm-aware recv for split communicators.
    pub async fn recv_bytes(&self, src: Rank) -> Result<PooledBuf> {
        match &self.raw_recv {
            RawRecvSource::Router => {
                let original_src = self.resolve_rank(src);
                let router = self
                    .routers
                    .get(&original_src)
                    .ok_or(NexarError::UnknownPeer { rank: src })?;
                router.recv_raw(original_src).await
            }
            RawRecvSource::Comm(channels) => {
                let rx = channels
                    .get(&src)
                    .ok_or(NexarError::UnknownPeer { rank: src })?;
                rx.lock()
                    .await
                    .recv()
                    .await
                    .ok_or(NexarError::PeerDisconnected { rank: src })
            }
        }
    }
}
