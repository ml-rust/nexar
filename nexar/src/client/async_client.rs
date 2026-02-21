use crate::cluster::HealthMonitor;
use crate::cluster::sparse::RoutingTable;
use crate::config::NexarConfig;
use crate::device::DeviceAdapter;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::rpc::RpcDispatcher;
use crate::rpc::registry::{RpcHandler, RpcRegistry};
use crate::transport::PeerConnection;
use crate::transport::buffer_pool::{BufferPool, PoolProfile, PooledBuf};
use crate::transport::relay::RelayDeliveries;
use crate::transport::router::PeerRouter;
use crate::types::{Priority, Rank};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, RwLock, watch};

/// Cached tagged receiver map: (original_rank, tag) -> shared Receiver.
type TaggedReceiverMap = HashMap<(Rank, u64), Arc<Mutex<tokio::sync::mpsc::Receiver<PooledBuf>>>>;

/// Return type for [`NexarClient::spawn_monitor`]: (failure_tx, failure_rx, monitor_handle).
type MonitorParts = (
    Arc<watch::Sender<Vec<Rank>>>,
    watch::Receiver<Vec<Rank>>,
    tokio::task::JoinHandle<()>,
);

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
    pub(super) comm_id: u64,
    /// Sending side: one `PeerConnection` per remote rank.
    pub(crate) peers: HashMap<Rank, Arc<PeerConnection>>,
    /// Receiving side: one `PeerRouter` per remote rank.
    pub(super) routers: HashMap<Rank, PeerRouter>,
    /// How this client receives raw bytes.
    pub(super) raw_recv: RawRecvSource,
    /// Background tasks; kept alive for the lifetime of this client.
    pub(super) _router_handles: Vec<tokio::task::JoinHandle<Result<()>>>,
    pub(super) adapter: Arc<dyn DeviceAdapter>,
    /// Shared buffer pool for router read buffers.
    pub(super) _pool: Arc<BufferPool>,
    pub(super) barrier_epoch: AtomicU64,
    pub(super) rpc_registry: Arc<RwLock<RpcRegistry>>,
    pub(super) rpc_req_id: AtomicU64,
    /// Per-client split generation counter.
    pub(super) split_generation: AtomicU64,
    /// Global rank mapping: new_rank -> original_rank (for split clients).
    pub(super) rank_map: HashMap<Rank, Rank>,
    /// Counter for generating unique collective tags.
    pub(super) collective_tag: AtomicU64,
    /// Cached receivers for tagged channels.
    pub(super) tagged_receivers: Mutex<TaggedReceiverMap>,
    /// Runtime configuration (timeouts, thresholds).
    pub(crate) config: Arc<NexarConfig>,
    /// Sender for failure notifications (health monitor writes here).
    pub(super) failure_tx: Arc<watch::Sender<Vec<Rank>>>,
    /// Receiver for failure notifications (application reads here).
    pub(super) failure_rx: watch::Receiver<Vec<Rank>>,
    /// Heartbeat monitor background task handle.
    pub(super) _monitor_handle: Option<tokio::task::JoinHandle<()>>,
    /// Routing table for sparse topologies (None for full mesh).
    pub(crate) routing_table: Option<Arc<RoutingTable>>,
    /// Relay delivery channels for messages arriving via intermediate hops.
    pub(crate) relay_deliveries: Option<Arc<RelayDeliveries>>,
    /// Background relay listener task handles.
    pub(super) _relay_handles: Vec<tokio::task::JoinHandle<()>>,
    /// QUIC client endpoints kept alive so their UDP sockets remain open.
    pub(crate) _endpoints: Vec<quinn::Endpoint>,
}

impl NexarClient {
    /// Create a client from pre-established peer connections.
    pub fn new(
        rank: Rank,
        world_size: u32,
        peers: HashMap<Rank, PeerConnection>,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Self {
        Self::new_with_config(
            rank,
            world_size,
            peers,
            adapter,
            PoolProfile::Training,
            NexarConfig::from_env(),
        )
    }

    /// Create a client with a specific buffer pool profile.
    pub fn new_with_profile(
        rank: Rank,
        world_size: u32,
        peers: HashMap<Rank, PeerConnection>,
        adapter: Arc<dyn DeviceAdapter>,
        profile: PoolProfile,
    ) -> Self {
        Self::new_with_config(
            rank,
            world_size,
            peers,
            adapter,
            profile,
            NexarConfig::from_env(),
        )
    }

    /// Create a client with a specific buffer pool profile and config.
    pub fn new_with_config(
        rank: Rank,
        world_size: u32,
        peers: HashMap<Rank, PeerConnection>,
        adapter: Arc<dyn DeviceAdapter>,
        profile: PoolProfile,
        config: NexarConfig,
    ) -> Self {
        let pool = BufferPool::with_profile(profile);
        Self::build(rank, world_size, peers, adapter, pool, config)
    }

    /// Create a client with a user-supplied buffer pool.
    ///
    /// Use this to share a single pool across multiple clients, or to pass
    /// a pool built with [`PoolBuilder`] for custom tier sizing.
    pub fn new_with_pool(
        rank: Rank,
        world_size: u32,
        peers: HashMap<Rank, PeerConnection>,
        adapter: Arc<dyn DeviceAdapter>,
        pool: Arc<BufferPool>,
    ) -> Self {
        Self::new_with_pool_and_config(
            rank,
            world_size,
            peers,
            adapter,
            pool,
            NexarConfig::from_env(),
        )
    }

    /// Create a client with a user-supplied buffer pool and config.
    pub fn new_with_pool_and_config(
        rank: Rank,
        world_size: u32,
        peers: HashMap<Rank, PeerConnection>,
        adapter: Arc<dyn DeviceAdapter>,
        pool: Arc<BufferPool>,
        config: NexarConfig,
    ) -> Self {
        Self::build(rank, world_size, peers, adapter, pool, config)
    }

    /// Shared constructor logic for all `new_*` variants.
    fn build(
        rank: Rank,
        world_size: u32,
        peers: HashMap<Rank, PeerConnection>,
        adapter: Arc<dyn DeviceAdapter>,
        pool: Arc<BufferPool>,
        config: NexarConfig,
    ) -> Self {
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

        let (failure_tx, failure_rx, monitor_handle) = Self::spawn_monitor(&config, &peer_arcs);

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
            collective_tag: AtomicU64::new(1),
            tagged_receivers: Mutex::new(HashMap::new()),
            config: Arc::new(config),
            failure_tx,
            failure_rx,
            _monitor_handle: Some(monitor_handle),
            routing_table: None,
            relay_deliveries: None,
            _relay_handles: Vec::new(),
            _endpoints: Vec::new(),
        }
    }

    /// Spawn the heartbeat monitor and return the failure notification channels.
    fn spawn_monitor(
        config: &NexarConfig,
        peers: &HashMap<Rank, Arc<PeerConnection>>,
    ) -> MonitorParts {
        let (failure_tx, failure_rx) = watch::channel(Vec::new());
        let failure_tx = Arc::new(failure_tx);
        let monitor =
            HealthMonitor::with_timeout(config.heartbeat_interval, config.heartbeat_timeout);
        let monitor_peers: Vec<_> = peers.iter().map(|(r, p)| (*r, Arc::clone(p))).collect();
        let handle = monitor.start_monitoring(monitor_peers, Arc::clone(&failure_tx));
        (failure_tx, failure_rx, handle)
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

    /// Get a reference to the RPC dispatcher for this client.
    pub fn rpc_dispatcher(&self) -> RpcDispatcher {
        RpcDispatcher::new(Arc::clone(&self.rpc_registry))
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
    pub fn comm_id(&self) -> u64 {
        self.comm_id
    }

    /// Reference to the device adapter used for memory staging.
    pub fn adapter(&self) -> &dyn DeviceAdapter {
        self.adapter.as_ref()
    }

    /// Runtime configuration.
    pub fn config(&self) -> &NexarConfig {
        &self.config
    }

    /// Get a reference to a peer connection (for sending).
    pub fn peer(&self, rank: Rank) -> Result<&Arc<PeerConnection>> {
        self.peers
            .get(&rank)
            .ok_or(NexarError::UnknownPeer { rank })
    }

    /// Resolve a rank in this communicator to the original (global) rank.
    /// For the root communicator, this is identity.
    pub(super) fn resolve_rank(&self, rank: Rank) -> Rank {
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

        let msg = NexarMessage::Data {
            tag,
            src_rank: self.rank,
            payload: data,
        };
        self.send_message_to(dest, &msg, Priority::Bulk).await
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

        // For non-neighbor sources in sparse topology, receive via relay.
        let msg = if !self.has_direct_peer(src) && self.relay_deliveries.is_some() {
            self.recv_control_from(src).await?
        } else {
            self.recv_data_message(src).await?
        };

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

    /// Get a receiver that notifies when peers are detected as failed.
    ///
    /// The receiver yields the current list of dead peer ranks whenever it changes.
    /// Use `.changed().await` to wait for the next failure event.
    pub fn failure_watch(&self) -> watch::Receiver<Vec<Rank>> {
        self.failure_rx.clone()
    }

    /// Get the next unique collective tag for non-blocking collectives.
    pub(crate) fn next_collective_tag(&self) -> u64 {
        self.collective_tag.fetch_add(1, Ordering::Relaxed)
    }

    /// Close all peer connections immediately.
    ///
    /// This sends a QUIC `CONNECTION_CLOSE` frame to every peer, causing
    /// their in-flight sends/recvs to fail promptly. Useful for graceful
    /// shutdown and fault-injection testing.
    pub fn close(&self) {
        for peer in self.peers.values() {
            peer.conn.close(0u32.into(), b"closed");
        }
    }

    /// Returns true if `rank` is a direct peer (has a QUIC connection).
    pub fn has_direct_peer(&self, rank: Rank) -> bool {
        self.peers.contains_key(&rank)
    }

    /// Returns true if the topology is sparse (not full mesh).
    pub fn is_sparse(&self) -> bool {
        self.routing_table.is_some()
    }
}
