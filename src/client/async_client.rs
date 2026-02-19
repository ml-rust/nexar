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
enum RawRecvSource {
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
    rank: Rank,
    world_size: u32,
    /// Communicator ID. 0 = default (root) communicator.
    comm_id: u32,
    /// Sending side: one `PeerConnection` per remote rank.
    /// For split clients, this contains only the peers in this comm group,
    /// keyed by the NEW rank within the group.
    pub(crate) peers: HashMap<Rank, Arc<PeerConnection>>,
    /// Receiving side: one `PeerRouter` per remote rank.
    /// For split clients, this is a reference to the parent's routers
    /// (keyed by the ORIGINAL rank). The split client uses comm-specific
    /// raw channels instead of the router's default raw lane.
    routers: HashMap<Rank, PeerRouter>,
    /// How this client receives raw bytes. Default clients use Router,
    /// split clients use Comm with per-comm_id channels.
    raw_recv: RawRecvSource,
    /// Background tasks; kept alive for the lifetime of this client.
    _router_handles: Vec<tokio::task::JoinHandle<Result<()>>>,
    adapter: Arc<dyn DeviceAdapter>,
    /// Shared buffer pool for router read buffers.
    _pool: Arc<BufferPool>,
    barrier_epoch: AtomicU64,
    rpc_registry: Arc<RwLock<RpcRegistry>>,
    rpc_req_id: AtomicU64,
    /// Global rank mapping: new_rank -> original_rank (for split clients).
    /// Empty for the root communicator.
    rank_map: HashMap<Rank, Rank>,
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

    /// Send raw bytes to a peer (used by collective algorithms).
    /// Uses comm-aware send for split communicators.
    pub(crate) async fn send_bytes(&self, dest: Rank, data: &[u8]) -> Result<()> {
        let peer = self.peer(dest)?;
        if self.comm_id == 0 {
            peer.send_raw(data).await
        } else {
            peer.send_raw_comm(self.comm_id, data).await
        }
    }

    /// Receive raw bytes from a peer (used by collective algorithms).
    /// Uses comm-aware recv for split communicators.
    pub(crate) async fn recv_bytes(&self, src: Rank) -> Result<PooledBuf> {
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

    // ========================================================================
    // GPUDirect operations
    // ========================================================================

    /// Send raw bytes directly from GPU memory to a peer.
    ///
    /// Uses GPUDirect RDMA if available, falling back to staged transfers.
    #[cfg(feature = "gpudirect")]
    pub async fn send_bytes_gpu(&self, dest: Rank, gpu_ptr: u64, size: usize) -> Result<()> {
        if dest >= self.world_size {
            return Err(NexarError::InvalidRank {
                rank: dest,
                world_size: self.world_size,
            });
        }
        let peer = self.peer(dest)?;
        peer.send_raw_gpu(gpu_ptr, size).await
    }

    /// Receive raw bytes directly into GPU memory from a peer.
    ///
    /// Uses GPUDirect RDMA if available, falling back to staged transfers.
    #[cfg(feature = "gpudirect")]
    pub async fn recv_bytes_gpu(&self, src: Rank, gpu_ptr: u64, size: usize) -> Result<()> {
        if src >= self.world_size {
            return Err(NexarError::InvalidRank {
                rank: src,
                world_size: self.world_size,
            });
        }
        let peer = self.peer(src)?;
        peer.recv_raw_gpu(gpu_ptr, size).await
    }

    // ========================================================================
    // Communicator splitting
    // ========================================================================

    /// Split this communicator into sub-groups.
    ///
    /// All ranks must call `split` with the same arguments simultaneously.
    /// Ranks with the same `color` end up in the same sub-communicator.
    /// Within each group, ranks are ordered by `key` (ties broken by original rank).
    ///
    /// The returned client has new rank/world_size within the sub-group and uses
    /// a unique `comm_id` for its raw stream traffic, so collectives on the
    /// sub-communicator don't interfere with the parent or other sub-groups.
    ///
    /// The parent client's routers demux raw streams by `comm_id`, so the parent
    /// must remain alive for the duration of the split client's use.
    pub async fn split(&self, color: u32, key: u32) -> Result<NexarClient> {
        let world = self.world_size as usize;
        let rank = self.rank;

        // Step 1: Exchange (color, key) tuples with all peers.
        // Encode as 8 bytes: [color: u32 LE][key: u32 LE].
        let mut my_info = [0u8; 8];
        my_info[..4].copy_from_slice(&color.to_le_bytes());
        my_info[4..].copy_from_slice(&key.to_le_bytes());

        // AllGather the info from all ranks.
        let mut all_info = vec![0u8; 8 * world];
        all_info[rank as usize * 8..(rank as usize + 1) * 8].copy_from_slice(&my_info);

        // Use the existing allgather collective. We pass raw pointers to our
        // stack-allocated buffers.
        let send_ptr = my_info.as_ptr() as u64;
        let recv_ptr = all_info.as_mut_ptr() as u64;
        unsafe {
            crate::collective::ring_allgather(
                self,
                send_ptr,
                recv_ptr,
                8, // 8 bytes per rank
                crate::types::DataType::U8,
            )
            .await?;
        }

        // Step 2: Parse all (color, key) tuples.
        let mut entries: Vec<(Rank, u32, u32)> = Vec::with_capacity(world);
        for r in 0..world {
            let off = r * 8;
            let c = u32::from_le_bytes(all_info[off..off + 4].try_into().unwrap());
            let k = u32::from_le_bytes(all_info[off + 4..off + 8].try_into().unwrap());
            entries.push((r as Rank, c, k));
        }

        // Step 3: Find our group (same color), sort by (key, original_rank).
        let my_color = color;
        let mut group: Vec<(Rank, u32)> = entries
            .iter()
            .filter(|&&(_, c, _)| c == my_color)
            .map(|&(r, _, k)| (r, k))
            .collect();
        group.sort_by_key(|&(orig_rank, k)| (k, orig_rank));

        let new_world_size = group.len() as u32;
        let new_rank = group
            .iter()
            .position(|&(r, _)| r == rank)
            .expect("rank must be in its own color group") as Rank;

        // Step 4: Generate a unique comm_id from color.
        // Use color + 1 to avoid comm_id 0 (reserved for root communicator).
        let new_comm_id = my_color + 1;

        // Step 5: Build rank_map (new_rank -> original_rank) and peer subset.
        let mut rank_map = HashMap::new();
        let mut new_peers = HashMap::new();
        let mut comm_receivers = HashMap::new();

        for (new_r, &(orig_rank, _)) in group.iter().enumerate() {
            let new_r = new_r as Rank;
            rank_map.insert(new_r, orig_rank);

            if orig_rank != rank {
                // Share the parent's PeerConnection (keyed by original rank).
                let peer = self.peer(orig_rank)?;
                new_peers.insert(new_r, Arc::clone(peer));

                // Register a per-comm_id channel on the parent's router for this peer.
                let original_rank_key = orig_rank;
                let router =
                    self.routers
                        .get(&original_rank_key)
                        .ok_or(NexarError::UnknownPeer {
                            rank: original_rank_key,
                        })?;
                let rx = router.register_comm(new_comm_id).await;
                comm_receivers.insert(new_r, Mutex::new(rx));
            }
        }

        // Step 6: Build the split client. It shares the parent's routers
        // but uses comm-specific raw channels.
        // Note: The split client doesn't own routers or router handles — it
        // borrows the parent's routers indirectly through the registered comm channels.
        // Control/data/RPC lanes are still on the parent's routers.
        // Split clients don't have their own routers — they use comm channels for
        // collective raw data. Barrier on a split client would need its own mechanism.
        // For the TP/PP/DP use case, the parent client handles barriers.

        Ok(NexarClient {
            rank: new_rank,
            world_size: new_world_size,
            comm_id: new_comm_id,
            peers: new_peers,
            routers: HashMap::new(), // Split clients don't own routers
            raw_recv: RawRecvSource::Comm(comm_receivers),
            _router_handles: Vec::new(),
            adapter: Arc::clone(&self.adapter),
            _pool: Arc::clone(&self._pool),
            barrier_epoch: AtomicU64::new(0),
            rpc_registry: Arc::new(RwLock::new(RpcRegistry::new())),
            rpc_req_id: AtomicU64::new(0),
            rank_map,
        })
    }

    // ========================================================================
    // Collective operations
    // ========================================================================

    /// AllReduce in-place using ring algorithm.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn all_reduce(
        &self,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> Result<()> {
        unsafe { crate::collective::ring_allreduce(self, ptr, count, dtype, op).await }
    }

    /// Broadcast from root rank to all others.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn broadcast(
        &self,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe { crate::collective::tree_broadcast(self, ptr, count, dtype, root).await }
    }

    /// AllGather: each rank contributes `count` elements, result is
    /// `count * world_size` elements on all ranks.
    ///
    /// # Safety
    /// - `send_ptr` must point to at least `count * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
    pub async unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
    ) -> Result<()> {
        unsafe { crate::collective::ring_allgather(self, send_ptr, recv_ptr, count, dtype).await }
    }

    /// ReduceScatter: reduce across all ranks, each rank gets a different slice.
    ///
    /// # Safety
    /// - `send_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr` must point to at least `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> Result<()> {
        unsafe {
            crate::collective::ring_reduce_scatter(self, send_ptr, recv_ptr, count, dtype, op).await
        }
    }

    /// Reduce to a single root rank.
    ///
    /// After completion, only `root` holds the reduced result.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn reduce(
        &self,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
        root: Rank,
    ) -> Result<()> {
        unsafe { crate::collective::tree_reduce(self, ptr, count, dtype, op, root).await }
    }

    /// All-to-all: each rank sends a distinct chunk to every other rank.
    ///
    /// # Safety
    /// - `send_ptr`: `count * world_size * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr`: `count * world_size * dtype.size_in_bytes()` bytes.
    pub async unsafe fn all_to_all(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
    ) -> Result<()> {
        unsafe { crate::collective::alltoall(self, send_ptr, recv_ptr, count, dtype).await }
    }

    /// Inclusive prefix scan: rank `i` holds the reduction of ranks 0..=i.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn scan(
        &self,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> Result<()> {
        unsafe { crate::collective::inclusive_scan(self, ptr, count, dtype, op).await }
    }

    /// Barrier: block until all ranks reach this point.
    ///
    /// Automatically selects the best algorithm based on world size:
    /// two-phase for small clusters, dissemination for larger ones.
    pub async fn barrier(&self) -> Result<()> {
        crate::collective::barrier(self, std::time::Duration::from_secs(30)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuAdapter;

    #[tokio::test]
    async fn test_bootstrap_single_node() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = NexarClient::bootstrap_local(1, adapter).await.unwrap();
        assert_eq!(clients.len(), 1);
        assert_eq!(clients[0].rank(), 0);
        assert_eq!(clients[0].world_size(), 1);
    }

    #[tokio::test]
    async fn test_bootstrap_two_nodes() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = NexarClient::bootstrap_local(2, adapter).await.unwrap();
        assert_eq!(clients.len(), 2);
        assert_eq!(clients[0].rank(), 0);
        assert_eq!(clients[1].rank(), 1);
        assert_eq!(clients[0].world_size(), 2);
    }

    #[tokio::test]
    async fn test_bootstrap_four_nodes() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = NexarClient::bootstrap_local(4, adapter).await.unwrap();
        assert_eq!(clients.len(), 4);
        for (i, c) in clients.iter().enumerate() {
            assert_eq!(c.rank() as usize, i);
            assert_eq!(c.world_size(), 4);
        }
    }

    #[tokio::test]
    async fn test_send_recv_two_nodes() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = NexarClient::bootstrap_local(2, adapter).await.unwrap();

        let send_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut recv_buf: Vec<f32> = vec![0.0; 4];
        let size = send_data.len() * std::mem::size_of::<f32>();

        let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();
        let c0 = Arc::clone(&clients[0]);
        let c1 = Arc::clone(&clients[1]);

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_buf.as_mut_ptr() as u64;

        let send_task =
            tokio::spawn(async move { unsafe { c0.send(send_ptr, size, 1, 42).await } });
        let recv_task =
            tokio::spawn(async move { unsafe { c1.recv(recv_ptr, size, 0, 42).await } });

        send_task.await.unwrap().unwrap();
        recv_task.await.unwrap().unwrap();

        assert_eq!(recv_buf, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
