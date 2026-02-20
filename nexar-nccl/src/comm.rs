use std::sync::Arc;

use cudarc::driver::CudaStream;
use nexar::NexarClient;
use nexar::types::{DataType, Rank, ReduceOp};

use crate::error::Result;
use crate::group::NcclGroup;
use crate::topology::NodeTopology;

/// Hierarchical communicator combining NCCL intra-node with nexar inter-node.
///
/// For single-node clusters, all operations delegate directly to NCCL.
/// For multi-node clusters, the 2D decomposition splits work between:
/// - NCCL for same-node GPU-GPU (NVLink/PCIe speed)
/// - nexar for cross-node (QUIC/RDMA)
pub struct HierarchicalComm {
    /// The full nexar client (all ranks).
    nexar: Arc<NexarClient>,
    /// Sub-communicator for inter-node leads only. `None` on non-lead ranks
    /// and on single-node clusters.
    inter_node: Option<NexarClient>,
    /// Intra-node NCCL group.
    nccl: NcclGroup,
    /// Topology information.
    topo: NodeTopology,
    /// Staging stream for D2H/H2D copies in the overlap path.
    /// Used by lead ranks to overlap NCCL ops with nexar network transfers.
    staging_stream: Option<Arc<CudaStream>>,
}

impl HierarchicalComm {
    pub(crate) fn new(
        nexar: Arc<NexarClient>,
        inter_node: Option<NexarClient>,
        nccl: NcclGroup,
        topo: NodeTopology,
        staging_stream: Option<Arc<CudaStream>>,
    ) -> Self {
        Self {
            nexar,
            inter_node,
            nccl,
            topo,
            staging_stream,
        }
    }

    /// Global rank of this process.
    pub fn rank(&self) -> Rank {
        self.nexar.rank()
    }

    /// Global world size.
    pub fn world_size(&self) -> u32 {
        self.nexar.world_size()
    }

    /// Number of nodes in the cluster.
    pub fn num_nodes(&self) -> usize {
        self.topo.num_nodes
    }

    /// Number of GPUs on this node.
    pub fn local_world_size(&self) -> usize {
        self.topo.local_world_size()
    }

    /// This rank's local index on its node.
    pub fn local_rank(&self) -> usize {
        self.topo.local_rank_idx
    }

    /// Whether this rank is the lead for its node.
    pub fn is_lead(&self) -> bool {
        self.topo.is_lead()
    }

    /// Whether this is a single-node cluster.
    pub fn is_single_node(&self) -> bool {
        self.topo.is_single_node()
    }

    /// Reference to the NCCL group.
    pub fn nccl(&self) -> &NcclGroup {
        &self.nccl
    }

    /// Reference to the inter-node nexar client (only available on lead ranks).
    pub fn inter_node(&self) -> Option<&NexarClient> {
        self.inter_node.as_ref()
    }

    /// Reference to the full nexar client.
    pub fn nexar(&self) -> &NexarClient {
        &self.nexar
    }

    /// Reference to the topology.
    pub fn topology(&self) -> &NodeTopology {
        &self.topo
    }

    /// Reference to the staging stream (for compute-communication overlap).
    pub fn staging_stream(&self) -> Option<&Arc<CudaStream>> {
        self.staging_stream.as_ref()
    }

    /// Synchronize the CUDA stream.
    pub fn synchronize(&self) -> Result<()> {
        self.nccl.synchronize()
    }

    // ========================================================================
    // Collective operations
    // ========================================================================

    /// Hierarchical allreduce in-place.
    ///
    /// # Safety
    /// `ptr` must be a valid GPU device pointer for `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn allreduce(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe { crate::collective::hierarchical_allreduce(self, ptr, count, dtype, op).await }
    }

    /// Hierarchical broadcast from a global root rank.
    ///
    /// # Safety
    /// `ptr` must be a valid GPU device pointer for `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn broadcast(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe { crate::collective::hierarchical_broadcast(self, ptr, count, dtype, root).await }
    }

    /// Hierarchical allgather.
    ///
    /// # Safety
    /// - `send_ptr`: `count * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr`: `count * world_size * dtype.size_in_bytes()` bytes.
    pub async unsafe fn allgather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        unsafe {
            crate::collective::hierarchical_allgather(self, send_ptr, recv_ptr, count, dtype).await
        }
    }

    /// Hierarchical reduce to a global root rank.
    ///
    /// # Safety
    /// `ptr` must be a valid GPU device pointer for `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn reduce(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        root: Rank,
    ) -> Result<()> {
        unsafe { crate::collective::hierarchical_reduce(self, ptr, count, dtype, op, root).await }
    }

    /// Hierarchical barrier.
    pub async fn barrier(&self) -> Result<()> {
        crate::collective::hierarchical_barrier(self).await
    }
}
