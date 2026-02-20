use crate::compression::Compressor;
use crate::error::Result;
use crate::types::Rank;

use super::NexarClient;

impl NexarClient {
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

    /// Gather: root collects `count` elements from each rank.
    ///
    /// # Safety
    /// - `send_ptr`: at least `count * dtype.size_in_bytes()` bytes on all ranks.
    /// - `recv_ptr`: at least `count * world_size * dtype.size_in_bytes()` bytes on root.
    pub async unsafe fn gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe { crate::collective::gather(self, send_ptr, recv_ptr, count, dtype, root).await }
    }

    /// Scatter: root distributes chunks to each rank.
    ///
    /// # Safety
    /// - `send_ptr`: at least `count * world_size * dtype.size_in_bytes()` bytes on root.
    /// - `recv_ptr`: at least `count * dtype.size_in_bytes()` bytes on all ranks.
    pub async unsafe fn scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe { crate::collective::scatter(self, send_ptr, recv_ptr, count, dtype, root).await }
    }

    /// Exclusive prefix scan: rank `i` holds the reduction of ranks 0..i.
    /// Rank 0 gets the identity element.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn exclusive_scan(
        &self,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> Result<()> {
        unsafe { crate::collective::exclusive_scan(self, ptr, count, dtype, op).await }
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
    pub async fn barrier(&self) -> Result<()> {
        crate::collective::barrier(self, self.config.barrier_timeout).await
    }

    /// Bucketed allreduce: fuse multiple small tensors into one allreduce.
    ///
    /// Each entry is `(ptr, element_count)`. All entries share the same dtype
    /// and reduction operation.
    ///
    /// # Safety
    /// Each `(ptr, count)` entry must point to at least `count * dtype.size_in_bytes()`
    /// valid bytes on the device.
    pub async unsafe fn all_reduce_bucketed(
        &self,
        entries: &[(u64, usize)],
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> Result<()> {
        unsafe { crate::collective::allreduce_bucketed(self, entries, dtype, op).await }
    }

    /// Allreduce via reduce-scatter + allgather decomposition.
    ///
    /// Equivalent to `all_reduce` but decomposed into two phases. Useful for
    /// ZeRO-style sharding where intermediate results are needed.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub async unsafe fn all_reduce_rs_ag(
        &self,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> Result<()> {
        unsafe { crate::collective::rs_ag_allreduce(self, ptr, count, dtype, op).await }
    }

    /// Compressed allreduce: bandwidth-efficient allreduce with gradient compression.
    ///
    /// # Safety
    /// - `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    /// - `residual` must be at least `count * dtype.size_in_bytes()` bytes,
    ///   zero-initialized on the first call, and preserved across calls.
    pub async unsafe fn all_reduce_compressed(
        &self,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
        compressor: &dyn Compressor,
        residual: &mut [u8],
    ) -> Result<()> {
        unsafe {
            crate::collective::ring_allreduce_compressed(
                self, ptr, count, dtype, op, compressor, residual,
            )
            .await
        }
    }
}
