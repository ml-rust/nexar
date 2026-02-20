use crate::collective::CollectiveHandle;
use crate::compression::Compressor;
use crate::error::Result;
use crate::types::Rank;
use std::sync::Arc;

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

    /// Non-blocking allreduce. Returns a handle that can be awaited later.
    ///
    /// # Safety
    /// `ptr` must remain valid until the handle is awaited.
    pub unsafe fn all_reduce_nb(
        self: &Arc<Self>,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::ring_allreduce_with_tag(
                    &client,
                    ptr,
                    count,
                    dtype,
                    op,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking broadcast.
    ///
    /// # Safety
    /// `ptr` must remain valid until the handle is awaited.
    pub unsafe fn broadcast_nb(
        self: &Arc<Self>,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        root: Rank,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::tree_broadcast_with_tag(
                    &client,
                    ptr,
                    count,
                    dtype,
                    root,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking allgather.
    ///
    /// # Safety
    /// Both pointers must remain valid until the handle is awaited.
    pub unsafe fn all_gather_nb(
        self: &Arc<Self>,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::ring_allgather_with_tag(
                    &client,
                    send_ptr,
                    recv_ptr,
                    count,
                    dtype,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking reduce-scatter.
    ///
    /// # Safety
    /// Both pointers must remain valid until the handle is awaited.
    pub unsafe fn reduce_scatter_nb(
        self: &Arc<Self>,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::ring_reduce_scatter_with_tag(
                    &client,
                    send_ptr,
                    recv_ptr,
                    count,
                    dtype,
                    op,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking reduce.
    ///
    /// # Safety
    /// `ptr` must remain valid until the handle is awaited.
    pub unsafe fn reduce_nb(
        self: &Arc<Self>,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
        root: Rank,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::tree_reduce_with_tag(
                    &client,
                    ptr,
                    count,
                    dtype,
                    op,
                    root,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking all-to-all.
    ///
    /// # Safety
    /// Both pointers must remain valid until the handle is awaited.
    pub unsafe fn all_to_all_nb(
        self: &Arc<Self>,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::alltoall_with_tag(
                    &client,
                    send_ptr,
                    recv_ptr,
                    count,
                    dtype,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking gather.
    ///
    /// # Safety
    /// Both pointers must remain valid until the handle is awaited.
    pub unsafe fn gather_nb(
        self: &Arc<Self>,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        root: Rank,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::gather_with_tag(
                    &client,
                    send_ptr,
                    recv_ptr,
                    count,
                    dtype,
                    root,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking scatter.
    ///
    /// # Safety
    /// Both pointers must remain valid until the handle is awaited.
    pub unsafe fn scatter_nb(
        self: &Arc<Self>,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        root: Rank,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::scatter_with_tag(
                    &client,
                    send_ptr,
                    recv_ptr,
                    count,
                    dtype,
                    root,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking exclusive scan.
    ///
    /// # Safety
    /// `ptr` must remain valid until the handle is awaited.
    pub unsafe fn exclusive_scan_nb(
        self: &Arc<Self>,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::exclusive_scan_with_tag(
                    &client,
                    ptr,
                    count,
                    dtype,
                    op,
                    Some(tag),
                )
                .await
            }
        })
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

    /// Non-blocking inclusive scan.
    ///
    /// # Safety
    /// `ptr` must remain valid until the handle is awaited.
    pub unsafe fn scan_nb(
        self: &Arc<Self>,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::inclusive_scan_with_tag(
                    &client,
                    ptr,
                    count,
                    dtype,
                    op,
                    Some(tag),
                )
                .await
            }
        })
    }

    /// Barrier: block until all ranks reach this point.
    pub async fn barrier(&self) -> Result<()> {
        crate::collective::barrier(self, std::time::Duration::from_secs(30)).await
    }

    /// Non-blocking barrier.
    pub fn barrier_nb(self: &Arc<Self>) -> CollectiveHandle {
        let client = Arc::clone(self);
        CollectiveHandle::spawn(async move {
            crate::collective::barrier(&client, std::time::Duration::from_secs(30)).await
        })
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
