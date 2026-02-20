use crate::collective::CollectiveHandle;
use crate::compression::Compressor;
use std::sync::Arc;

use super::NexarClient;

/// Wrapper to send a raw `*mut u8` across thread boundaries.
///
/// # Safety
/// The caller must ensure the pointer remains valid and exclusively
/// borrowed for the lifetime of the future that uses it.
struct SendMutSlice {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for SendMutSlice {}

impl SendMutSlice {
    /// Reconstruct the mutable slice.
    ///
    /// # Safety
    /// The pointer must still be valid and exclusively borrowed.
    unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl NexarClient {
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

    /// Non-blocking broadcast.
    ///
    /// # Safety
    /// `ptr` must remain valid until the handle is awaited.
    pub unsafe fn broadcast_nb(
        self: &Arc<Self>,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        root: crate::types::Rank,
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
        root: crate::types::Rank,
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
        root: crate::types::Rank,
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
        root: crate::types::Rank,
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

    /// Non-blocking barrier.
    pub fn barrier_nb(self: &Arc<Self>) -> CollectiveHandle {
        let client = Arc::clone(self);
        CollectiveHandle::spawn(async move {
            crate::collective::barrier(&client, client.config.barrier_timeout).await
        })
    }

    /// Non-blocking bucketed allreduce.
    ///
    /// # Safety
    /// All entry pointers must remain valid until the handle is awaited.
    pub unsafe fn all_reduce_bucketed_nb(
        self: &Arc<Self>,
        entries: Vec<(u64, usize)>,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
    ) -> CollectiveHandle {
        let client = Arc::clone(self);
        let tag = client.next_collective_tag();
        CollectiveHandle::spawn(async move {
            unsafe {
                crate::collective::allreduce_bucketed_with_tag(
                    &client,
                    &entries,
                    dtype,
                    op,
                    Some(tag),
                )
                .await
            }
        })
    }

    /// Non-blocking RS+AG allreduce.
    ///
    /// # Safety
    /// `ptr` must remain valid until the handle is awaited.
    pub unsafe fn all_reduce_rs_ag_nb(
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
                crate::collective::rs_ag_allreduce_with_tag(
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

    /// Non-blocking compressed allreduce.
    ///
    /// # Safety
    /// - `ptr` must remain valid until the handle is awaited.
    /// - `residual` must remain valid and exclusively borrowed until the handle is awaited.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn all_reduce_compressed_nb(
        self: &Arc<Self>,
        ptr: u64,
        count: usize,
        dtype: crate::types::DataType,
        op: crate::types::ReduceOp,
        compressor: Arc<dyn Compressor>,
        residual: *mut u8,
        residual_len: usize,
    ) -> CollectiveHandle {
        let tag = self.next_collective_tag();
        let client = Arc::clone(self);
        let mut residual_buf = SendMutSlice {
            ptr: residual,
            len: residual_len,
        };
        CollectiveHandle::spawn(async move {
            // SAFETY: caller guarantees residual is valid and exclusively borrowed.
            let residual_slice = unsafe { residual_buf.as_mut_slice() };
            unsafe {
                crate::collective::ring_allreduce_compressed(
                    &client,
                    ptr,
                    count,
                    dtype,
                    op,
                    compressor.as_ref(),
                    residual_slice,
                    Some(tag),
                )
                .await
            }
        })
    }
}
