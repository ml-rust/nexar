//! Safe typed-buffer wrappers for collective operations.
//!
//! These methods accept [`BufferRef<Host>`] / [`BufferRef<Device>`] instead
//! of raw `u64` pointers, providing compile-time memory space safety.
//! The underlying unsafe collective implementations are unchanged.

use crate::error::Result;
use crate::memory::{BufferRef, Host};
use crate::types::{DataType, Rank, ReduceOp};

use super::NexarClient;

impl NexarClient {
    /// AllReduce in-place on a host buffer.
    pub async fn all_reduce_host(
        &self,
        buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe { self.all_reduce(buf.as_u64(), count, dtype, op).await }
    }

    /// Broadcast from root on a host buffer.
    pub async fn broadcast_host(
        &self,
        buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe { self.broadcast(buf.as_u64(), count, dtype, root).await }
    }

    /// AllGather on host buffers.
    pub async fn all_gather_host(
        &self,
        send_buf: &BufferRef<Host>,
        recv_buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        unsafe {
            self.all_gather(send_buf.as_u64(), recv_buf.as_u64(), count, dtype)
                .await
        }
    }

    /// ReduceScatter on host buffers.
    pub async fn reduce_scatter_host(
        &self,
        send_buf: &BufferRef<Host>,
        recv_buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe {
            self.reduce_scatter(send_buf.as_u64(), recv_buf.as_u64(), count, dtype, op)
                .await
        }
    }

    /// Reduce to root on a host buffer.
    pub async fn reduce_host(
        &self,
        buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        root: Rank,
    ) -> Result<()> {
        unsafe { self.reduce(buf.as_u64(), count, dtype, op, root).await }
    }
}
