//! Safe typed-buffer wrappers for collective operations.
//!
//! These methods accept [`BufferRef<Host>`] / [`BufferRef<Device>`] instead
//! of raw `u64` pointers, providing compile-time memory space safety.
//! The underlying unsafe collective implementations are unchanged.

use crate::compression::Compressor;
use crate::error::Result;
use crate::memory::{BufferRef, Device, Host};
use crate::types::{DataType, Rank, ReduceOp};

use super::NexarClient;

// ── Host buffer collectives ─────────────────────────────────────────

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

    /// All-to-all on host buffers.
    pub async fn all_to_all_host(
        &self,
        send_buf: &BufferRef<Host>,
        recv_buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        unsafe {
            self.all_to_all(send_buf.as_u64(), recv_buf.as_u64(), count, dtype)
                .await
        }
    }

    /// Gather to root on host buffers.
    pub async fn gather_host(
        &self,
        send_buf: &BufferRef<Host>,
        recv_buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe {
            self.gather(send_buf.as_u64(), recv_buf.as_u64(), count, dtype, root)
                .await
        }
    }

    /// Scatter from root on host buffers.
    pub async fn scatter_host(
        &self,
        send_buf: &BufferRef<Host>,
        recv_buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe {
            self.scatter(send_buf.as_u64(), recv_buf.as_u64(), count, dtype, root)
                .await
        }
    }

    /// Inclusive prefix scan on a host buffer.
    pub async fn scan_host(
        &self,
        buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe { self.scan(buf.as_u64(), count, dtype, op).await }
    }

    /// Exclusive prefix scan on a host buffer.
    pub async fn exclusive_scan_host(
        &self,
        buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe { self.exclusive_scan(buf.as_u64(), count, dtype, op).await }
    }

    /// Bucketed allreduce on host buffers.
    ///
    /// Fuses multiple small host buffers into a single allreduce. Each
    /// `BufferRef<Host>` must hold exactly `count * dtype.size_in_bytes()`
    /// bytes for its corresponding element count.
    ///
    /// This is host-only by design — no `_device` variant exists because
    /// the bucketed algorithm operates on host memory. GPU users should
    /// use `nexar-nccl`'s on-device bucketed operations.
    pub async fn all_reduce_bucketed_host(
        &self,
        entries: &[(BufferRef<Host>, usize)],
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        let raw: Vec<(u64, usize)> = entries
            .iter()
            .map(|(b, count)| (b.as_u64(), *count))
            .collect();
        unsafe { self.all_reduce_bucketed(&raw, dtype, op).await }
    }

    /// Compressed allreduce on a host buffer.
    pub async fn all_reduce_compressed_host(
        &self,
        buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        compressor: &dyn Compressor,
        residual: &mut [u8],
    ) -> Result<()> {
        unsafe {
            self.all_reduce_compressed(buf.as_u64(), count, dtype, op, compressor, residual)
                .await
        }
    }
}

// ── Device buffer collectives ───────────────────────────────────────

impl NexarClient {
    /// AllReduce in-place on a device buffer.
    pub async fn all_reduce_device(
        &self,
        buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe { self.all_reduce(buf.as_u64(), count, dtype, op).await }
    }

    /// Broadcast from root on a device buffer.
    pub async fn broadcast_device(
        &self,
        buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe { self.broadcast(buf.as_u64(), count, dtype, root).await }
    }

    /// AllGather on device buffers.
    pub async fn all_gather_device(
        &self,
        send_buf: &BufferRef<Device>,
        recv_buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        unsafe {
            self.all_gather(send_buf.as_u64(), recv_buf.as_u64(), count, dtype)
                .await
        }
    }

    /// ReduceScatter on device buffers.
    pub async fn reduce_scatter_device(
        &self,
        send_buf: &BufferRef<Device>,
        recv_buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe {
            self.reduce_scatter(send_buf.as_u64(), recv_buf.as_u64(), count, dtype, op)
                .await
        }
    }

    /// Reduce to root on a device buffer.
    pub async fn reduce_device(
        &self,
        buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        root: Rank,
    ) -> Result<()> {
        unsafe { self.reduce(buf.as_u64(), count, dtype, op, root).await }
    }

    /// All-to-all on device buffers.
    pub async fn all_to_all_device(
        &self,
        send_buf: &BufferRef<Device>,
        recv_buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        unsafe {
            self.all_to_all(send_buf.as_u64(), recv_buf.as_u64(), count, dtype)
                .await
        }
    }

    /// Gather to root on device buffers.
    pub async fn gather_device(
        &self,
        send_buf: &BufferRef<Device>,
        recv_buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe {
            self.gather(send_buf.as_u64(), recv_buf.as_u64(), count, dtype, root)
                .await
        }
    }

    /// Scatter from root on device buffers.
    pub async fn scatter_device(
        &self,
        send_buf: &BufferRef<Device>,
        recv_buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        unsafe {
            self.scatter(send_buf.as_u64(), recv_buf.as_u64(), count, dtype, root)
                .await
        }
    }

    /// Inclusive prefix scan on a device buffer.
    pub async fn scan_device(
        &self,
        buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe { self.scan(buf.as_u64(), count, dtype, op).await }
    }

    /// Exclusive prefix scan on a device buffer.
    pub async fn exclusive_scan_device(
        &self,
        buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe { self.exclusive_scan(buf.as_u64(), count, dtype, op).await }
    }

    /// Compressed allreduce on a device buffer.
    pub async fn all_reduce_compressed_device(
        &self,
        buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        compressor: &dyn Compressor,
        residual: &mut [u8],
    ) -> Result<()> {
        unsafe {
            self.all_reduce_compressed(buf.as_u64(), count, dtype, op, compressor, residual)
                .await
        }
    }
}
