use crate::compression::Compressor;
use crate::device::DeviceAdapter;
use crate::error::Result;
use crate::memory::{BufferRef, Device, Host};
use crate::rpc::registry::RpcHandler;
use crate::types::{DataType, Rank, ReduceOp};
use std::sync::Arc;

/// Blocking wrapper around [`NexarClient`](super::NexarClient).
///
/// Owns a `tokio::runtime::Runtime` and calls `block_on()` for each operation.
pub struct SyncClient {
    inner: super::NexarClient,
    rt: tokio::runtime::Runtime,
}

/// Generate a blocking wrapper method that delegates to an unsafe async method on `inner`.
macro_rules! sync_unsafe {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(&self $(, $arg:ident: $ty:ty)*) -> Result<()>
    ) => {
        $(#[$meta])*
        $vis unsafe fn $name(&self $(, $arg: $ty)*) -> Result<()> {
            self.rt.block_on(unsafe { self.inner.$name($($arg),*) })
        }
    };
}

/// Generate a blocking wrapper for a typed buffer method (Host or Device).
macro_rules! sync_typed {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(&self, buf: &mut BufferRef<$loc:ty>
            $(, $arg:ident: $ty:ty)*) -> Result<()>
    ) => {
        $(#[$meta])*
        $vis fn $name(&self, buf: &mut BufferRef<$loc> $(, $arg: $ty)*) -> Result<()> {
            self.rt.block_on(self.inner.$name(buf $(, $arg)*))
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(&self,
            send_buf: &BufferRef<$loc:ty>,
            recv_buf: &mut BufferRef<$loc2:ty>
            $(, $arg:ident: $ty:ty)*) -> Result<()>
    ) => {
        $(#[$meta])*
        $vis fn $name(
            &self,
            send_buf: &BufferRef<$loc>,
            recv_buf: &mut BufferRef<$loc2>
            $(, $arg: $ty)*
        ) -> Result<()> {
            self.rt.block_on(self.inner.$name(send_buf, recv_buf $(, $arg)*))
        }
    };
}

impl SyncClient {
    /// Bootstrap a local cluster and return sync clients for each rank.
    pub fn bootstrap_local(world_size: u32, adapter: Arc<dyn DeviceAdapter>) -> Result<Vec<Self>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| crate::error::NexarError::transport_with_source("tokio runtime", e))?;

        let clients = rt.block_on(super::NexarClient::bootstrap_local(world_size, adapter))?;

        let mut sync_clients = Vec::new();
        let mut iter = clients.into_iter();

        if let Some(first) = iter.next() {
            sync_clients.push(SyncClient { inner: first, rt });
        }

        for client in iter {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| crate::error::NexarError::transport_with_source("tokio runtime", e))?;
            sync_clients.push(SyncClient { inner: client, rt });
        }

        Ok(sync_clients)
    }

    /// Wrap an existing async client with a new tokio runtime.
    pub fn from_async(inner: super::NexarClient) -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| crate::error::NexarError::transport_with_source("tokio runtime", e))?;
        Ok(Self { inner, rt })
    }

    /// This client's rank within its communicator group (0-indexed).
    pub fn rank(&self) -> Rank {
        self.inner.rank()
    }

    /// Total number of ranks in the communicator group.
    pub fn world_size(&self) -> u32 {
        self.inner.world_size()
    }

    // ── Raw pointer collectives ─────────────────────────────────────

    sync_unsafe! {
        /// AllReduce in-place.
        ///
        /// # Safety
        /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
        pub fn all_reduce(&self, ptr: u64, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_unsafe! {
        /// Bucketed allreduce: fuse multiple small tensors into one allreduce.
        ///
        /// # Safety
        /// Each `(ptr, count)` entry must point to valid memory.
        pub fn all_reduce_bucketed(&self, entries: &[(u64, usize)], dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_unsafe! {
        /// Allreduce via reduce-scatter + allgather decomposition.
        ///
        /// # Safety
        /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
        pub fn all_reduce_rs_ag(&self, ptr: u64, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_unsafe! {
        /// Broadcast from root.
        ///
        /// # Safety
        /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
        pub fn broadcast(&self, ptr: u64, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_unsafe! {
        /// AllGather.
        ///
        /// # Safety
        /// Both pointers must be valid for the required sizes.
        pub fn all_gather(&self, send_ptr: u64, recv_ptr: u64, count: usize, dtype: DataType) -> Result<()>
    }

    sync_unsafe! {
        /// ReduceScatter.
        ///
        /// # Safety
        /// Both pointers must be valid for the required sizes.
        pub fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_unsafe! {
        /// Reduce to root.
        ///
        /// # Safety
        /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
        pub fn reduce(&self, ptr: u64, count: usize, dtype: DataType, op: ReduceOp, root: Rank) -> Result<()>
    }

    sync_unsafe! {
        /// All-to-all.
        ///
        /// # Safety
        /// Both pointers must be valid for `count * world_size * dtype.size_in_bytes()` bytes.
        pub fn all_to_all(&self, send_ptr: u64, recv_ptr: u64, count: usize, dtype: DataType) -> Result<()>
    }

    sync_unsafe! {
        /// Gather to root.
        ///
        /// # Safety
        /// Both pointers must be valid for the required sizes.
        pub fn gather(&self, send_ptr: u64, recv_ptr: u64, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_unsafe! {
        /// Scatter from root.
        ///
        /// # Safety
        /// Both pointers must be valid for the required sizes.
        pub fn scatter(&self, send_ptr: u64, recv_ptr: u64, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_unsafe! {
        /// Exclusive prefix scan.
        ///
        /// # Safety
        /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
        pub fn exclusive_scan(&self, ptr: u64, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_unsafe! {
        /// Inclusive prefix scan.
        ///
        /// # Safety
        /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
        pub fn scan(&self, ptr: u64, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_unsafe! {
        /// Point-to-point send.
        ///
        /// # Safety
        /// `ptr` must be valid for `size` bytes.
        pub fn send(&self, ptr: u64, size: usize, dest: Rank, tag: u32) -> Result<()>
    }

    sync_unsafe! {
        /// Point-to-point receive.
        ///
        /// # Safety
        /// `ptr` must be valid for `size` bytes.
        pub fn recv(&self, ptr: u64, size: usize, src: Rank, tag: u32) -> Result<()>
    }

    /// Barrier.
    pub fn barrier(&self) -> Result<()> {
        self.rt.block_on(self.inner.barrier())
    }

    // ── Typed host buffer collectives ───────────────────────────────

    sync_typed! {
        /// AllReduce in-place on a host buffer.
        pub fn all_reduce_host(&self, buf: &mut BufferRef<Host>, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_typed! {
        /// Broadcast from root on a host buffer.
        pub fn broadcast_host(&self, buf: &mut BufferRef<Host>, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_typed! {
        /// AllGather on host buffers.
        pub fn all_gather_host(&self, send_buf: &BufferRef<Host>, recv_buf: &mut BufferRef<Host>, count: usize, dtype: DataType) -> Result<()>
    }

    sync_typed! {
        /// ReduceScatter on host buffers.
        pub fn reduce_scatter_host(&self, send_buf: &BufferRef<Host>, recv_buf: &mut BufferRef<Host>, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_typed! {
        /// Reduce to root on a host buffer.
        pub fn reduce_host(&self, buf: &mut BufferRef<Host>, count: usize, dtype: DataType, op: ReduceOp, root: Rank) -> Result<()>
    }

    sync_typed! {
        /// All-to-all on host buffers.
        pub fn all_to_all_host(&self, send_buf: &BufferRef<Host>, recv_buf: &mut BufferRef<Host>, count: usize, dtype: DataType) -> Result<()>
    }

    sync_typed! {
        /// Gather to root on host buffers.
        pub fn gather_host(&self, send_buf: &BufferRef<Host>, recv_buf: &mut BufferRef<Host>, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_typed! {
        /// Scatter from root on host buffers.
        pub fn scatter_host(&self, send_buf: &BufferRef<Host>, recv_buf: &mut BufferRef<Host>, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_typed! {
        /// Inclusive prefix scan on a host buffer.
        pub fn scan_host(&self, buf: &mut BufferRef<Host>, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_typed! {
        /// Exclusive prefix scan on a host buffer.
        pub fn exclusive_scan_host(&self, buf: &mut BufferRef<Host>, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    /// Compressed allreduce on a host buffer.
    pub fn all_reduce_compressed_host(
        &self,
        buf: &mut BufferRef<Host>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        compressor: &dyn Compressor,
        residual: &mut [u8],
    ) -> Result<()> {
        self.rt.block_on(
            self.inner
                .all_reduce_compressed_host(buf, count, dtype, op, compressor, residual),
        )
    }

    // ── Typed device buffer collectives ─────────────────────────────

    sync_typed! {
        /// AllReduce in-place on a device buffer.
        pub fn all_reduce_device(&self, buf: &mut BufferRef<Device>, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_typed! {
        /// Broadcast from root on a device buffer.
        pub fn broadcast_device(&self, buf: &mut BufferRef<Device>, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_typed! {
        /// AllGather on device buffers.
        pub fn all_gather_device(&self, send_buf: &BufferRef<Device>, recv_buf: &mut BufferRef<Device>, count: usize, dtype: DataType) -> Result<()>
    }

    sync_typed! {
        /// ReduceScatter on device buffers.
        pub fn reduce_scatter_device(&self, send_buf: &BufferRef<Device>, recv_buf: &mut BufferRef<Device>, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_typed! {
        /// Reduce to root on a device buffer.
        pub fn reduce_device(&self, buf: &mut BufferRef<Device>, count: usize, dtype: DataType, op: ReduceOp, root: Rank) -> Result<()>
    }

    sync_typed! {
        /// All-to-all on device buffers.
        pub fn all_to_all_device(&self, send_buf: &BufferRef<Device>, recv_buf: &mut BufferRef<Device>, count: usize, dtype: DataType) -> Result<()>
    }

    sync_typed! {
        /// Gather to root on device buffers.
        pub fn gather_device(&self, send_buf: &BufferRef<Device>, recv_buf: &mut BufferRef<Device>, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_typed! {
        /// Scatter from root on device buffers.
        pub fn scatter_device(&self, send_buf: &BufferRef<Device>, recv_buf: &mut BufferRef<Device>, count: usize, dtype: DataType, root: Rank) -> Result<()>
    }

    sync_typed! {
        /// Inclusive prefix scan on a device buffer.
        pub fn scan_device(&self, buf: &mut BufferRef<Device>, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    sync_typed! {
        /// Exclusive prefix scan on a device buffer.
        pub fn exclusive_scan_device(&self, buf: &mut BufferRef<Device>, count: usize, dtype: DataType, op: ReduceOp) -> Result<()>
    }

    /// Compressed allreduce on a device buffer.
    pub fn all_reduce_compressed_device(
        &self,
        buf: &mut BufferRef<Device>,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        compressor: &dyn Compressor,
        residual: &mut [u8],
    ) -> Result<()> {
        self.rt.block_on(
            self.inner
                .all_reduce_compressed_device(buf, count, dtype, op, compressor, residual),
        )
    }

    // ── RPC ─────────────────────────────────────────────────────────

    /// Register an RPC handler for a function ID.
    pub fn register_rpc(&self, fn_id: u16, handler: RpcHandler) {
        self.rt.block_on(self.inner.register_rpc(fn_id, handler))
    }

    /// Call a remote function on the target rank and wait for the response.
    pub fn rpc(&self, target: Rank, fn_id: u16, args: &[u8]) -> Result<Vec<u8>> {
        self.rt.block_on(self.inner.rpc(target, fn_id, args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuAdapter;

    #[test]
    fn test_sync_client_single_node() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = SyncClient::bootstrap_local(1, adapter).unwrap();
        assert_eq!(clients.len(), 1);
        assert_eq!(clients[0].rank(), 0);
        assert_eq!(clients[0].world_size(), 1);
    }
}
