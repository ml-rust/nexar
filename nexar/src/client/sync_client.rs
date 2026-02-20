use crate::device::DeviceAdapter;
use crate::error::Result;
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

impl SyncClient {
    /// Bootstrap a local cluster and return sync clients for each rank.
    pub fn bootstrap_local(world_size: u32, adapter: Arc<dyn DeviceAdapter>) -> Result<Vec<Self>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| crate::error::NexarError::transport(format!("tokio runtime: {e}")))?;

        let clients = rt.block_on(super::NexarClient::bootstrap_local(world_size, adapter))?;

        // Each SyncClient needs its own runtime since `block_on` is exclusive.
        // We keep the first runtime for the first client, create new ones for the rest.
        let mut sync_clients = Vec::new();
        let mut iter = clients.into_iter();

        if let Some(first) = iter.next() {
            sync_clients.push(SyncClient { inner: first, rt });
        }

        for client in iter {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| crate::error::NexarError::transport(format!("tokio runtime: {e}")))?;
            sync_clients.push(SyncClient { inner: client, rt });
        }

        Ok(sync_clients)
    }

    /// Wrap an existing async client with a new tokio runtime.
    pub fn from_async(inner: super::NexarClient) -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| crate::error::NexarError::transport(format!("tokio runtime: {e}")))?;
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

    /// AllReduce in-place.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn all_reduce(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.all_reduce(ptr, count, dtype, op) })
    }

    /// Bucketed allreduce: fuse multiple small tensors into one allreduce.
    ///
    /// # Safety
    /// Each `(ptr, count)` entry must point to at least `count * dtype.size_in_bytes()`
    /// valid bytes on the device.
    pub unsafe fn all_reduce_bucketed(
        &self,
        entries: &[(u64, usize)],
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.all_reduce_bucketed(entries, dtype, op) })
    }

    /// Allreduce via reduce-scatter + allgather decomposition.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn all_reduce_rs_ag(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.all_reduce_rs_ag(ptr, count, dtype, op) })
    }

    /// Broadcast from root.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn broadcast(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.broadcast(ptr, count, dtype, root) })
    }

    /// AllGather.
    ///
    /// # Safety
    /// - `send_ptr` must point to at least `count * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
    pub unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.all_gather(send_ptr, recv_ptr, count, dtype) })
    }

    /// ReduceScatter.
    ///
    /// # Safety
    /// - `send_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr` must point to at least `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        self.rt.block_on(unsafe {
            self.inner
                .reduce_scatter(send_ptr, recv_ptr, count, dtype, op)
        })
    }

    /// Reduce to root.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn reduce(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        root: Rank,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.reduce(ptr, count, dtype, op, root) })
    }

    /// All-to-all.
    ///
    /// # Safety
    /// - `send_ptr`: `count * world_size * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr`: `count * world_size * dtype.size_in_bytes()` bytes.
    pub unsafe fn all_to_all(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.all_to_all(send_ptr, recv_ptr, count, dtype) })
    }

    /// Gather to root.
    ///
    /// # Safety
    /// - `send_ptr`: at least `count * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr`: at least `count * world_size * dtype.size_in_bytes()` bytes on root.
    pub unsafe fn gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.gather(send_ptr, recv_ptr, count, dtype, root) })
    }

    /// Scatter from root.
    ///
    /// # Safety
    /// - `send_ptr`: at least `count * world_size * dtype.size_in_bytes()` bytes on root.
    /// - `recv_ptr`: at least `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DataType,
        root: Rank,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.scatter(send_ptr, recv_ptr, count, dtype, root) })
    }

    /// Exclusive prefix scan.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn exclusive_scan(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.exclusive_scan(ptr, count, dtype, op) })
    }

    /// Inclusive prefix scan.
    ///
    /// # Safety
    /// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn scan(&self, ptr: u64, count: usize, dtype: DataType, op: ReduceOp) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.scan(ptr, count, dtype, op) })
    }

    /// Point-to-point send.
    ///
    /// # Safety
    /// `ptr` must be valid for `size` bytes.
    pub unsafe fn send(&self, ptr: u64, size: usize, dest: Rank, tag: u32) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.send(ptr, size, dest, tag) })
    }

    /// Point-to-point receive.
    ///
    /// # Safety
    /// `ptr` must be valid for `size` bytes.
    pub unsafe fn recv(&self, ptr: u64, size: usize, src: Rank, tag: u32) -> Result<()> {
        self.rt
            .block_on(unsafe { self.inner.recv(ptr, size, src, tag) })
    }

    /// Barrier.
    pub fn barrier(&self) -> Result<()> {
        self.rt.block_on(self.inner.barrier())
    }

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
