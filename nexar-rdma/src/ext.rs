//! Extension traits for `PeerConnection` that add RDMA capabilities.

use crate::rdma::{RdmaConnection, RdmaMemoryPool};
use futures::future::BoxFuture;
use nexar::PeerConnection;
use nexar::error::{NexarError, Result};
use nexar::transport::BulkTransport;
use std::sync::Arc;

/// RDMA state attached to a `PeerConnection` via the extensions slot.
///
/// Wrapped in `Arc` so it can be cloned out of the extension guard
/// without holding the `RwLockReadGuard` across `.await` points.
pub(crate) struct RdmaStateHolder(pub Arc<RdmaState>);

pub(crate) struct RdmaState {
    pub conn: std::sync::Mutex<RdmaConnection>,
    pub pool: Arc<RdmaMemoryPool>,
}

/// Extract the `Arc<RdmaState>` from a `PeerConnection`, dropping the guard immediately.
fn get_rdma(peer: &PeerConnection) -> Option<Arc<RdmaState>> {
    peer.extension::<RdmaStateHolder>()
        .map(|holder| Arc::clone(&holder.0))
}

/// Extension trait that adds RDMA bulk-send capabilities to `PeerConnection`.
///
/// Call `set_rdma` to attach RDMA state, then `send_raw_rdma` to send data
/// via RDMA with automatic QUIC fallback.
pub trait PeerConnectionRdmaExt {
    /// Attach an RDMA connection for bulk data offload.
    fn set_rdma(&self, rdma_conn: RdmaConnection, pool: Arc<RdmaMemoryPool>);

    /// Send raw bytes via RDMA. Falls back to QUIC if no RDMA state is attached.
    fn send_raw_rdma(&self, data: &[u8]) -> impl std::future::Future<Output = Result<()>> + Send;
}

/// `BulkTransport` implementation backed by RDMA.
struct RdmaBulkTransport(Arc<RdmaState>);

impl BulkTransport for RdmaBulkTransport {
    fn send_bulk<'a>(&'a self, data: &'a [u8]) -> BoxFuture<'a, Result<()>> {
        let rdma = Arc::clone(&self.0);
        Box::pin(async move { send_via_rdma(rdma, data).await })
    }

    fn recv_bulk<'a>(&'a self, expected_size: usize) -> BoxFuture<'a, Result<Vec<u8>>> {
        let rdma = Arc::clone(&self.0);
        Box::pin(async move {
            let mut pooled = rdma.pool.checkout()?;
            tokio::task::spawn_blocking(move || {
                let mut conn = rdma
                    .conn
                    .lock()
                    .map_err(|e| NexarError::device(format!("RDMA lock poisoned: {e}")))?;
                conn.recv(pooled.mr_mut(), 0)?;
                Ok(pooled[..expected_size].to_vec())
            })
            .await
            .map_err(|e| NexarError::device(format!("RDMA spawn_blocking: {e}")))?
        })
    }
}

impl PeerConnectionRdmaExt for PeerConnection {
    fn set_rdma(&self, rdma_conn: RdmaConnection, pool: Arc<RdmaMemoryPool>) {
        let state = Arc::new(RdmaState {
            conn: std::sync::Mutex::new(rdma_conn),
            pool,
        });
        self.add_extension(RdmaStateHolder(Arc::clone(&state)));
        // Register as BulkTransport so collectives auto-select RDMA.
        let bulk: Arc<dyn BulkTransport> = Arc::new(RdmaBulkTransport(state));
        self.add_extension(bulk);
    }

    async fn send_raw_rdma(&self, data: &[u8]) -> Result<()> {
        // Extract Arc and drop the extension guard before any .await.
        if let Some(rdma) = get_rdma(self) {
            return send_via_rdma(rdma, data).await;
        }
        // Fallback to QUIC.
        self.send_raw(data).await
    }
}

/// Send data via RDMA: copy into a registered MR, post send, wait for completion.
async fn send_via_rdma(rdma: Arc<RdmaState>, data: &[u8]) -> Result<()> {
    let mut pooled = rdma.pool.checkout()?;
    let len = data.len();
    pooled[..len].copy_from_slice(data);

    tokio::task::spawn_blocking(move || {
        let mut conn = rdma
            .conn
            .lock()
            .map_err(|e| NexarError::device(format!("RDMA lock poisoned: {e}")))?;
        conn.send(pooled.mr_mut(), 0)?;
        Ok::<(), NexarError>(())
    })
    .await
    .map_err(|e| NexarError::device(format!("RDMA spawn_blocking: {e}")))?
}

#[cfg(feature = "gpudirect")]
mod gpudirect_ext {
    use super::*;
    use crate::gpudirect::{GpuDirectPool, GpuDirectQp, PooledGpuMr};

    /// GPUDirect state attached to a `PeerConnection` via the extensions slot.
    pub(crate) struct GpuDirectStateHolder(pub Arc<GpuDirectState>);

    pub(crate) struct GpuDirectState {
        pub qp: std::sync::Mutex<GpuDirectQp>,
        pub pool: Arc<GpuDirectPool>,
    }

    fn get_gpudirect(peer: &PeerConnection) -> Option<Arc<GpuDirectState>> {
        peer.extension::<GpuDirectStateHolder>()
            .map(|holder| Arc::clone(&holder.0))
    }

    /// Extension trait for GPUDirect RDMA on `PeerConnection`.
    pub trait PeerConnectionGpuDirectExt: PeerConnectionRdmaExt {
        /// Attach a GPUDirect RDMA QP and buffer pool.
        fn set_gpudirect(&self, qp: GpuDirectQp, pool: Arc<GpuDirectPool>);

        /// Send directly from GPU memory via GPUDirect RDMA.
        ///
        /// Tiered fallback: GPUDirect → staged GPU→host → RDMA/QUIC.
        fn send_raw_gpu(
            &self,
            gpu_ptr: u64,
            size: usize,
        ) -> impl std::future::Future<Output = Result<()>> + Send;

        /// Receive directly into GPU memory via GPUDirect RDMA.
        fn recv_raw_gpu(
            &self,
            gpu_ptr: u64,
            size: usize,
        ) -> impl std::future::Future<Output = Result<()>> + Send;
    }

    impl PeerConnectionGpuDirectExt for PeerConnection {
        fn set_gpudirect(&self, qp: GpuDirectQp, pool: Arc<GpuDirectPool>) {
            self.add_extension(GpuDirectStateHolder(Arc::new(GpuDirectState {
                qp: std::sync::Mutex::new(qp),
                pool,
            })));
        }

        async fn send_raw_gpu(&self, gpu_ptr: u64, size: usize) -> Result<()> {
            // Extract Arc and drop extension guard before any .await.
            if let Some(gd) = get_gpudirect(self) {
                if let Some(pooled) = gd.pool.checkout() {
                    let mr_size = pooled.mr().size();
                    let mr_gpu_ptr = pooled.mr().gpu_ptr();
                    if mr_size >= size {
                        // Single chunk: D2D copy into MR, send.
                        if mr_gpu_ptr != gpu_ptr {
                            unsafe {
                                cudarc::driver::result::memcpy_dtod_sync(mr_gpu_ptr, gpu_ptr, size)
                                    .map_err(|e| {
                                        NexarError::device(format!(
                                            "GPUDirect D2D copy failed: {e}"
                                        ))
                                    })?;
                            }
                        }
                        return send_via_gpudirect(Arc::clone(&gd), pooled).await;
                    }
                    // Pipelined chunking: data exceeds MR size, send in MR-sized pieces
                    // via GPUDirect instead of bouncing the entire payload through the CPU.
                    let mut offset = 0usize;
                    while offset < size {
                        let chunk = std::cmp::min(mr_size, size - offset);
                        unsafe {
                            cudarc::driver::result::memcpy_dtod_sync(
                                mr_gpu_ptr,
                                gpu_ptr + offset as u64,
                                chunk,
                            )
                            .map_err(|e| {
                                NexarError::device(format!(
                                    "GPUDirect D2D copy (chunk at offset {offset}) failed: {e}"
                                ))
                            })?;
                        }
                        // Re-checkout isn't needed — we reuse the same pooled MR for each chunk.
                        send_via_gpudirect_sized(Arc::clone(&gd), &pooled, chunk).await?;
                        offset += chunk;
                    }
                    return Ok(());
                }
            }

            // Tier 2: Stage GPU→host, then use RDMA or QUIC.
            let host_data = crate::gpudirect::stage_gpu_to_host(gpu_ptr, size)?;
            self.send_raw_rdma(&host_data).await
        }

        async fn recv_raw_gpu(&self, gpu_ptr: u64, size: usize) -> Result<()> {
            if let Some(gd) = get_gpudirect(self) {
                if let Some(pooled) = gd.pool.checkout() {
                    let mr_size = pooled.mr().size();
                    let mr_gpu_ptr = pooled.mr().gpu_ptr();
                    if mr_size >= size {
                        // Single chunk: receive into MR, D2D copy to destination.
                        recv_via_gpudirect(Arc::clone(&gd), pooled).await?;

                        if mr_gpu_ptr != gpu_ptr {
                            unsafe {
                                cudarc::driver::result::memcpy_dtod_sync(gpu_ptr, mr_gpu_ptr, size)
                                    .map_err(|e| {
                                        NexarError::device(format!(
                                            "GPUDirect D2D copy failed: {e}"
                                        ))
                                    })?;
                            }
                        }
                        return Ok(());
                    }
                    // Pipelined chunking: receive in MR-sized pieces.
                    let mut offset = 0usize;
                    while offset < size {
                        let chunk = std::cmp::min(mr_size, size - offset);
                        recv_via_gpudirect_sized(Arc::clone(&gd), &pooled, chunk).await?;
                        unsafe {
                            cudarc::driver::result::memcpy_dtod_sync(
                                gpu_ptr + offset as u64,
                                mr_gpu_ptr,
                                chunk,
                            )
                            .map_err(|e| {
                                NexarError::device(format!(
                                    "GPUDirect D2D copy (chunk at offset {offset}) failed: {e}"
                                ))
                            })?;
                        }
                        offset += chunk;
                    }
                    return Ok(());
                }
            }

            Err(NexarError::device(
                "GPUDirect recv_raw_gpu: no suitable GPUDirect MR available; \
                 use recv_bytes() + stage_host_to_gpu() at the application layer",
            ))
        }
    }

    async fn send_via_gpudirect(gd: Arc<GpuDirectState>, pooled: PooledGpuMr) -> Result<()> {
        tokio::task::spawn_blocking(move || {
            let qp = gd
                .qp
                .lock()
                .map_err(|e| NexarError::device(format!("GPUDirect lock poisoned: {e}")))?;
            qp.send(pooled.mr(), 0)
        })
        .await
        .map_err(|e| NexarError::device(format!("GPUDirect spawn_blocking: {e}")))?
    }

    /// Send a chunk of size `chunk_size` from a pooled MR (which may be larger).
    /// The caller must have already D2D-copied the data into the MR.
    async fn send_via_gpudirect_sized(
        gd: Arc<GpuDirectState>,
        pooled: &PooledGpuMr,
        _chunk_size: usize,
    ) -> Result<()> {
        // The GpuMr is registered with the NIC at its full size; the QP send
        // posts the entire MR. The receiver knows the expected chunk size from
        // the protocol (total size / MR size). For chunked sends we reuse the
        // full MR — the receiver will read only `chunk_size` bytes.
        let mr_ptr = pooled.mr() as *const _ as usize;
        tokio::task::spawn_blocking(move || {
            // SAFETY: The PooledGpuMr is borrowed for the duration of this call,
            // and we only read the pointer to reconstruct a reference inside
            // the blocking task. The MR is valid because the caller holds it.
            let mr = unsafe { &*(mr_ptr as *const crate::gpudirect::GpuMr) };
            let qp = gd
                .qp
                .lock()
                .map_err(|e| NexarError::device(format!("GPUDirect lock poisoned: {e}")))?;
            qp.send(mr, 0)
        })
        .await
        .map_err(|e| NexarError::device(format!("GPUDirect spawn_blocking: {e}")))?
    }

    /// Receive a chunk into a pooled MR (which may be larger than the chunk).
    async fn recv_via_gpudirect_sized(
        gd: Arc<GpuDirectState>,
        pooled: &PooledGpuMr,
        _chunk_size: usize,
    ) -> Result<()> {
        let mr_ptr = pooled.mr() as *const _ as usize;
        tokio::task::spawn_blocking(move || {
            let mr = unsafe { &*(mr_ptr as *const crate::gpudirect::GpuMr) };
            let qp = gd
                .qp
                .lock()
                .map_err(|e| NexarError::device(format!("GPUDirect lock poisoned: {e}")))?;
            qp.recv(mr, 0)
        })
        .await
        .map_err(|e| NexarError::device(format!("GPUDirect spawn_blocking: {e}")))?
    }

    async fn recv_via_gpudirect(gd: Arc<GpuDirectState>, pooled: PooledGpuMr) -> Result<()> {
        tokio::task::spawn_blocking(move || {
            let qp = gd
                .qp
                .lock()
                .map_err(|e| NexarError::device(format!("GPUDirect lock poisoned: {e}")))?;
            qp.recv(pooled.mr(), 0)
        })
        .await
        .map_err(|e| NexarError::device(format!("GPUDirect spawn_blocking: {e}")))?
    }
}

#[cfg(feature = "gpudirect")]
pub use gpudirect_ext::PeerConnectionGpuDirectExt;
