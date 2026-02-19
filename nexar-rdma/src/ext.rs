//! Extension traits for `PeerConnection` that add RDMA capabilities.

use crate::rdma::{RdmaConnection, RdmaMemoryPool};
use nexar::PeerConnection;
use nexar::error::{NexarError, Result};
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

impl PeerConnectionRdmaExt for PeerConnection {
    fn set_rdma(&self, rdma_conn: RdmaConnection, pool: Arc<RdmaMemoryPool>) {
        self.add_extension(RdmaStateHolder(Arc::new(RdmaState {
            conn: std::sync::Mutex::new(rdma_conn),
            pool,
        })));
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
            .map_err(|e| NexarError::DeviceError(format!("RDMA lock poisoned: {e}")))?;
        conn.send(pooled.mr_mut(), 0)?;
        Ok::<(), NexarError>(())
    })
    .await
    .map_err(|e| NexarError::DeviceError(format!("RDMA spawn_blocking: {e}")))?
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
                        if mr_gpu_ptr != gpu_ptr {
                            unsafe {
                                cudarc::driver::result::memcpy_dtod_sync(mr_gpu_ptr, gpu_ptr, size)
                                    .map_err(|e| {
                                        NexarError::DeviceError(format!(
                                            "GPUDirect D2D copy failed: {e}"
                                        ))
                                    })?;
                            }
                        }
                        return send_via_gpudirect(Arc::clone(&gd), pooled).await;
                    }
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
                        recv_via_gpudirect(Arc::clone(&gd), pooled).await?;

                        if mr_gpu_ptr != gpu_ptr {
                            unsafe {
                                cudarc::driver::result::memcpy_dtod_sync(gpu_ptr, mr_gpu_ptr, size)
                                    .map_err(|e| {
                                        NexarError::DeviceError(format!(
                                            "GPUDirect D2D copy failed: {e}"
                                        ))
                                    })?;
                            }
                        }
                        return Ok(());
                    }
                }
            }

            Err(NexarError::DeviceError(
                "GPUDirect recv_raw_gpu: no suitable GPUDirect MR available; \
                 use recv_bytes() + stage_host_to_gpu() at the application layer"
                    .into(),
            ))
        }
    }

    async fn send_via_gpudirect(gd: Arc<GpuDirectState>, pooled: PooledGpuMr) -> Result<()> {
        tokio::task::spawn_blocking(move || {
            let qp = gd
                .qp
                .lock()
                .map_err(|e| NexarError::DeviceError(format!("GPUDirect lock poisoned: {e}")))?;
            qp.send(pooled.mr(), 0)
        })
        .await
        .map_err(|e| NexarError::DeviceError(format!("GPUDirect spawn_blocking: {e}")))?
    }

    async fn recv_via_gpudirect(gd: Arc<GpuDirectState>, pooled: PooledGpuMr) -> Result<()> {
        tokio::task::spawn_blocking(move || {
            let qp = gd
                .qp
                .lock()
                .map_err(|e| NexarError::DeviceError(format!("GPUDirect lock poisoned: {e}")))?;
            qp.recv(pooled.mr(), 0)
        })
        .await
        .map_err(|e| NexarError::DeviceError(format!("GPUDirect spawn_blocking: {e}")))?
    }
}

#[cfg(feature = "gpudirect")]
pub use gpudirect_ext::{GpuDirectState, GpuDirectStateHolder, PeerConnectionGpuDirectExt};
