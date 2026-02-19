use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::encode_message;
use crate::types::{Priority, Rank};

/// Stream type tag: first byte on every QUIC uni stream.
/// Allows the router to dispatch streams to the correct channel without ambiguity.
pub(crate) const STREAM_TAG_FRAMED: u8 = 0x01;
pub(crate) const STREAM_TAG_RAW: u8 = 0x02;
/// Raw stream with a communicator ID prefix (for split communicators).
pub(crate) const STREAM_TAG_RAW_COMM: u8 = 0x03;

/// A connection to a single peer node, wrapping a QUIC connection.
///
/// Handles the **send side** of communication. All receiving is done by
/// `PeerRouter`, which runs a single `accept_uni` loop per peer and
/// demultiplexes incoming streams into typed channels.
///
/// Every outbound stream begins with a 1-byte stream type tag
/// (`STREAM_TAG_FRAMED`, `STREAM_TAG_RAW`, or `STREAM_TAG_RAW_COMM`) so
/// the remote router can dispatch it correctly.
///
/// When the `rdma` feature is enabled and an RDMA connection has been
/// established, bulk raw sends are offloaded to RDMA for kernel-bypass
/// performance. Control messages always use QUIC.
pub struct PeerConnection {
    pub rank: Rank,
    pub(crate) conn: quinn::Connection,
    /// Optional RDMA connection for bulk data (feature = "rdma").
    #[cfg(feature = "rdma")]
    pub(crate) rdma: Option<RdmaState>,
    /// Optional GPUDirect RDMA for GPU-direct sends/recvs (feature = "gpudirect").
    #[cfg(feature = "gpudirect")]
    pub(crate) gpudirect: Option<GpuDirectState>,
}

/// RDMA send state for a single peer connection.
#[cfg(feature = "rdma")]
pub(crate) struct RdmaState {
    pub conn: std::sync::Mutex<crate::transport::rdma::RdmaConnection>,
    pub pool: std::sync::Arc<crate::transport::rdma::RdmaMemoryPool>,
}

/// GPUDirect RDMA state for a single peer connection.
#[cfg(feature = "gpudirect")]
pub(crate) struct GpuDirectState {
    pub qp: std::sync::Mutex<crate::transport::rdma::GpuDirectQp>,
    pub pool: std::sync::Arc<crate::transport::rdma::GpuDirectPool>,
}

impl PeerConnection {
    /// Create a `PeerConnection` from an established QUIC connection.
    pub fn new(rank: Rank, conn: quinn::Connection) -> Self {
        Self {
            rank,
            conn,
            #[cfg(feature = "rdma")]
            rdma: None,
            #[cfg(feature = "gpudirect")]
            gpudirect: None,
        }
    }

    /// Attach an RDMA connection for bulk data offload.
    #[cfg(feature = "rdma")]
    pub fn set_rdma(
        &mut self,
        rdma_conn: crate::transport::rdma::RdmaConnection,
        pool: std::sync::Arc<crate::transport::rdma::RdmaMemoryPool>,
    ) {
        self.rdma = Some(RdmaState {
            conn: std::sync::Mutex::new(rdma_conn),
            pool,
        });
    }

    /// Send a control message as a framed uni stream (always QUIC).
    pub async fn send_message(&self, msg: &NexarMessage, priority: Priority) -> Result<()> {
        let buf = encode_message(msg, priority)?;
        self.send_tagged(STREAM_TAG_FRAMED, &buf).await
    }

    /// Send raw bytes on a new unidirectional stream (for bulk tensor data).
    /// If RDMA is available, offloads to RDMA via `spawn_blocking`.
    pub async fn send_raw(&self, data: &[u8]) -> Result<()> {
        #[cfg(feature = "rdma")]
        if let Some(ref rdma) = self.rdma {
            return send_via_rdma(rdma, data).await;
        }
        self.send_tagged(STREAM_TAG_RAW, data).await
    }

    /// Send raw bytes tagged with a communicator ID (for split communicators).
    /// Always uses QUIC (split comms are a logical overlay).
    pub async fn send_raw_comm(&self, comm_id: u32, data: &[u8]) -> Result<()> {
        let mut stream = self
            .conn
            .open_uni()
            .await
            .map_err(|e| NexarError::Transport(format!("open uni stream: {e}")))?;
        stream
            .write_all(&[STREAM_TAG_RAW_COMM])
            .await
            .map_err(|e| NexarError::Transport(format!("write stream tag: {e}")))?;
        stream
            .write_all(&comm_id.to_le_bytes())
            .await
            .map_err(|e| NexarError::Transport(format!("write comm_id: {e}")))?;
        stream
            .write_all(&(data.len() as u64).to_le_bytes())
            .await
            .map_err(|e| NexarError::Transport(format!("write length: {e}")))?;
        stream
            .write_all(data)
            .await
            .map_err(|e| NexarError::Transport(format!("write payload: {e}")))?;
        stream
            .finish()
            .map_err(|e| NexarError::Transport(format!("finish stream: {e}")))?;
        Ok(())
    }

    /// Get the remote address of this connection.
    pub fn remote_addr(&self) -> std::net::SocketAddr {
        self.conn.remote_address()
    }

    /// Attach a GPUDirect RDMA connection for GPU-direct sends/recvs.
    #[cfg(feature = "gpudirect")]
    pub fn set_gpudirect(
        &mut self,
        qp: crate::transport::rdma::GpuDirectQp,
        pool: std::sync::Arc<crate::transport::rdma::GpuDirectPool>,
    ) {
        self.gpudirect = Some(GpuDirectState {
            qp: std::sync::Mutex::new(qp),
            pool,
        });
    }

    /// Send directly from GPU memory via GPUDirect RDMA.
    ///
    /// Tiered fallback: GPUDirect RDMA → RDMA (staged via CUDA D2H) → QUIC (staged).
    #[cfg(feature = "gpudirect")]
    pub async fn send_raw_gpu(&self, gpu_ptr: u64, size: usize) -> Result<()> {
        // Tier 1: GPUDirect RDMA — NIC reads directly from GPU memory.
        if let Some(ref gd) = self.gpudirect {
            if let Some(pooled) = gd.pool.checkout() {
                let mr_size = pooled.mr().size();
                let mr_gpu_ptr = pooled.mr().gpu_ptr();
                if mr_size >= size {
                    // D2D copy into the registered MR's GPU region, then RDMA send.
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

                    return send_via_gpudirect(gd, pooled).await;
                }
            }
        }

        // Tier 2: Stage GPU→host, then use regular RDMA or QUIC.
        let host_data = crate::transport::rdma::gpudirect::stage_gpu_to_host(gpu_ptr, size)?;
        self.send_raw(&host_data).await
    }

    /// Receive directly into GPU memory via GPUDirect RDMA.
    ///
    /// Tiered fallback: GPUDirect RDMA → RDMA (staged via CUDA H2D) → QUIC (staged).
    #[cfg(feature = "gpudirect")]
    pub async fn recv_raw_gpu(&self, gpu_ptr: u64, size: usize) -> Result<()> {
        // Tier 1: GPUDirect RDMA — NIC writes directly to GPU memory.
        if let Some(ref gd) = self.gpudirect {
            if let Some(pooled) = gd.pool.checkout() {
                let mr_size = pooled.mr().size();
                let mr_gpu_ptr = pooled.mr().gpu_ptr();
                if mr_size >= size {
                    recv_via_gpudirect(gd, pooled).await?;

                    // Copy from pooled MR's GPU region to target if different.
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

        // Tier 2: Receive via QUIC/RDMA into host memory, then stage to GPU.
        // This requires the caller to have already posted a recv on the
        // QUIC/RDMA path. Since recv_raw_gpu doesn't have access to the
        // raw recv channel, we stage through a host buffer using RDMA if
        // available, otherwise return an error (QUIC recv requires the
        // router path which is at a higher layer).
        #[cfg(feature = "rdma")]
        if let Some(ref _rdma) = self.rdma {
            // RDMA recv requires the remote to also use RDMA send. Since
            // send_raw_gpu tier 2 uses send_raw (which triggers RDMA send),
            // we can't reliably recv here without a paired protocol.
            // Fall through to error.
        }

        Err(NexarError::DeviceError(
            "GPUDirect recv_raw_gpu: no suitable GPUDirect MR available; \
             use recv_bytes() + stage_host_to_gpu() at the application layer"
                .into(),
        ))
    }

    /// Open a uni stream, write the stream type tag + length-prefixed payload,
    /// then finish. Shared by both framed and raw sends.
    async fn send_tagged(&self, tag: u8, data: &[u8]) -> Result<()> {
        let mut stream = self
            .conn
            .open_uni()
            .await
            .map_err(|e| NexarError::Transport(format!("open uni stream: {e}")))?;
        stream
            .write_all(&[tag])
            .await
            .map_err(|e| NexarError::Transport(format!("write stream tag: {e}")))?;
        stream
            .write_all(&(data.len() as u64).to_le_bytes())
            .await
            .map_err(|e| NexarError::Transport(format!("write length: {e}")))?;
        stream
            .write_all(data)
            .await
            .map_err(|e| NexarError::Transport(format!("write payload: {e}")))?;
        stream
            .finish()
            .map_err(|e| NexarError::Transport(format!("finish stream: {e}")))?;
        Ok(())
    }
}

/// Wrapper to force a closure to be Send for `spawn_blocking`.
///
/// # Safety
/// The caller must guarantee all captured values are safe to send across threads
/// and that any pointers outlive the blocking task.
#[cfg(any(feature = "rdma", feature = "gpudirect"))]
struct ForceSend<F>(F);
#[cfg(any(feature = "rdma", feature = "gpudirect"))]
unsafe impl<F> Send for ForceSend<F> {}

#[cfg(any(feature = "rdma", feature = "gpudirect"))]
impl<F: FnOnce() -> R, R> ForceSend<F> {
    fn call(self) -> R {
        (self.0)()
    }
}

/// Send data via GPUDirect RDMA from a pooled GPU MR.
#[cfg(feature = "gpudirect")]
async fn send_via_gpudirect(
    gd: &GpuDirectState,
    pooled: crate::transport::rdma::gpudirect::PooledGpuMr,
) -> Result<()> {
    let qp_ptr = std::ptr::addr_of!(gd.qp);

    // Safety: GpuDirectState lives in PeerConnection behind Arc, kept alive
    // by NexarClient. The blocking task completes before drop.
    let task = ForceSend(move || {
        let qp_mutex = unsafe { &*qp_ptr };
        let qp = qp_mutex
            .lock()
            .map_err(|e| NexarError::DeviceError(format!("GPUDirect lock poisoned: {e}")))?;
        qp.send(pooled.mr(), 0)
    });

    tokio::task::spawn_blocking(move || task.call())
        .await
        .map_err(|e| NexarError::DeviceError(format!("GPUDirect spawn_blocking: {e}")))?
}

/// Receive data via GPUDirect RDMA into a pooled GPU MR.
#[cfg(feature = "gpudirect")]
async fn recv_via_gpudirect(
    gd: &GpuDirectState,
    pooled: crate::transport::rdma::gpudirect::PooledGpuMr,
) -> Result<()> {
    let qp_ptr = std::ptr::addr_of!(gd.qp);

    let task = ForceSend(move || {
        let qp_mutex = unsafe { &*qp_ptr };
        let qp = qp_mutex
            .lock()
            .map_err(|e| NexarError::DeviceError(format!("GPUDirect lock poisoned: {e}")))?;
        qp.recv(pooled.mr(), 0)
    });

    tokio::task::spawn_blocking(move || task.call())
        .await
        .map_err(|e| NexarError::DeviceError(format!("GPUDirect spawn_blocking: {e}")))?
}

/// Send data via RDMA: copy into a registered MR, post send, wait for completion.
/// Runs in `spawn_blocking` since ibverbs operations are synchronous.
#[cfg(feature = "rdma")]
async fn send_via_rdma(rdma: &RdmaState, data: &[u8]) -> Result<()> {
    let mut pooled = rdma.pool.checkout()?;
    let len = data.len();
    pooled[..len].copy_from_slice(data);

    let conn_ptr = std::ptr::addr_of!(rdma.conn);

    // Safety: The RdmaState is behind an Arc in PeerConnection which is
    // kept alive by the NexarClient. The blocking task completes before
    // the PeerConnection is dropped. `pooled` is moved entirely into the
    // closure, avoiding aliased mutable references.
    let task = ForceSend(move || {
        let conn_mutex = unsafe { &*conn_ptr };
        let mut conn = conn_mutex
            .lock()
            .map_err(|e| NexarError::DeviceError(format!("RDMA lock poisoned: {e}")))?;
        conn.send(pooled.mr_mut(), 0)?;
        Ok::<(), NexarError>(())
    });

    tokio::task::spawn_blocking(move || task.call())
        .await
        .map_err(|e| NexarError::DeviceError(format!("RDMA spawn_blocking: {e}")))?
}
