//! RDMA queue pair management for reliable connected (RC) transport.
//!
//! Two-phase construction:
//! 1. `RdmaContext::new()` — opens device, allocates PD and CQs.
//! 2. `RdmaContext::prepare_connection()` — creates a QP in INIT state.
//! 3. Exchange `QueuePairEndpoint` via QUIC control channel.
//! 4. `PreparedRdmaConnection::complete()` — handshake to RTS.
//!
//! # Lifetime management
//!
//! ibverbs types have interlinked lifetimes (CQ borrows Context, PD borrows
//! Context, QP borrows both). We use raw pointers internally to break the
//! self-referential lifetime chain, keeping all resources alive through
//! `RdmaContext` ownership.

use crate::error::{NexarError, Result};
use crate::types::Rank;
use std::sync::Arc;

/// Shared per-device RDMA resources.
///
/// Owns the ibverbs `Context`, protection domain, and completion queues.
/// All RDMA connections created from this context share these resources.
///
/// Heap-allocated via `Box` to ensure stable addresses. The ibverbs objects
/// borrow from `Context`, so we transmute their lifetimes to `'static` —
/// safe because field drop order guarantees CQs and PD drop before Context.
///
/// CQs are protected by Mutexes to allow safe concurrent polling from
/// multiple QPs.
pub struct RdmaContext {
    // Fields declared in reverse-dependency order so that the PD and CQs
    // are dropped before the Context.
    send_cq: std::sync::Mutex<ibverbs::CompletionQueue<'static>>,
    recv_cq: std::sync::Mutex<ibverbs::CompletionQueue<'static>>,
    pd: ibverbs::ProtectionDomain<'static>,
    // Context must be last — everything borrows from it.
    _ctx: Box<ibverbs::Context>,
}

// Safety: ibverbs types are Send+Sync per the crate docs.
unsafe impl Send for RdmaContext {}
unsafe impl Sync for RdmaContext {}

impl RdmaContext {
    /// Open an RDMA device and allocate shared resources.
    ///
    /// `device_index` selects which RDMA device to use (default: first).
    /// Creates a protection domain and two CQs (256 entries each).
    pub fn new(device_index: Option<usize>) -> Result<Self> {
        let dev_list = ibverbs::devices().map_err(|e| NexarError::DeviceError(e.to_string()))?;
        if dev_list.is_empty() {
            return Err(NexarError::DeviceError("no RDMA devices found".into()));
        }
        let idx = device_index.unwrap_or(0);
        let dev = dev_list.get(idx).ok_or_else(|| {
            NexarError::DeviceError(format!(
                "RDMA device index {idx} out of range (have {})",
                dev_list.len()
            ))
        })?;
        let ctx = Box::new(
            dev.open()
                .map_err(|e| NexarError::DeviceError(format!("open RDMA device: {e}")))?,
        );

        // Safety: We extend lifetimes to 'static because we own the Context
        // in a Box (stable address) and guarantee all borrowed objects are
        // dropped before it. The struct field order ensures correct drop order.
        let ctx_ptr: *const ibverbs::Context = &*ctx;
        let ctx_ref: &'static ibverbs::Context = unsafe { &*ctx_ptr };

        let send_cq = ctx_ref
            .create_cq(256, 0)
            .map_err(|e| NexarError::DeviceError(format!("create send CQ: {e}")))?;
        let recv_cq = ctx_ref
            .create_cq(256, 1)
            .map_err(|e| NexarError::DeviceError(format!("create recv CQ: {e}")))?;
        let pd = ctx_ref
            .alloc_pd()
            .map_err(|e| NexarError::DeviceError(format!("alloc PD: {e}")))?;

        Ok(Self {
            send_cq: std::sync::Mutex::new(send_cq),
            recv_cq: std::sync::Mutex::new(recv_cq),
            pd,
            _ctx: ctx,
        })
    }

    /// Allocate a registered memory region of `size` bytes.
    pub fn allocate(&self, size: usize) -> Result<ibverbs::MemoryRegion<u8>> {
        self.pd
            .allocate::<u8>(size)
            .map_err(|e| NexarError::DeviceError(format!("allocate MR: {e}")))
    }

    /// Prepare a new RDMA connection (QP in INIT state) for the given peer.
    ///
    /// Each QP gets references to the shared CQs (protected by Mutex in
    /// `RdmaContext`). The `RdmaConnection` stores `Arc<RdmaContext>` to
    /// ensure CQ lifetime and provide synchronized CQ access.
    pub fn prepare_connection(self: &Arc<Self>, peer_rank: Rank) -> Result<PreparedRdmaConnection> {
        let send_cq = self
            .send_cq
            .lock()
            .map_err(|e| NexarError::DeviceError(format!("send CQ lock poisoned: {e}")))?;
        let recv_cq = self
            .recv_cq
            .lock()
            .map_err(|e| NexarError::DeviceError(format!("recv CQ lock poisoned: {e}")))?;

        let mut builder = self
            .pd
            .create_qp(&send_cq, &recv_cq, ibverbs::ibv_qp_type::IBV_QPT_RC);
        builder
            .allow_remote_rw()
            .set_max_send_wr(128)
            .set_max_recv_wr(128);
        let pqp = builder
            .build()
            .map_err(|e| NexarError::DeviceError(format!("build QP: {e}")))?;

        // Safety: The CQ guards are released here, but the underlying CQs
        // live inside the Arc<RdmaContext> which the PreparedRdmaConnection
        // holds a reference to. The transmute to 'static is safe because
        // we guarantee the RdmaContext (and its CQs) outlive the QP.
        let pqp = unsafe {
            std::mem::transmute::<ibverbs::PreparedQueuePair<'_>, ibverbs::PreparedQueuePair<'static>>(
                pqp,
            )
        };

        // CQ guards are dropped here (after transmute erases the borrow).
        drop(send_cq);
        drop(recv_cq);

        Ok(PreparedRdmaConnection {
            pqp,
            peer_rank,
            ctx: Arc::clone(self),
        })
    }
}

/// An RDMA connection that has been prepared but not yet handshaken.
pub struct PreparedRdmaConnection {
    pqp: ibverbs::PreparedQueuePair<'static>,
    peer_rank: Rank,
    ctx: Arc<RdmaContext>,
}

// Safety: ibverbs types are thread-safe per docs.
unsafe impl Send for PreparedRdmaConnection {}
unsafe impl Sync for PreparedRdmaConnection {}

impl PreparedRdmaConnection {
    /// Get the local endpoint to exchange with the remote peer.
    pub fn endpoint(&self) -> ibverbs::QueuePairEndpoint {
        self.pqp.endpoint()
    }

    /// Complete the handshake with the remote peer's endpoint.
    pub fn complete(self, remote: ibverbs::QueuePairEndpoint) -> Result<RdmaConnection> {
        let qp = self
            .pqp
            .handshake(remote)
            .map_err(|e| NexarError::ConnectionFailed {
                rank: self.peer_rank,
                reason: format!("RDMA handshake: {e}"),
            })?;
        Ok(RdmaConnection {
            _peer_rank: self.peer_rank,
            qp: unsafe {
                std::mem::transmute::<ibverbs::QueuePair<'_>, ibverbs::QueuePair<'static>>(qp)
            },
            ctx: self.ctx,
        })
    }
}

/// A fully connected RDMA RC queue pair to a single peer.
pub struct RdmaConnection {
    _peer_rank: Rank,
    qp: ibverbs::QueuePair<'static>,
    ctx: Arc<RdmaContext>,
}

unsafe impl Send for RdmaConnection {}
unsafe impl Sync for RdmaConnection {}

/// Default timeout for CQ polling (5 seconds).
const CQ_POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl RdmaConnection {
    /// Post an RDMA send and wait for completion.
    pub fn send(&mut self, mr: &mut ibverbs::MemoryRegion<u8>, wr_id: u64) -> Result<()> {
        let len = mr.len();
        unsafe {
            self.qp
                .post_send(mr, 0..len, wr_id)
                .map_err(|e| NexarError::DeviceError(format!("post_send: {e}")))?;
        }
        self.wait_send_completion()
    }

    /// Post an RDMA receive and wait for completion.
    pub fn recv(&mut self, mr: &mut ibverbs::MemoryRegion<u8>, wr_id: u64) -> Result<()> {
        let len = mr.len();
        unsafe {
            self.qp
                .post_receive(mr, 0..len, wr_id)
                .map_err(|e| NexarError::DeviceError(format!("post_receive: {e}")))?;
        }
        self.wait_recv_completion()
    }

    /// Poll the send CQ without blocking. Returns completed work request IDs.
    pub fn poll_send_cq(&self) -> Result<Vec<u64>> {
        let cq = self
            .ctx
            .send_cq
            .lock()
            .map_err(|e| NexarError::DeviceError(format!("send CQ lock poisoned: {e}")))?;
        let mut completions = [ibverbs::ibv_wc::default(); 16];
        let done = cq
            .poll(&mut completions)
            .map_err(|e| NexarError::DeviceError(format!("poll send CQ: {e}")))?;
        check_completions(done)?;
        Ok(done.iter().map(|wc| wc.wr_id()).collect())
    }

    /// Poll the recv CQ without blocking. Returns completed work request IDs.
    pub fn poll_recv_cq(&self) -> Result<Vec<u64>> {
        let cq = self
            .ctx
            .recv_cq
            .lock()
            .map_err(|e| NexarError::DeviceError(format!("recv CQ lock poisoned: {e}")))?;
        let mut completions = [ibverbs::ibv_wc::default(); 16];
        let done = cq
            .poll(&mut completions)
            .map_err(|e| NexarError::DeviceError(format!("poll recv CQ: {e}")))?;
        check_completions(done)?;
        Ok(done.iter().map(|wc| wc.wr_id()).collect())
    }

    fn wait_send_completion(&self) -> Result<()> {
        poll_until_complete(&self.ctx.send_cq, CQ_POLL_TIMEOUT)
    }

    fn wait_recv_completion(&self) -> Result<()> {
        poll_until_complete(&self.ctx.recv_cq, CQ_POLL_TIMEOUT)
    }
}

/// Poll a Mutex-protected CQ until at least one completion arrives, with timeout.
fn poll_until_complete(
    cq_mutex: &std::sync::Mutex<ibverbs::CompletionQueue<'static>>,
    timeout: std::time::Duration,
) -> Result<()> {
    let start = std::time::Instant::now();
    let mut completions = [ibverbs::ibv_wc::default(); 1];
    loop {
        {
            let cq = cq_mutex
                .lock()
                .map_err(|e| NexarError::DeviceError(format!("CQ lock poisoned: {e}")))?;
            let done = cq
                .poll(&mut completions)
                .map_err(|e| NexarError::DeviceError(format!("poll CQ: {e}")))?;
            if !done.is_empty() {
                check_completions(done)?;
                return Ok(());
            }
        }
        if start.elapsed() > timeout {
            return Err(NexarError::DeviceError(format!(
                "CQ poll timed out after {}ms",
                timeout.as_millis()
            )));
        }
        std::hint::spin_loop();
    }
}

/// Check that all work completions succeeded.
fn check_completions(completions: &[ibverbs::ibv_wc]) -> Result<()> {
    for wc in completions {
        if let Some((status, vendor_err)) = wc.error() {
            return Err(NexarError::DeviceError(format!(
                "work completion failed: status={status:?}, vendor_err={vendor_err}, wr_id={}",
                wc.wr_id()
            )));
        }
    }
    Ok(())
}
