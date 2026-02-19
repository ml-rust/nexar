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

/// Shared per-device RDMA resources.
///
/// Owns the ibverbs `Context`, protection domain, and completion queues.
/// All RDMA connections created from this context share these resources.
///
/// # Safety
///
/// The raw pointers are valid for the lifetime of `RdmaContext` because we
/// own `_ctx` which keeps the device context alive, and `_pd`/`_send_cq`/
/// `_recv_cq` are stored alongside it, preventing drop reordering.
pub struct RdmaContext {
    // These fields exist solely to prevent the ibverbs objects from being
    // dropped. They must be declared in reverse-dependency order so that
    // the PD and CQs are dropped before the Context.
    _send_cq: ibverbs::CompletionQueue<'static>,
    _recv_cq: ibverbs::CompletionQueue<'static>,
    _pd: ibverbs::ProtectionDomain<'static>,
    // Context must be last — everything borrows from it.
    _ctx: Box<ibverbs::Context>,

    // Raw pointers for actual use (derived from the owned objects above).
    pd_ptr: *mut ibverbs::ProtectionDomain<'static>,
    send_cq_ptr: *mut ibverbs::CompletionQueue<'static>,
    recv_cq_ptr: *mut ibverbs::CompletionQueue<'static>,
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

        let mut result = Self {
            pd_ptr: std::ptr::null_mut(),
            send_cq_ptr: std::ptr::null_mut(),
            recv_cq_ptr: std::ptr::null_mut(),
            _send_cq: send_cq,
            _recv_cq: recv_cq,
            _pd: pd,
            _ctx: ctx,
        };
        // Set the raw pointers to point at the owned fields.
        result.pd_ptr = &mut result._pd as *mut _;
        result.send_cq_ptr = &mut result._send_cq as *mut _;
        result.recv_cq_ptr = &mut result._recv_cq as *mut _;
        Ok(result)
    }

    fn pd(&self) -> &ibverbs::ProtectionDomain<'static> {
        unsafe { &*self.pd_ptr }
    }

    fn send_cq(&self) -> &ibverbs::CompletionQueue<'static> {
        unsafe { &*self.send_cq_ptr }
    }

    fn recv_cq(&self) -> &ibverbs::CompletionQueue<'static> {
        unsafe { &*self.recv_cq_ptr }
    }

    /// Allocate a registered memory region of `size` bytes.
    pub fn allocate(&self, size: usize) -> Result<ibverbs::MemoryRegion<u8>> {
        self.pd()
            .allocate::<u8>(size)
            .map_err(|e| NexarError::DeviceError(format!("allocate MR: {e}")))
    }

    /// Prepare a new RDMA connection (QP in INIT state) for the given peer.
    pub fn prepare_connection(&self, peer_rank: Rank) -> Result<PreparedRdmaConnection> {
        let mut builder = self.pd().create_qp(
            self.send_cq(),
            self.recv_cq(),
            ibverbs::ibv_qp_type::IBV_QPT_RC,
        );
        builder
            .allow_remote_rw()
            .set_max_send_wr(128)
            .set_max_recv_wr(128);
        let pqp = builder
            .build()
            .map_err(|e| NexarError::DeviceError(format!("build QP: {e}")))?;
        Ok(PreparedRdmaConnection {
            pqp: unsafe {
                std::mem::transmute::<
                    ibverbs::PreparedQueuePair<'_>,
                    ibverbs::PreparedQueuePair<'static>,
                >(pqp)
            },
            peer_rank,
            send_cq_ptr: self.send_cq_ptr,
            recv_cq_ptr: self.recv_cq_ptr,
        })
    }
}

/// An RDMA connection that has been prepared but not yet handshaken.
pub struct PreparedRdmaConnection {
    pqp: ibverbs::PreparedQueuePair<'static>,
    peer_rank: Rank,
    send_cq_ptr: *mut ibverbs::CompletionQueue<'static>,
    recv_cq_ptr: *mut ibverbs::CompletionQueue<'static>,
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
            send_cq_ptr: self.send_cq_ptr,
            recv_cq_ptr: self.recv_cq_ptr,
        })
    }
}

/// A fully connected RDMA RC queue pair to a single peer.
pub struct RdmaConnection {
    _peer_rank: Rank,
    qp: ibverbs::QueuePair<'static>,
    send_cq_ptr: *mut ibverbs::CompletionQueue<'static>,
    recv_cq_ptr: *mut ibverbs::CompletionQueue<'static>,
}

unsafe impl Send for RdmaConnection {}
unsafe impl Sync for RdmaConnection {}

impl RdmaConnection {
    /// Post an RDMA send and wait for completion.
    pub fn send(&mut self, mr: &mut ibverbs::MemoryRegion<u8>, wr_id: u64) -> Result<()> {
        let len = mr.len();
        unsafe {
            self.qp
                .post_send(mr, 0..len, wr_id)
                .map_err(|e| NexarError::DeviceError(format!("post_send: {e}")))?;
        }
        self.wait_send_completion()?;
        Ok(())
    }

    /// Post an RDMA receive and wait for completion.
    pub fn recv(&mut self, mr: &mut ibverbs::MemoryRegion<u8>, wr_id: u64) -> Result<()> {
        let len = mr.len();
        unsafe {
            self.qp
                .post_receive(mr, 0..len, wr_id)
                .map_err(|e| NexarError::DeviceError(format!("post_receive: {e}")))?;
        }
        self.wait_recv_completion()?;
        Ok(())
    }

    /// Poll the send CQ without blocking. Returns completed work request IDs.
    pub fn poll_send_cq(&self) -> Result<Vec<u64>> {
        let cq = unsafe { &*self.send_cq_ptr };
        let mut completions = [ibverbs::ibv_wc::default(); 16];
        let done = cq
            .poll(&mut completions)
            .map_err(|e| NexarError::DeviceError(format!("poll send CQ: {e}")))?;
        check_completions(done)?;
        Ok(done.iter().map(|wc| wc.wr_id()).collect())
    }

    /// Poll the recv CQ without blocking. Returns completed work request IDs.
    pub fn poll_recv_cq(&self) -> Result<Vec<u64>> {
        let cq = unsafe { &*self.recv_cq_ptr };
        let mut completions = [ibverbs::ibv_wc::default(); 16];
        let done = cq
            .poll(&mut completions)
            .map_err(|e| NexarError::DeviceError(format!("poll recv CQ: {e}")))?;
        check_completions(done)?;
        Ok(done.iter().map(|wc| wc.wr_id()).collect())
    }

    fn wait_send_completion(&self) -> Result<()> {
        poll_until_complete(unsafe { &*self.send_cq_ptr })
    }

    fn wait_recv_completion(&self) -> Result<()> {
        poll_until_complete(unsafe { &*self.recv_cq_ptr })
    }
}

/// Spin-poll a CQ until at least one completion arrives.
fn poll_until_complete(cq: &ibverbs::CompletionQueue<'_>) -> Result<()> {
    let mut completions = [ibverbs::ibv_wc::default(); 1];
    loop {
        let done = cq
            .poll(&mut completions)
            .map_err(|e| NexarError::DeviceError(format!("poll CQ: {e}")))?;
        if !done.is_empty() {
            check_completions(done)?;
            return Ok(());
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
