//! GPUDirect RDMA: register GPU memory directly with the NIC via raw `ibv_reg_mr`.
//!
//! The `ibverbs` crate's `ProtectionDomain::allocate()` only registers host memory.
//! For GPUDirect we need `ibv_reg_mr` on a CUDA device pointer so the NIC can
//! DMA directly to/from GPU memory. This module uses `ibverbs-sys` raw FFI to
//! build a self-contained RDMA context (`GpuDirectContext`) with its own device
//! handle, PD, and CQs — completely independent of the `ibverbs` crate's managed
//! types.
//!
//! # Requirements
//!
//! - NVIDIA GPU with GPUDirect RDMA support (Kepler or newer)
//! - Mellanox/NVIDIA InfiniBand HCA with PeerDirect support
//! - `nvidia-peermem` kernel module loaded
//! - CUDA runtime available

use crate::error::{NexarError, Result};
use crossbeam_queue::ArrayQueue;
use ibverbs_sys::{
    ibv_access_flags, ibv_qp_attr_mask, ibv_qp_state, ibv_qp_type, ibv_send_flags, ibv_wr_opcode,
};
use std::os::raw::c_int;
use std::ptr;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// GpuDirectContext — raw FFI RDMA device, PD, and CQs
// ---------------------------------------------------------------------------

/// Self-contained RDMA context for GPUDirect operations via raw `ibverbs-sys` FFI.
///
/// Opens its own device handle, PD, and CQ pair. This is separate from the
/// `ibverbs` crate's `RdmaContext` because we need raw `ibv_reg_mr` access
/// for GPU device pointers, which the safe `ibverbs` crate doesn't expose.
pub struct GpuDirectContext {
    ctx: *mut ibverbs_sys::ibv_context,
    pd: *mut ibverbs_sys::ibv_pd,
}

unsafe impl Send for GpuDirectContext {}
unsafe impl Sync for GpuDirectContext {}

impl GpuDirectContext {
    /// Open an RDMA device and allocate PD + CQs for GPUDirect operations.
    ///
    /// `device_index`: which RDMA device to use (default: first available).
    pub fn new(device_index: Option<usize>) -> Result<Self> {
        unsafe {
            let mut num_devices: c_int = 0;
            let dev_list = ibverbs_sys::ibv_get_device_list(&mut num_devices);
            if dev_list.is_null() || num_devices == 0 {
                return Err(NexarError::DeviceError(
                    "GPUDirect: no RDMA devices found".into(),
                ));
            }

            let idx = device_index.unwrap_or(0);
            if idx >= num_devices as usize {
                ibverbs_sys::ibv_free_device_list(dev_list);
                return Err(NexarError::DeviceError(format!(
                    "GPUDirect: device index {idx} out of range (have {num_devices})"
                )));
            }

            let dev = *dev_list.add(idx);
            let ctx = ibverbs_sys::ibv_open_device(dev);
            ibverbs_sys::ibv_free_device_list(dev_list);

            if ctx.is_null() {
                return Err(NexarError::DeviceError(
                    "GPUDirect: ibv_open_device failed".into(),
                ));
            }

            let pd = ibverbs_sys::ibv_alloc_pd(ctx);
            if pd.is_null() {
                ibverbs_sys::ibv_close_device(ctx);
                return Err(NexarError::DeviceError(
                    "GPUDirect: ibv_alloc_pd failed".into(),
                ));
            }

            Ok(Self { ctx, pd })
        }
    }

    /// Register a CUDA device pointer as an RDMA memory region.
    ///
    /// The NIC will DMA directly to/from this GPU memory. Requires `nvidia-peermem`.
    ///
    /// # Safety
    ///
    /// - `gpu_ptr` must be a valid CUDA device pointer.
    /// - The GPU memory must remain allocated for the lifetime of the returned `GpuMr`.
    pub unsafe fn register_gpu_memory(&self, gpu_ptr: u64, size: usize) -> Result<GpuMr> {
        let access = ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
            | ibv_access_flags::IBV_ACCESS_REMOTE_READ;

        let mr =
            unsafe { ibverbs_sys::ibv_reg_mr(self.pd, gpu_ptr as *mut _, size, access.0 as c_int) };
        if mr.is_null() {
            return Err(NexarError::DeviceError(format!(
                "GPUDirect: ibv_reg_mr failed for gpu_ptr=0x{gpu_ptr:x} size={size}. \
                 Is nvidia-peermem loaded? (modprobe nvidia-peermem)"
            )));
        }

        Ok(GpuMr { mr, gpu_ptr, size })
    }

    /// Prepare a QP for GPUDirect sends/recvs.
    ///
    /// Each QP gets its own dedicated CQ pair, avoiding the need for CQ-level
    /// synchronization when multiple QPs are polled concurrently.
    pub fn prepare_qp(&self) -> Result<PreparedGpuDirectQp> {
        unsafe {
            // Create per-QP CQs.
            let send_cq =
                ibverbs_sys::ibv_create_cq(self.ctx, 256, ptr::null_mut(), ptr::null_mut(), 0);
            if send_cq.is_null() {
                return Err(NexarError::DeviceError(
                    "GPUDirect: ibv_create_cq (send) failed".into(),
                ));
            }
            let recv_cq =
                ibverbs_sys::ibv_create_cq(self.ctx, 256, ptr::null_mut(), ptr::null_mut(), 0);
            if recv_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(send_cq);
                return Err(NexarError::DeviceError(
                    "GPUDirect: ibv_create_cq (recv) failed".into(),
                ));
            }

            let mut qp_init_attr: ibverbs_sys::ibv_qp_init_attr = std::mem::zeroed();
            qp_init_attr.qp_type = ibv_qp_type::IBV_QPT_RC;
            qp_init_attr.send_cq = send_cq;
            qp_init_attr.recv_cq = recv_cq;
            qp_init_attr.cap.max_send_wr = 128;
            qp_init_attr.cap.max_recv_wr = 128;
            qp_init_attr.cap.max_send_sge = 1;
            qp_init_attr.cap.max_recv_sge = 1;

            let qp = ibverbs_sys::ibv_create_qp(self.pd, &mut qp_init_attr);
            if qp.is_null() {
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                return Err(NexarError::DeviceError(
                    "GPUDirect: ibv_create_qp failed".into(),
                ));
            }

            // Transition to INIT.
            let mut attr: ibverbs_sys::ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
            attr.pkey_index = 0;
            attr.port_num = 1;
            attr.qp_access_flags = (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | ibv_access_flags::IBV_ACCESS_REMOTE_READ)
                .0;

            let mask = ibv_qp_attr_mask::IBV_QP_STATE
                | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
                | ibv_qp_attr_mask::IBV_QP_PORT
                | ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;

            let rc = ibverbs_sys::ibv_modify_qp(qp, &mut attr, mask.0 as c_int);
            if rc != 0 {
                ibverbs_sys::ibv_destroy_qp(qp);
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                return Err(NexarError::DeviceError(format!(
                    "GPUDirect: ibv_modify_qp to INIT failed (rc={rc})"
                )));
            }

            // Query GID for addressing (works for both IB and RoCE).
            let mut gid: ibverbs_sys::ibv_gid = std::mem::zeroed();
            let rc = ibverbs_sys::ibv_query_gid(self.ctx, 1, 0, &mut gid);
            if rc != 0 {
                ibverbs_sys::ibv_destroy_qp(qp);
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                return Err(NexarError::DeviceError(format!(
                    "GPUDirect: ibv_query_gid failed (rc={rc})"
                )));
            }

            let local_ep = GpuDirectEndpoint {
                qp_num: (*qp).qp_num,
                gid: gid.raw,
            };

            Ok(PreparedGpuDirectQp {
                qp,
                send_cq,
                recv_cq,
                local_ep,
            })
        }
    }
}

impl Drop for GpuDirectContext {
    fn drop(&mut self) {
        unsafe {
            if !self.pd.is_null() {
                ibverbs_sys::ibv_dealloc_pd(self.pd);
            }
            if !self.ctx.is_null() {
                ibverbs_sys::ibv_close_device(self.ctx);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpuMr — GPU memory registered as an RDMA MR
// ---------------------------------------------------------------------------

/// A GPU device pointer registered as an RDMA memory region.
///
/// The NIC can DMA directly to/from this memory without host staging.
pub struct GpuMr {
    mr: *mut ibverbs_sys::ibv_mr,
    gpu_ptr: u64,
    size: usize,
}

unsafe impl Send for GpuMr {}
unsafe impl Sync for GpuMr {}

impl GpuMr {
    /// The local key for send work requests.
    pub fn lkey(&self) -> u32 {
        unsafe { (*self.mr).lkey }
    }

    /// The remote key for RDMA read/write.
    pub fn rkey(&self) -> u32 {
        unsafe { (*self.mr).rkey }
    }

    /// The registered GPU device pointer.
    pub fn gpu_ptr(&self) -> u64 {
        self.gpu_ptr
    }

    /// The registered size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for GpuMr {
    fn drop(&mut self) {
        if !self.mr.is_null() {
            unsafe {
                ibverbs_sys::ibv_dereg_mr(self.mr);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpuDirectPool — pooled pre-registered GPU MRs
// ---------------------------------------------------------------------------

/// Pool of pre-registered GPU memory regions for reuse.
pub struct GpuDirectPool {
    queue: ArrayQueue<GpuMr>,
    #[allow(dead_code)]
    ctx: Arc<GpuDirectContext>,
    buf_size: usize,
}

unsafe impl Send for GpuDirectPool {}
unsafe impl Sync for GpuDirectPool {}

impl GpuDirectPool {
    /// Create a pool by registering `count` GPU buffers starting at `gpu_base_ptr`.
    ///
    /// Each buffer is `buf_size` bytes. The GPU memory at
    /// `[gpu_base_ptr, gpu_base_ptr + buf_size * count)` must be pre-allocated.
    ///
    /// # Safety
    ///
    /// The GPU memory must remain allocated for the lifetime of this pool.
    pub unsafe fn new(
        ctx: Arc<GpuDirectContext>,
        gpu_base_ptr: u64,
        buf_size: usize,
        count: usize,
    ) -> Result<Arc<Self>> {
        let queue = ArrayQueue::new(count);
        for i in 0..count {
            let ptr = gpu_base_ptr + (i * buf_size) as u64;
            let mr = unsafe { ctx.register_gpu_memory(ptr, buf_size)? };
            let _ = queue.push(mr);
        }
        Ok(Arc::new(Self {
            queue,
            ctx,
            buf_size,
        }))
    }

    /// Create an empty pool (no pre-registered buffers).
    ///
    /// Useful when the QP is established but GPU memory hasn't been allocated yet.
    /// `checkout()` will return `None` until buffers are registered.
    pub fn empty(ctx: Arc<GpuDirectContext>) -> Self {
        Self {
            queue: ArrayQueue::new(1),
            ctx,
            buf_size: 0,
        }
    }

    /// Checkout a registered GPU MR from the pool.
    ///
    /// Returns `None` if the pool is empty (caller should allocate or wait).
    pub fn checkout(self: &Arc<Self>) -> Option<PooledGpuMr> {
        self.queue.pop().map(|mr| PooledGpuMr {
            mr: Some(mr),
            pool: Arc::clone(self),
        })
    }

    /// Size of each buffer in the pool.
    pub fn buf_size(&self) -> usize {
        self.buf_size
    }
}

/// A pooled GPU MR that auto-returns to the pool on drop.
pub struct PooledGpuMr {
    mr: Option<GpuMr>,
    pool: Arc<GpuDirectPool>,
}

// Safety: GpuMr is Send+Sync (raw ibv_mr pointer is thread-safe per ibverbs docs),
// and Arc<GpuDirectPool> is Send+Sync.
unsafe impl Send for PooledGpuMr {}
unsafe impl Sync for PooledGpuMr {}

impl PooledGpuMr {
    /// Access the underlying `GpuMr`.
    pub fn mr(&self) -> &GpuMr {
        self.mr.as_ref().expect("GpuMr taken after drop")
    }
}

impl Drop for PooledGpuMr {
    fn drop(&mut self) {
        if let Some(mr) = self.mr.take() {
            let _ = self.pool.queue.push(mr);
        }
    }
}

// ---------------------------------------------------------------------------
// GpuDirectEndpoint — exchanged between peers for QP handshake
// ---------------------------------------------------------------------------

/// Endpoint data exchanged between peers to complete a GPUDirect QP handshake.
#[derive(Debug, Clone, Copy)]
pub struct GpuDirectEndpoint {
    pub qp_num: u32,
    pub gid: [u8; 16],
}

impl GpuDirectEndpoint {
    /// Serialize to bytes for exchange over QUIC.
    pub fn to_bytes(&self) -> [u8; 20] {
        let mut buf = [0u8; 20];
        buf[0..4].copy_from_slice(&self.qp_num.to_le_bytes());
        buf[4..20].copy_from_slice(&self.gid);
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(buf: &[u8; 20]) -> Self {
        Self {
            qp_num: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            gid: buf[4..20].try_into().unwrap(),
        }
    }
}

// ---------------------------------------------------------------------------
// PreparedGpuDirectQp — QP in INIT, ready for handshake
// ---------------------------------------------------------------------------

/// A GPUDirect QP in INIT state, ready for endpoint exchange and handshake.
pub struct PreparedGpuDirectQp {
    qp: *mut ibverbs_sys::ibv_qp,
    send_cq: *mut ibverbs_sys::ibv_cq,
    recv_cq: *mut ibverbs_sys::ibv_cq,
    local_ep: GpuDirectEndpoint,
}

unsafe impl Send for PreparedGpuDirectQp {}
unsafe impl Sync for PreparedGpuDirectQp {}

impl Drop for PreparedGpuDirectQp {
    fn drop(&mut self) {
        unsafe {
            if !self.qp.is_null() {
                ibverbs_sys::ibv_destroy_qp(self.qp);
                self.qp = ptr::null_mut();
            }
            // CQs are owned per-QP; destroy after QP.
            if !self.send_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(self.send_cq);
                self.send_cq = ptr::null_mut();
            }
            if !self.recv_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(self.recv_cq);
                self.recv_cq = ptr::null_mut();
            }
        }
    }
}

impl PreparedGpuDirectQp {
    /// Local endpoint to send to the remote peer.
    pub fn endpoint(&self) -> GpuDirectEndpoint {
        self.local_ep
    }

    /// Complete the handshake: INIT → RTR → RTS.
    ///
    /// Consumes self. On success, the QP is transferred to `GpuDirectQp`.
    /// On failure, the QP is destroyed by the error path (not by Drop,
    /// since we null out the pointer before returning the error).
    pub fn complete(mut self, remote: GpuDirectEndpoint) -> Result<GpuDirectQp> {
        unsafe {
            // INIT → RTR
            let mut attr: ibverbs_sys::ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
            attr.path_mtu = ibverbs_sys::IBV_MTU_4096;
            attr.dest_qp_num = remote.qp_num;
            attr.rq_psn = 0;
            attr.max_dest_rd_atomic = 4;
            attr.min_rnr_timer = 12;

            // Always use GRH (GID-based addressing) — works for both IB and RoCE.
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.dgid.raw = remote.gid;
            attr.ah_attr.grh.sgid_index = 0;
            attr.ah_attr.grh.hop_limit = 64;
            attr.ah_attr.grh.traffic_class = 0;
            attr.ah_attr.dlid = 0;
            attr.ah_attr.sl = 0;
            attr.ah_attr.src_path_bits = 0;
            attr.ah_attr.port_num = 1;

            let mask = ibv_qp_attr_mask::IBV_QP_STATE
                | ibv_qp_attr_mask::IBV_QP_AV
                | ibv_qp_attr_mask::IBV_QP_PATH_MTU
                | ibv_qp_attr_mask::IBV_QP_DEST_QPN
                | ibv_qp_attr_mask::IBV_QP_RQ_PSN
                | ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
                | ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;

            let rc = ibverbs_sys::ibv_modify_qp(self.qp, &mut attr, mask.0 as c_int);
            if rc != 0 {
                // Null out so Drop doesn't double-free after explicit destroy.
                let qp = self.qp;
                self.qp = ptr::null_mut();
                ibverbs_sys::ibv_destroy_qp(qp);
                return Err(NexarError::DeviceError(format!(
                    "GPUDirect: ibv_modify_qp to RTR failed (rc={rc})"
                )));
            }

            // RTR → RTS
            let mut attr: ibverbs_sys::ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
            attr.sq_psn = 0;
            attr.timeout = 14;
            attr.retry_cnt = 7;
            attr.rnr_retry = 7;
            attr.max_rd_atomic = 4;

            let mask = ibv_qp_attr_mask::IBV_QP_STATE
                | ibv_qp_attr_mask::IBV_QP_TIMEOUT
                | ibv_qp_attr_mask::IBV_QP_RETRY_CNT
                | ibv_qp_attr_mask::IBV_QP_RNR_RETRY
                | ibv_qp_attr_mask::IBV_QP_SQ_PSN
                | ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;

            let rc = ibverbs_sys::ibv_modify_qp(self.qp, &mut attr, mask.0 as c_int);
            if rc != 0 {
                let qp = self.qp;
                self.qp = ptr::null_mut();
                ibverbs_sys::ibv_destroy_qp(qp);
                return Err(NexarError::DeviceError(format!(
                    "GPUDirect: ibv_modify_qp to RTS failed (rc={rc})"
                )));
            }

            // Transfer ownership to GpuDirectQp; prevent Drop from destroying them.
            let qp = self.qp;
            let send_cq = self.send_cq;
            let recv_cq = self.recv_cq;
            self.qp = ptr::null_mut();
            self.send_cq = ptr::null_mut();
            self.recv_cq = ptr::null_mut();

            Ok(GpuDirectQp {
                qp,
                send_cq,
                recv_cq,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// GpuDirectQp — fully connected QP for GPU-direct sends/recvs
// ---------------------------------------------------------------------------

/// A fully connected GPUDirect QP for sending/receiving directly from GPU memory.
pub struct GpuDirectQp {
    qp: *mut ibverbs_sys::ibv_qp,
    send_cq: *mut ibverbs_sys::ibv_cq,
    recv_cq: *mut ibverbs_sys::ibv_cq,
}

unsafe impl Send for GpuDirectQp {}
unsafe impl Sync for GpuDirectQp {}

impl GpuDirectQp {
    /// Post a send from a GPU MR and wait for completion.
    pub fn send(&self, mr: &GpuMr, wr_id: u64) -> Result<()> {
        unsafe {
            let mut sge: ibverbs_sys::ibv_sge = std::mem::zeroed();
            sge.addr = mr.gpu_ptr();
            sge.length = mr.size() as u32;
            sge.lkey = mr.lkey();

            let mut wr: ibverbs_sys::ibv_send_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.opcode = ibv_wr_opcode::IBV_WR_SEND;
            wr.send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;

            let mut bad_wr: *mut ibverbs_sys::ibv_send_wr = ptr::null_mut();
            let ctx = (*self.qp).context;
            let ops = &mut (*ctx).ops;
            let rc =
                ops.post_send.as_mut().unwrap()(self.qp, &mut wr as *mut _, &mut bad_wr as *mut _);
            if rc != 0 {
                return Err(NexarError::DeviceError(format!(
                    "GPUDirect: post_send failed (rc={rc})"
                )));
            }
        }
        self.poll_cq_until_complete(self.send_cq)
    }

    /// Post a receive into a GPU MR and wait for completion.
    pub fn recv(&self, mr: &GpuMr, wr_id: u64) -> Result<()> {
        unsafe {
            let mut sge: ibverbs_sys::ibv_sge = std::mem::zeroed();
            sge.addr = mr.gpu_ptr();
            sge.length = mr.size() as u32;
            sge.lkey = mr.lkey();

            let mut wr: ibverbs_sys::ibv_recv_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;

            let mut bad_wr: *mut ibverbs_sys::ibv_recv_wr = ptr::null_mut();
            let ctx = (*self.qp).context;
            let ops = &mut (*ctx).ops;
            let rc =
                ops.post_recv.as_mut().unwrap()(self.qp, &mut wr as *mut _, &mut bad_wr as *mut _);
            if rc != 0 {
                return Err(NexarError::DeviceError(format!(
                    "GPUDirect: post_recv failed (rc={rc})"
                )));
            }
        }
        self.poll_cq_until_complete(self.recv_cq)
    }

    /// Default timeout for CQ polling.
    const CQ_POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    fn poll_cq_until_complete(&self, cq: *mut ibverbs_sys::ibv_cq) -> Result<()> {
        let start = std::time::Instant::now();
        unsafe {
            let mut wc = ibverbs_sys::ibv_wc::default();
            let ctx = (*self.qp).context;
            let ops = &mut (*ctx).ops;
            loop {
                let n = ops.poll_cq.as_mut().unwrap()(cq, 1, &mut wc as *mut _);
                if n < 0 {
                    return Err(NexarError::DeviceError("GPUDirect: poll_cq failed".into()));
                }
                if n > 0 {
                    if let Some((status, vendor_err)) = wc.error() {
                        return Err(NexarError::DeviceError(format!(
                            "GPUDirect: work completion failed \
                             (status={status:?}, vendor_err={vendor_err}, wr_id={})",
                            wc.wr_id()
                        )));
                    }
                    return Ok(());
                }
                if start.elapsed() > Self::CQ_POLL_TIMEOUT {
                    return Err(NexarError::DeviceError(format!(
                        "GPUDirect: CQ poll timed out after {}ms",
                        Self::CQ_POLL_TIMEOUT.as_millis()
                    )));
                }
                std::hint::spin_loop();
            }
        }
    }
}

impl Drop for GpuDirectQp {
    fn drop(&mut self) {
        unsafe {
            if !self.qp.is_null() {
                ibverbs_sys::ibv_destroy_qp(self.qp);
            }
            // CQs are owned per-QP; destroy after QP.
            if !self.send_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(self.send_cq);
            }
            if !self.recv_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(self.recv_cq);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback staging functions (when nvidia-peermem is not loaded)
// ---------------------------------------------------------------------------

/// Copy GPU memory to a host Vec via CUDA D2H copy.
///
/// Used as fallback when `nvidia-peermem` is not available: the data is staged
/// to host, then sent via regular RDMA (host MR).
pub fn stage_gpu_to_host(gpu_ptr: u64, size: usize) -> Result<Vec<u8>> {
    let mut host_buf = vec![0u8; size];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut host_buf, gpu_ptr).map_err(|e| {
            NexarError::DeviceError(format!("GPUDirect fallback D2H copy failed: {e}"))
        })?;
    }
    Ok(host_buf)
}

/// Copy host data into GPU memory via CUDA H2D copy.
///
/// Used as fallback when receiving data via regular RDMA into a host MR,
/// then staging it to the GPU.
pub fn stage_host_to_gpu(data: &[u8], gpu_ptr: u64) -> Result<()> {
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(gpu_ptr, data).map_err(|e| {
            NexarError::DeviceError(format!("GPUDirect fallback H2D copy failed: {e}"))
        })?;
    }
    Ok(())
}
