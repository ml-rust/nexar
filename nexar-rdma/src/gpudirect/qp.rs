//! GPUDirect queue pair: creation, handshake, send/recv operations.

use ibverbs_sys::{
    ibv_access_flags, ibv_qp_attr_mask, ibv_qp_state, ibv_qp_type, ibv_send_flags, ibv_wr_opcode,
};
use nexar::error::{NexarError, Result};
use std::os::raw::c_int;
use std::ptr;

use super::context::GpuDirectContext;
use super::mr::GpuMr;

/// Endpoint data exchanged between peers to complete a GPUDirect QP handshake.
#[derive(Debug, Clone, Copy)]
pub struct GpuDirectEndpoint {
    pub qp_num: u32,
    pub gid: [u8; 16],
}

const ENDPOINT_SIZE: usize = 20;

impl GpuDirectEndpoint {
    pub fn to_bytes(&self) -> [u8; ENDPOINT_SIZE] {
        let mut buf = [0u8; ENDPOINT_SIZE];
        buf[0..4].copy_from_slice(&self.qp_num.to_le_bytes());
        buf[4..20].copy_from_slice(&self.gid);
        buf
    }

    pub fn from_bytes(buf: &[u8; ENDPOINT_SIZE]) -> Self {
        // SAFETY: buf is exactly ENDPOINT_SIZE bytes (enforced by type).
        Self {
            qp_num: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            gid: [
                buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11], buf[12], buf[13],
                buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
            ],
        }
    }
}

/// A GPUDirect QP in INIT state, ready for endpoint exchange and handshake.
pub struct PreparedGpuDirectQp {
    qp: *mut ibverbs_sys::ibv_qp,
    send_cq: *mut ibverbs_sys::ibv_cq,
    recv_cq: *mut ibverbs_sys::ibv_cq,
    local_ep: GpuDirectEndpoint,
}

unsafe impl Send for PreparedGpuDirectQp {}
unsafe impl Sync for PreparedGpuDirectQp {}

impl PreparedGpuDirectQp {
    /// Create a QP in INIT state from a GpuDirectContext.
    pub(super) fn create(ctx: &GpuDirectContext) -> Result<Self> {
        unsafe {
            let send_cq =
                ibverbs_sys::ibv_create_cq(ctx.ctx, 256, ptr::null_mut(), ptr::null_mut(), 0);
            if send_cq.is_null() {
                return Err(NexarError::device("GPUDirect: ibv_create_cq (send) failed"));
            }
            let recv_cq =
                ibverbs_sys::ibv_create_cq(ctx.ctx, 256, ptr::null_mut(), ptr::null_mut(), 0);
            if recv_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(send_cq);
                return Err(NexarError::device("GPUDirect: ibv_create_cq (recv) failed"));
            }

            let mut qp_init_attr: ibverbs_sys::ibv_qp_init_attr = std::mem::zeroed();
            qp_init_attr.qp_type = ibv_qp_type::IBV_QPT_RC;
            qp_init_attr.send_cq = send_cq;
            qp_init_attr.recv_cq = recv_cq;
            qp_init_attr.cap.max_send_wr = 128;
            qp_init_attr.cap.max_recv_wr = 128;
            qp_init_attr.cap.max_send_sge = 1;
            qp_init_attr.cap.max_recv_sge = 1;

            let qp = ibverbs_sys::ibv_create_qp(ctx.pd, &mut qp_init_attr);
            if qp.is_null() {
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                return Err(NexarError::device("GPUDirect: ibv_create_qp failed"));
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
                return Err(NexarError::device(format!(
                    "GPUDirect: ibv_modify_qp to INIT failed (rc={rc})"
                )));
            }

            // Query GID for addressing (works for both IB and RoCE).
            let mut gid: ibverbs_sys::ibv_gid = std::mem::zeroed();
            let rc = ibverbs_sys::ibv_query_gid(ctx.ctx, 1, 0, &mut gid);
            if rc != 0 {
                ibverbs_sys::ibv_destroy_qp(qp);
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                return Err(NexarError::device(format!(
                    "GPUDirect: ibv_query_gid failed (rc={rc})"
                )));
            }

            let local_ep = GpuDirectEndpoint {
                qp_num: (*qp).qp_num,
                gid: gid.raw,
            };

            Ok(Self {
                qp,
                send_cq,
                recv_cq,
                local_ep,
            })
        }
    }

    pub fn endpoint(&self) -> GpuDirectEndpoint {
        self.local_ep
    }

    /// Complete the handshake: INIT → RTR → RTS.
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
                let qp = self.qp;
                self.qp = ptr::null_mut();
                ibverbs_sys::ibv_destroy_qp(qp);
                return Err(NexarError::device(format!(
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
                return Err(NexarError::device(format!(
                    "GPUDirect: ibv_modify_qp to RTS failed (rc={rc})"
                )));
            }

            // Transfer ownership to GpuDirectQp.
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

impl Drop for PreparedGpuDirectQp {
    fn drop(&mut self) {
        unsafe {
            if !self.qp.is_null() {
                ibverbs_sys::ibv_destroy_qp(self.qp);
                self.qp = ptr::null_mut();
            }
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
            let rc = ops.post_send.as_mut().expect("post_send missing")(
                self.qp,
                &mut wr as *mut _,
                &mut bad_wr as *mut _,
            );
            if rc != 0 {
                return Err(NexarError::device(format!(
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
            let rc = ops.post_recv.as_mut().expect("post_recv missing")(
                self.qp,
                &mut wr as *mut _,
                &mut bad_wr as *mut _,
            );
            if rc != 0 {
                return Err(NexarError::device(format!(
                    "GPUDirect: post_recv failed (rc={rc})"
                )));
            }
        }
        self.poll_cq_until_complete(self.recv_cq)
    }

    const CQ_POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    fn poll_cq_until_complete(&self, cq: *mut ibverbs_sys::ibv_cq) -> Result<()> {
        let start = std::time::Instant::now();
        unsafe {
            let mut wc = ibverbs_sys::ibv_wc::default();
            let ctx = (*self.qp).context;
            let ops = &mut (*ctx).ops;
            loop {
                let n = ops.poll_cq.as_mut().expect("poll_cq missing")(cq, 1, &mut wc as *mut _);
                if n < 0 {
                    return Err(NexarError::device("GPUDirect: poll_cq failed"));
                }
                if n > 0 {
                    if let Some((status, vendor_err)) = wc.error() {
                        return Err(NexarError::device(format!(
                            "GPUDirect: work completion failed \
                             (status={status:?}, vendor_err={vendor_err}, wr_id={})",
                            wc.wr_id()
                        )));
                    }
                    return Ok(());
                }
                if start.elapsed() > Self::CQ_POLL_TIMEOUT {
                    return Err(NexarError::device(format!(
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
            if !self.send_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(self.send_cq);
            }
            if !self.recv_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(self.recv_cq);
            }
        }
    }
}

/// Copy GPU memory to a host Vec via CUDA D2H copy.
pub fn stage_gpu_to_host(gpu_ptr: u64, size: usize) -> Result<Vec<u8>> {
    let mut host_buf = vec![0u8; size];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut host_buf, gpu_ptr)
            .map_err(|e| NexarError::device(format!("GPUDirect fallback D2H copy failed: {e}")))?;
    }
    Ok(host_buf)
}

/// Copy host data into GPU memory via CUDA H2D copy.
pub fn stage_host_to_gpu(data: &[u8], gpu_ptr: u64) -> Result<()> {
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(gpu_ptr, data)
            .map_err(|e| NexarError::device(format!("GPUDirect fallback H2D copy failed: {e}")))?;
    }
    Ok(())
}
