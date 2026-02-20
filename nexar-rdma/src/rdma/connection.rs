//! RDMA queue pair management for reliable connected (RC) transport.
//!
//! Uses raw `ibverbs-sys` FFI (no safe wrapper).
//!
//! Two-phase construction:
//! 1. `RdmaContext::new()` — opens device, allocates PD and CQs.
//! 2. `RdmaContext::prepare_connection()` — creates a QP in INIT state.
//! 3. Exchange `RdmaEndpoint` via QUIC control channel.
//! 4. `PreparedRdmaConnection::complete()` — handshake to RTS.

use super::context::RdmaContext;
use super::mr::{RdmaMr, SendCq, wait_for_completion};
use ibverbs_sys::{ibv_qp_attr_mask, ibv_qp_state, ibv_send_flags, ibv_wr_opcode};
use nexar::error::{NexarError, Result};
use std::os::raw::c_int;
use std::ptr;
use std::sync::Arc;

/// Endpoint data exchanged between peers to complete an RDMA QP handshake.
#[derive(Debug, Clone, Copy)]
pub struct RdmaEndpoint {
    pub qp_num: u32,
    pub lid: u16,
    pub gid: [u8; 16],
}

const ENDPOINT_SIZE: usize = 22;

impl RdmaEndpoint {
    pub fn to_bytes(&self) -> [u8; ENDPOINT_SIZE] {
        let mut buf = [0u8; ENDPOINT_SIZE];
        buf[0..4].copy_from_slice(&self.qp_num.to_le_bytes());
        buf[4..6].copy_from_slice(&self.lid.to_le_bytes());
        buf[6..22].copy_from_slice(&self.gid);
        buf
    }

    pub fn from_bytes(buf: &[u8; ENDPOINT_SIZE]) -> Self {
        Self {
            qp_num: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            lid: u16::from_le_bytes([buf[4], buf[5]]),
            gid: [
                buf[6], buf[7], buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14],
                buf[15], buf[16], buf[17], buf[18], buf[19], buf[20], buf[21],
            ],
        }
    }
}

/// An RDMA connection that has been prepared but not yet handshaken.
pub struct PreparedRdmaConnection {
    pub(super) qp: *mut ibverbs_sys::ibv_qp,
    pub(super) send_cq: *mut ibverbs_sys::ibv_cq,
    pub(super) recv_cq: *mut ibverbs_sys::ibv_cq,
    pub(super) local_ep: RdmaEndpoint,
    pub(super) ctx: Arc<RdmaContext>,
}

unsafe impl Send for PreparedRdmaConnection {}
unsafe impl Sync for PreparedRdmaConnection {}

impl PreparedRdmaConnection {
    /// Get the local endpoint to exchange with the remote peer.
    pub fn endpoint(&self) -> RdmaEndpoint {
        self.local_ep
    }

    /// Complete the handshake with the remote peer's endpoint.
    pub fn complete(mut self, remote: RdmaEndpoint) -> Result<RdmaConnection> {
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
            attr.ah_attr.dlid = remote.lid;
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
                ibverbs_sys::ibv_destroy_qp(self.qp);
                self.qp = ptr::null_mut();
                return Err(NexarError::device(format!(
                    "RDMA: ibv_modify_qp to RTR failed (rc={rc})"
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
                ibverbs_sys::ibv_destroy_qp(self.qp);
                self.qp = ptr::null_mut();
                return Err(NexarError::device(format!(
                    "RDMA: ibv_modify_qp to RTS failed (rc={rc})"
                )));
            }

            let qp = self.qp;
            self.qp = ptr::null_mut();

            Ok(RdmaConnection {
                qp,
                send_cq: self.send_cq,
                recv_cq: self.recv_cq,
                ctx: Arc::clone(&self.ctx),
            })
        }
    }
}

impl Drop for PreparedRdmaConnection {
    fn drop(&mut self) {
        unsafe {
            if !self.qp.is_null() {
                ibverbs_sys::ibv_destroy_qp(self.qp);
                self.qp = ptr::null_mut();
            }
        }
    }
}

/// A fully connected RDMA RC queue pair to a single peer.
pub struct RdmaConnection {
    qp: *mut ibverbs_sys::ibv_qp,
    send_cq: *mut ibverbs_sys::ibv_cq,
    recv_cq: *mut ibverbs_sys::ibv_cq,
    ctx: Arc<RdmaContext>,
}

unsafe impl Send for RdmaConnection {}
unsafe impl Sync for RdmaConnection {}

/// Default timeout for CQ completion events (5 seconds).
const CQ_POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl RdmaConnection {
    /// Post an RDMA send and wait for completion via event-driven notification.
    pub async fn send_async(&mut self, mr: &RdmaMr, wr_id: u64) -> Result<()> {
        unsafe {
            let mut sge: ibverbs_sys::ibv_sge = std::mem::zeroed();
            sge.addr = mr.ptr as u64;
            sge.length = mr.size as u32;
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
                    "RDMA: post_send failed (rc={rc})"
                )));
            }
        }
        wait_for_completion(
            SendCq::new(self.send_cq),
            &self.ctx.send_async_fd,
            CQ_POLL_TIMEOUT,
        )
        .await
    }

    /// Post an RDMA receive and wait for completion via event-driven notification.
    pub async fn recv_async(&mut self, mr: &RdmaMr, wr_id: u64) -> Result<()> {
        unsafe {
            let mut sge: ibverbs_sys::ibv_sge = std::mem::zeroed();
            sge.addr = mr.ptr as u64;
            sge.length = mr.size as u32;
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
                    "RDMA: post_recv failed (rc={rc})"
                )));
            }
        }
        wait_for_completion(
            SendCq::new(self.recv_cq),
            &self.ctx.recv_async_fd,
            CQ_POLL_TIMEOUT,
        )
        .await
    }

    /// Post an RDMA send and wait for completion (blocking, for non-async contexts).
    pub fn send(&mut self, mr: &RdmaMr, wr_id: u64) -> Result<()> {
        unsafe {
            let mut sge: ibverbs_sys::ibv_sge = std::mem::zeroed();
            sge.addr = mr.ptr as u64;
            sge.length = mr.size as u32;
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
                    "RDMA: post_send failed (rc={rc})"
                )));
            }
        }
        self.wait_send_completion_sync()
    }

    /// Post an RDMA receive and wait for completion (blocking, for non-async contexts).
    pub fn recv(&mut self, mr: &RdmaMr, wr_id: u64) -> Result<()> {
        unsafe {
            let mut sge: ibverbs_sys::ibv_sge = std::mem::zeroed();
            sge.addr = mr.ptr as u64;
            sge.length = mr.size as u32;
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
                    "RDMA: post_recv failed (rc={rc})"
                )));
            }
        }
        self.wait_recv_completion_sync()
    }

    /// Synchronous CQ poll with tiered backoff (for blocking `send`/`recv`).
    fn poll_until_complete_sync(
        cq: *mut ibverbs_sys::ibv_cq,
        timeout: std::time::Duration,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        let mut iter = 0u32;
        loop {
            unsafe {
                let mut wc = ibverbs_sys::ibv_wc::default();
                let ctx = (*cq).context;
                let ops = &mut (*ctx).ops;
                let n = ops.poll_cq.as_mut().expect("poll_cq missing")(cq, 1, &mut wc as *mut _);
                if n < 0 {
                    return Err(NexarError::device("RDMA: poll_cq failed"));
                }
                if n > 0 {
                    if let Some((status, vendor_err)) = wc.error() {
                        return Err(NexarError::device(format!(
                            "RDMA: work completion failed (status={status:?}, vendor_err={vendor_err}, wr_id={})",
                            wc.wr_id()
                        )));
                    }
                    return Ok(());
                }
            }
            if start.elapsed() > timeout {
                return Err(NexarError::device(format!(
                    "RDMA: CQ poll timed out after {}ms",
                    timeout.as_millis()
                )));
            }
            if iter < 1000 {
                std::hint::spin_loop();
            } else if iter < 5000 {
                std::thread::sleep(std::time::Duration::from_micros(10));
            } else {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            iter = iter.saturating_add(1);
        }
    }

    fn wait_send_completion_sync(&self) -> Result<()> {
        Self::poll_until_complete_sync(self.send_cq, CQ_POLL_TIMEOUT)
    }

    fn wait_recv_completion_sync(&self) -> Result<()> {
        Self::poll_until_complete_sync(self.recv_cq, CQ_POLL_TIMEOUT)
    }
}

impl Drop for RdmaConnection {
    fn drop(&mut self) {
        unsafe {
            if !self.qp.is_null() {
                ibverbs_sys::ibv_destroy_qp(self.qp);
            }
        }
    }
}
