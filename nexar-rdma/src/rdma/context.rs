//! Shared per-device RDMA resources: device, protection domain, completion queues.

use super::connection::{PreparedRdmaConnection, RdmaEndpoint};
use super::mr::{CompChannelFd, RdmaMr};
use ibverbs_sys::ibv_access_flags;
use nexar::error::{NexarError, Result};
use nexar::types::Rank;
use std::os::raw::c_int;
use std::ptr;
use std::sync::Arc;
use tokio::io::unix::AsyncFd;

/// Shared per-device RDMA resources.
///
/// Owns the ibverbs `Context`, protection domain, completion queues, and
/// completion channels. All RDMA connections created from this context
/// share these resources.
pub struct RdmaContext {
    pub(super) ctx: *mut ibverbs_sys::ibv_context,
    pub(super) pd: *mut ibverbs_sys::ibv_pd,
    pub(super) send_cq: *mut ibverbs_sys::ibv_cq,
    pub(super) recv_cq: *mut ibverbs_sys::ibv_cq,
    send_channel: *mut ibverbs_sys::ibv_comp_channel,
    recv_channel: *mut ibverbs_sys::ibv_comp_channel,
    /// AsyncFd wrappers for event-driven CQ completion (non-blocking).
    pub(super) send_async_fd: AsyncFd<CompChannelFd>,
    pub(super) recv_async_fd: AsyncFd<CompChannelFd>,
}

unsafe impl Send for RdmaContext {}
unsafe impl Sync for RdmaContext {}

impl RdmaContext {
    /// Open an RDMA device and allocate shared resources.
    ///
    /// `device_index` selects which RDMA device to use (default: first).
    /// Creates a protection domain, two completion channels, and two CQs
    /// (256 entries each) bound to those channels for event-driven completion.
    pub fn new(device_index: Option<usize>) -> Result<Self> {
        unsafe {
            let mut num_devices: c_int = 0;
            let dev_list = ibverbs_sys::ibv_get_device_list(&mut num_devices);
            if dev_list.is_null() || num_devices == 0 {
                return Err(NexarError::device("RDMA: no devices found"));
            }

            let idx = device_index.unwrap_or(0);
            if idx >= num_devices as usize {
                ibverbs_sys::ibv_free_device_list(dev_list);
                return Err(NexarError::device(format!(
                    "RDMA: device index {idx} out of range (have {num_devices})"
                )));
            }

            let dev = *dev_list.add(idx);
            let ctx = ibverbs_sys::ibv_open_device(dev);
            ibverbs_sys::ibv_free_device_list(dev_list);

            if ctx.is_null() {
                return Err(NexarError::device("RDMA: ibv_open_device failed"));
            }

            let pd = ibverbs_sys::ibv_alloc_pd(ctx);
            if pd.is_null() {
                ibverbs_sys::ibv_close_device(ctx);
                return Err(NexarError::device("RDMA: ibv_alloc_pd failed"));
            }

            // Create completion channels for event-driven CQ notification.
            let send_channel = ibverbs_sys::ibv_create_comp_channel(ctx);
            if send_channel.is_null() {
                ibverbs_sys::ibv_dealloc_pd(pd);
                ibverbs_sys::ibv_close_device(ctx);
                return Err(NexarError::device(
                    "RDMA: ibv_create_comp_channel (send) failed",
                ));
            }

            let recv_channel = ibverbs_sys::ibv_create_comp_channel(ctx);
            if recv_channel.is_null() {
                ibverbs_sys::ibv_destroy_comp_channel(send_channel);
                ibverbs_sys::ibv_dealloc_pd(pd);
                ibverbs_sys::ibv_close_device(ctx);
                return Err(NexarError::device(
                    "RDMA: ibv_create_comp_channel (recv) failed",
                ));
            }

            // Create CQs bound to their completion channels.
            let send_cq = ibverbs_sys::ibv_create_cq(ctx, 256, ptr::null_mut(), send_channel, 0);
            if send_cq.is_null() {
                ibverbs_sys::ibv_destroy_comp_channel(recv_channel);
                ibverbs_sys::ibv_destroy_comp_channel(send_channel);
                ibverbs_sys::ibv_dealloc_pd(pd);
                ibverbs_sys::ibv_close_device(ctx);
                return Err(NexarError::device("RDMA: ibv_create_cq (send) failed"));
            }

            let recv_cq = ibverbs_sys::ibv_create_cq(ctx, 256, ptr::null_mut(), recv_channel, 1);
            if recv_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(send_cq);
                ibverbs_sys::ibv_destroy_comp_channel(recv_channel);
                ibverbs_sys::ibv_destroy_comp_channel(send_channel);
                ibverbs_sys::ibv_dealloc_pd(pd);
                ibverbs_sys::ibv_close_device(ctx);
                return Err(NexarError::device("RDMA: ibv_create_cq (recv) failed"));
            }

            // Wrap comp channel fds for async I/O (sets O_NONBLOCK).
            let send_fd = CompChannelFd::new(send_channel).inspect_err(|_e| {
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                ibverbs_sys::ibv_destroy_comp_channel(recv_channel);
                ibverbs_sys::ibv_destroy_comp_channel(send_channel);
                ibverbs_sys::ibv_dealloc_pd(pd);
                ibverbs_sys::ibv_close_device(ctx);
            })?;
            let recv_fd = CompChannelFd::new(recv_channel).inspect_err(|_e| {
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                ibverbs_sys::ibv_destroy_comp_channel(recv_channel);
                ibverbs_sys::ibv_destroy_comp_channel(send_channel);
                ibverbs_sys::ibv_dealloc_pd(pd);
                ibverbs_sys::ibv_close_device(ctx);
            })?;

            let send_async_fd = AsyncFd::new(send_fd).map_err(|e| {
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                ibverbs_sys::ibv_destroy_comp_channel(recv_channel);
                ibverbs_sys::ibv_destroy_comp_channel(send_channel);
                ibverbs_sys::ibv_dealloc_pd(pd);
                ibverbs_sys::ibv_close_device(ctx);
                NexarError::device(format!("RDMA: AsyncFd (send) failed: {e}"))
            })?;
            let recv_async_fd = AsyncFd::new(recv_fd).map_err(|e| {
                ibverbs_sys::ibv_destroy_cq(recv_cq);
                ibverbs_sys::ibv_destroy_cq(send_cq);
                ibverbs_sys::ibv_destroy_comp_channel(recv_channel);
                ibverbs_sys::ibv_destroy_comp_channel(send_channel);
                ibverbs_sys::ibv_dealloc_pd(pd);
                ibverbs_sys::ibv_close_device(ctx);
                NexarError::device(format!("RDMA: AsyncFd (recv) failed: {e}"))
            })?;

            Ok(Self {
                ctx,
                pd,
                send_cq,
                recv_cq,
                send_channel,
                recv_channel,
                send_async_fd,
                recv_async_fd,
            })
        }
    }

    /// Allocate a registered memory region of `size` bytes.
    pub fn allocate(&self, size: usize) -> Result<RdmaMr> {
        unsafe {
            let buf = vec![0u8; size];
            let boxed = buf.into_boxed_slice();
            let ptr = Box::into_raw(boxed) as *mut u8;

            let access = ibverbs_sys::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | ibverbs_sys::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | ibverbs_sys::ibv_access_flags::IBV_ACCESS_REMOTE_READ;

            let mr = ibverbs_sys::ibv_reg_mr(self.pd, ptr as *mut _, size, access.0 as c_int);
            if mr.is_null() {
                // Convert back to box and drop
                let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, size));
                return Err(NexarError::device(format!(
                    "RDMA: ibv_reg_mr failed for size={size}"
                )));
            }

            Ok(RdmaMr::new(mr, ptr, size))
        }
    }

    /// Prepare a new RDMA connection (QP in INIT state) for the given peer.
    pub fn prepare_connection(
        self: &Arc<Self>,
        _peer_rank: Rank,
    ) -> Result<PreparedRdmaConnection> {
        unsafe {
            let mut qp_init_attr: ibverbs_sys::ibv_qp_init_attr = std::mem::zeroed();
            qp_init_attr.qp_type = ibverbs_sys::ibv_qp_type::IBV_QPT_RC;
            qp_init_attr.send_cq = self.send_cq;
            qp_init_attr.recv_cq = self.recv_cq;
            qp_init_attr.cap.max_send_wr = 128;
            qp_init_attr.cap.max_recv_wr = 128;
            qp_init_attr.cap.max_send_sge = 1;
            qp_init_attr.cap.max_recv_sge = 1;

            let qp = ibverbs_sys::ibv_create_qp(self.pd, &mut qp_init_attr);
            if qp.is_null() {
                return Err(NexarError::device("RDMA: ibv_create_qp failed"));
            }

            // Transition to INIT.
            let mut attr: ibverbs_sys::ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ibverbs_sys::ibv_qp_state::IBV_QPS_INIT;
            attr.pkey_index = 0;
            attr.port_num = 1;
            attr.qp_access_flags = (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | ibv_access_flags::IBV_ACCESS_REMOTE_READ)
                .0;

            let mask = ibverbs_sys::ibv_qp_attr_mask::IBV_QP_STATE
                | ibverbs_sys::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
                | ibverbs_sys::ibv_qp_attr_mask::IBV_QP_PORT
                | ibverbs_sys::ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;

            let rc = ibverbs_sys::ibv_modify_qp(qp, &mut attr, mask.0 as c_int);
            if rc != 0 {
                ibverbs_sys::ibv_destroy_qp(qp);
                return Err(NexarError::device(format!(
                    "RDMA: ibv_modify_qp to INIT failed (rc={rc})"
                )));
            }

            // LID is primarily used for InfiniBand; RoCE uses GID-based routing.
            // For simplicity, we use LID 0 for RoCE environments.
            let lid: u16 = 0;

            // Query GID for addressing (works for both IB and RoCE).
            let mut gid: ibverbs_sys::ibv_gid = std::mem::zeroed();
            let rc = ibverbs_sys::ibv_query_gid(self.ctx, 1, 0, &mut gid);
            if rc != 0 {
                ibverbs_sys::ibv_destroy_qp(qp);
                return Err(NexarError::device(format!(
                    "RDMA: ibv_query_gid failed (rc={rc})"
                )));
            }

            let local_ep = RdmaEndpoint {
                qp_num: (*qp).qp_num,
                lid,
                gid: gid.raw,
            };

            Ok(PreparedRdmaConnection {
                qp,
                send_cq: self.send_cq,
                recv_cq: self.recv_cq,
                local_ep,
                ctx: Arc::clone(self),
            })
        }
    }
}

impl Drop for RdmaContext {
    fn drop(&mut self) {
        unsafe {
            if !self.send_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(self.send_cq);
            }
            if !self.recv_cq.is_null() {
                ibverbs_sys::ibv_destroy_cq(self.recv_cq);
            }
            if !self.send_channel.is_null() {
                ibverbs_sys::ibv_destroy_comp_channel(self.send_channel);
            }
            if !self.recv_channel.is_null() {
                ibverbs_sys::ibv_destroy_comp_channel(self.recv_channel);
            }
            if !self.pd.is_null() {
                ibverbs_sys::ibv_dealloc_pd(self.pd);
            }
            if !self.ctx.is_null() {
                ibverbs_sys::ibv_close_device(self.ctx);
            }
        }
    }
}
