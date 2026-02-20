//! RDMA registered memory region.

use nexar::error::{NexarError, Result};

/// A registered RDMA memory region for send/recv.
pub struct RdmaMr {
    mr: *mut ibverbs_sys::ibv_mr,
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
}

unsafe impl Send for RdmaMr {}
unsafe impl Sync for RdmaMr {}

impl RdmaMr {
    pub(crate) fn new(mr: *mut ibverbs_sys::ibv_mr, ptr: *mut u8, size: usize) -> Self {
        Self { mr, ptr, size }
    }

    pub fn lkey(&self) -> u32 {
        unsafe { (*self.mr).lkey }
    }

    pub fn rkey(&self) -> u32 {
        unsafe { (*self.mr).rkey }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl std::ops::Deref for RdmaMr {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for RdmaMr {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for RdmaMr {
    fn drop(&mut self) {
        unsafe {
            if !self.mr.is_null() {
                ibverbs_sys::ibv_dereg_mr(self.mr);
            }
            if !self.ptr.is_null() {
                let _ = Vec::from_raw_parts(self.ptr, self.size, self.size);
            }
        }
    }
}

/// Send-safe wrapper around a raw CQ pointer.
///
/// RDMA CQ pointers are thread-safe (completions can be polled from any thread).
/// This wrapper allows them to cross `.await` points in `Send` futures.
#[derive(Clone, Copy)]
pub(crate) struct SendCq(*mut ibverbs_sys::ibv_cq);

unsafe impl Send for SendCq {}
unsafe impl Sync for SendCq {}

impl SendCq {
    pub(crate) fn new(cq: *mut ibverbs_sys::ibv_cq) -> Self {
        Self(cq)
    }

    fn raw(self) -> *mut ibverbs_sys::ibv_cq {
        self.0
    }
}

/// Wait for a CQ completion using the completion channel's file descriptor.
///
/// Uses `tokio::io::unix::AsyncFd` to sleep on the completion channel's fd
/// instead of spin-polling. This avoids exhausting Tokio's blocking thread
/// pool under high RDMA concurrency.
///
/// Flow:
/// 1. `ibv_req_notify_cq` — arm the CQ to signal the completion channel
/// 2. `AsyncFd::readable()` — yield to Tokio until the fd is readable
/// 3. `ibv_get_cq_event` — consume the event (blocks briefly, but fd is ready)
/// 4. `ibv_ack_cq_events` — acknowledge so the channel doesn't stall
/// 5. Poll the CQ for the actual work completion
pub(crate) async fn wait_for_completion(
    cq: SendCq,
    channel: &tokio::io::unix::AsyncFd<CompChannelFd>,
    timeout: std::time::Duration,
) -> Result<()> {
    // Arm the CQ notification before checking for existing completions.
    req_notify(cq)?;

    // Check if a completion is already available (avoid blocking if work
    // completed between post and notify arm).
    if let Some(result) = try_poll_cq(cq)? {
        return result;
    }

    // Wait for the completion channel fd to become readable.
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let wait = tokio::time::timeout_at(deadline, channel.readable());
        match wait.await {
            Ok(Ok(mut guard)) => {
                // fd is readable — consume the event.
                guard.clear_ready();
                drain_cq_events(cq, channel)?;

                // Poll the CQ for the work completion.
                if let Some(result) = try_poll_cq(cq)? {
                    return result;
                }

                // Spurious wakeup or event for a different WR. Re-arm and retry.
                req_notify(cq)?;
            }
            Ok(Err(e)) => {
                return Err(NexarError::device(format!("RDMA: AsyncFd error: {e}")));
            }
            Err(_) => {
                return Err(NexarError::device(format!(
                    "RDMA: CQ event timed out after {}ms",
                    timeout.as_millis()
                )));
            }
        }
    }
}

/// Arm CQ notification via the ibverbs ops table.
fn req_notify(cq: SendCq) -> Result<()> {
    unsafe {
        let raw = cq.raw();
        let ctx = (*raw).context;
        let ops = &mut (*ctx).ops;
        let rc = ops.req_notify_cq.as_mut().expect("req_notify_cq missing")(raw, 0);
        if rc != 0 {
            return Err(NexarError::device(format!(
                "RDMA: ibv_req_notify_cq failed (rc={rc})"
            )));
        }
    }
    Ok(())
}

/// Try to poll one completion from the CQ without blocking.
///
/// Returns:
/// - `Ok(Some(Ok(())))` — completion found, success
/// - `Ok(Some(Err(...)))` — completion found, error
/// - `Ok(None)` — no completion available yet
fn try_poll_cq(cq: SendCq) -> Result<Option<Result<()>>> {
    unsafe {
        let raw = cq.raw();
        let mut wc = ibverbs_sys::ibv_wc::default();
        let ctx = (*raw).context;
        let ops = &mut (*ctx).ops;
        let n = ops.poll_cq.as_mut().expect("poll_cq missing")(raw, 1, &mut wc as *mut _);
        if n < 0 {
            return Err(NexarError::device("RDMA: poll_cq failed"));
        }
        if n > 0 {
            if let Some((status, vendor_err)) = wc.error() {
                return Ok(Some(Err(NexarError::device(format!(
                    "RDMA: work completion failed (status={status:?}, vendor_err={vendor_err}, wr_id={})",
                    wc.wr_id()
                )))));
            }
            return Ok(Some(Ok(())));
        }
        Ok(None)
    }
}

/// Consume all pending CQ events from the completion channel.
fn drain_cq_events(cq: SendCq, channel: &tokio::io::unix::AsyncFd<CompChannelFd>) -> Result<()> {
    unsafe {
        let mut ev_cq: *mut ibverbs_sys::ibv_cq = std::ptr::null_mut();
        let mut ev_ctx: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = ibverbs_sys::ibv_get_cq_event(channel.get_ref().raw, &mut ev_cq, &mut ev_ctx);
        if rc != 0 {
            return Err(NexarError::device("RDMA: ibv_get_cq_event failed"));
        }
        ibverbs_sys::ibv_ack_cq_events(cq.raw(), 1);
    }
    Ok(())
}

/// Wrapper around a raw comp channel fd for use with `AsyncFd`.
pub(crate) struct CompChannelFd {
    raw: *mut ibverbs_sys::ibv_comp_channel,
}

unsafe impl Send for CompChannelFd {}
unsafe impl Sync for CompChannelFd {}

impl std::os::unix::io::AsRawFd for CompChannelFd {
    fn as_raw_fd(&self) -> std::os::unix::io::RawFd {
        unsafe { (*self.raw).fd }
    }
}

impl CompChannelFd {
    pub(crate) fn new(channel: *mut ibverbs_sys::ibv_comp_channel) -> Result<Self> {
        if channel.is_null() {
            return Err(NexarError::device("RDMA: null comp_channel"));
        }
        // Set the fd to non-blocking so AsyncFd works correctly.
        unsafe {
            let fd = (*channel).fd;
            let flags = libc::fcntl(fd, libc::F_GETFL);
            if flags < 0 {
                return Err(NexarError::device("RDMA: fcntl F_GETFL failed"));
            }
            if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
                return Err(NexarError::device("RDMA: fcntl F_SETFL O_NONBLOCK failed"));
            }
        }
        Ok(Self { raw: channel })
    }
}
