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

/// Poll a Mutex-protected CQ until at least one completion arrives, with timeout.
///
/// Uses tiered backoff: spin for 1000 iterations, then sleep 10µs for 4000
/// iterations, then sleep 100µs until timeout.
pub(crate) fn poll_until_complete(
    cq_lock: &std::sync::Mutex<()>,
    cq: *mut ibverbs_sys::ibv_cq,
    timeout: std::time::Duration,
) -> Result<()> {
    let start = std::time::Instant::now();
    let mut iter = 0u32;
    loop {
        {
            let _guard = cq_lock
                .lock()
                .map_err(|e| NexarError::device(format!("CQ lock poisoned: {e}")))?;
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
