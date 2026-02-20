use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::driver::CudaStream;
use cudarc::nccl::{result as nccl, safe::Id, sys};

use crate::error::{NcclCommError, Result};
use crate::types::{to_nccl_dtype, to_nccl_op};
use nexar::types::{DataType, ReduceOp};

/// Wrapper around a raw `ncclComm_t` handle for intra-node NCCL operations.
///
/// Uses cudarc's `result` layer directly (raw pointers) rather than the `safe`
/// layer, because we manage GPU pointers as `u64` matching nexar's pointer model.
pub struct NcclGroup {
    comm: sys::ncclComm_t,
    stream: Arc<CudaStream>,
    local_rank: usize,
    local_world: usize,
}

// SAFETY: ncclComm_t is thread-safe per NCCL documentation when used with
// proper stream synchronization. The CudaStream is already Send+Sync via Arc.
unsafe impl Send for NcclGroup {}
unsafe impl Sync for NcclGroup {}

impl NcclGroup {
    /// Initialize an NCCL communicator from a pre-shared unique ID.
    ///
    /// Each local rank must call this with the same `id` and `local_world`,
    /// but its own `local_rank`.
    pub fn init(
        stream: Arc<CudaStream>,
        local_rank: usize,
        local_world: usize,
        id: Id,
    ) -> Result<Self> {
        let mut comm = MaybeUninit::uninit();
        unsafe {
            nccl::comm_init_rank(
                comm.as_mut_ptr(),
                local_world as i32,
                *id_to_sys(&id),
                local_rank as i32,
            )?;
        }
        Ok(Self {
            comm: unsafe { comm.assume_init() },
            stream,
            local_rank,
            local_world,
        })
    }

    pub fn local_rank(&self) -> usize {
        self.local_rank
    }

    pub fn local_world(&self) -> usize {
        self.local_world
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// In-place allreduce on GPU memory.
    ///
    /// # Safety
    /// `ptr` must be a valid device pointer for `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn allreduce_inplace(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        let p = ptr as *mut c_void;
        unsafe {
            nccl::all_reduce(
                p as *const c_void,
                p,
                count,
                to_nccl_dtype(dtype),
                to_nccl_op(op),
                self.comm,
                self.cu_stream(),
            )?;
        }
        Ok(())
    }

    /// Allreduce with separate send/recv buffers.
    ///
    /// # Safety
    /// Both pointers must be valid device pointers for `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn allreduce(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe {
            nccl::all_reduce(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                count,
                to_nccl_dtype(dtype),
                to_nccl_op(op),
                self.comm,
                self.cu_stream(),
            )?;
        }
        Ok(())
    }

    /// Reduce-scatter: each rank gets 1/N of the reduced result.
    ///
    /// # Safety
    /// - `send_ptr`: `recv_count * local_world * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr`: `recv_count * dtype.size_in_bytes()` bytes.
    pub unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        recv_count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        unsafe {
            nccl::reduce_scatter(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                recv_count,
                to_nccl_dtype(dtype),
                to_nccl_op(op),
                self.comm,
                self.cu_stream(),
            )?;
        }
        Ok(())
    }

    /// Allgather: each rank contributes `send_count` elements.
    ///
    /// # Safety
    /// - `send_ptr`: `send_count * dtype.size_in_bytes()` bytes.
    /// - `recv_ptr`: `send_count * local_world * dtype.size_in_bytes()` bytes.
    pub unsafe fn allgather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        send_count: usize,
        dtype: DataType,
    ) -> Result<()> {
        unsafe {
            nccl::all_gather(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                send_count,
                to_nccl_dtype(dtype),
                self.comm,
                self.cu_stream(),
            )?;
        }
        Ok(())
    }

    /// Broadcast from `root` (local rank index) to all local ranks.
    ///
    /// # Safety
    /// `ptr` must be a valid device pointer for `count * dtype.size_in_bytes()` bytes.
    pub unsafe fn broadcast_inplace(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        root: usize,
    ) -> Result<()> {
        let p = ptr as *mut c_void;
        unsafe {
            nccl::broadcast(
                p as *const c_void,
                p,
                count,
                to_nccl_dtype(dtype),
                root as i32,
                self.comm,
                self.cu_stream(),
            )?;
        }
        Ok(())
    }

    /// Reduce to `root` (local rank index).
    ///
    /// # Safety
    /// `ptr` must be a valid device pointer for `count * dtype.size_in_bytes()` bytes.
    /// Result is only valid on `root`.
    pub unsafe fn reduce_inplace(
        &self,
        ptr: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        root: usize,
    ) -> Result<()> {
        let p = ptr as *mut c_void;
        unsafe {
            nccl::reduce(
                p as *const c_void,
                p,
                count,
                to_nccl_dtype(dtype),
                to_nccl_op(op),
                root as i32,
                self.comm,
                self.cu_stream(),
            )?;
        }
        Ok(())
    }

    /// Synchronize the CUDA stream (block until all NCCL ops complete).
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            cudarc::driver::result::stream::synchronize(self.stream.cu_stream() as _)
                .map_err(NcclCommError::CudaDriver)?;
        }
        Ok(())
    }

    /// Create a CUDA event (disable timing for lower overhead).
    pub fn create_event(&self) -> Result<cudarc::driver::sys::CUevent> {
        cudarc::driver::result::event::create(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        )
        .map_err(NcclCommError::CudaDriver)
    }

    /// Record an event on this group's NCCL stream.
    ///
    /// # Safety
    /// The event must have been created by `create_event` and not yet destroyed.
    pub unsafe fn record_event(&self, event: cudarc::driver::sys::CUevent) -> Result<()> {
        unsafe {
            cudarc::driver::result::event::record(event, self.stream.cu_stream())
                .map_err(NcclCommError::CudaDriver)?;
        }
        Ok(())
    }

    /// Make the given stream wait for an event (GPU-side dependency, no CPU block).
    ///
    /// # Safety
    /// The event and stream must be valid.
    pub unsafe fn stream_wait_event(
        &self,
        stream: &CudaStream,
        event: cudarc::driver::sys::CUevent,
    ) -> Result<()> {
        unsafe {
            cudarc::driver::result::stream::wait_event(
                stream.cu_stream(),
                event,
                cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
            .map_err(NcclCommError::CudaDriver)?;
        }
        Ok(())
    }

    /// Destroy a CUDA event.
    ///
    /// # Safety
    /// The event must have been created by `create_event` and not already destroyed.
    pub unsafe fn destroy_event(&self, event: cudarc::driver::sys::CUevent) -> Result<()> {
        unsafe {
            cudarc::driver::result::event::destroy(event).map_err(NcclCommError::CudaDriver)?;
        }
        Ok(())
    }

    fn cu_stream(&self) -> sys::cudaStream_t {
        self.stream.cu_stream() as sys::cudaStream_t
    }
}

impl Drop for NcclGroup {
    fn drop(&mut self) {
        unsafe {
            // comm_abort is the safest cleanup — it doesn't require stream sync.
            let _ = nccl::comm_abort(self.comm);
        }
    }
}

/// Convert our `Id` to the sys-level `ncclUniqueId`.
///
/// # Safety rationale
/// `ncclUniqueId` is `#[repr(C)]` with a single field: `internal: [c_char; 128]`.
/// `Id::internal()` returns `&[c_char; 128]` — the same memory layout as the
/// entire `ncclUniqueId` struct. The pointer cast is valid because the struct
/// has no padding and no other fields.
fn id_to_sys(id: &Id) -> &sys::ncclUniqueId {
    let internal = id.internal();
    unsafe { &*(internal as *const [std::ffi::c_char; 128] as *const sys::ncclUniqueId) }
}

/// Serialize an NCCL unique ID to bytes for network transfer.
pub fn id_to_bytes(id: &Id) -> Vec<u8> {
    let internal = id.internal();
    internal.iter().map(|&c| c as u8).collect()
}

/// Deserialize an NCCL unique ID from bytes received over the network.
pub fn id_from_bytes(bytes: &[u8]) -> Id {
    let mut internal = [0i8; 128];
    for (i, &b) in bytes.iter().enumerate().take(128) {
        internal[i] = b as i8;
    }
    Id::uninit(internal)
}
