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

mod context;
mod mr;
mod qp;

pub use context::GpuDirectContext;
pub use mr::{GpuDirectPool, GpuMr, PooledGpuMr};
pub use qp::{
    GpuDirectEndpoint, GpuDirectQp, PreparedGpuDirectQp, stage_gpu_to_host, stage_host_to_gpu,
};
