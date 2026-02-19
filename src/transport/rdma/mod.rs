//! RDMA transport backend for InfiniBand/RoCE kernel bypass.
//!
//! Only compiled when the `rdma` feature is enabled.

mod connection;
#[cfg(feature = "gpudirect")]
pub mod gpudirect;
mod memory;

pub use connection::{PreparedRdmaConnection, RdmaConnection, RdmaContext};
pub use memory::{RdmaMemoryPool, RdmaPooledBuf};

#[cfg(feature = "gpudirect")]
pub use gpudirect::{
    GpuDirectContext, GpuDirectEndpoint, GpuDirectPool, GpuDirectQp, GpuMr, PreparedGpuDirectQp,
};
