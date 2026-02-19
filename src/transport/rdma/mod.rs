//! RDMA transport backend for InfiniBand/RoCE kernel bypass.
//!
//! Only compiled when the `rdma` feature is enabled.

mod connection;
mod memory;

pub use connection::{PreparedRdmaConnection, RdmaConnection, RdmaContext};
pub use memory::{RdmaMemoryPool, RdmaPooledBuf};
