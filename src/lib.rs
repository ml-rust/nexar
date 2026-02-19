pub mod client;
pub mod cluster;
pub mod collective;
pub mod device;
pub mod error;
pub mod memory;
pub mod protocol;
mod reduce;
pub mod rpc;
pub mod transport;
pub mod types;

pub use client::{NexarClient, SyncClient};
pub use cluster::{SeedNode, WorkerNode};
#[cfg(feature = "cuda")]
pub use device::CudaAdapter;
pub use device::{CpuAdapter, DeviceAdapter};
pub use error::{NexarError, Result};
pub use memory::GlobalPtr;
pub use protocol::NexarMessage;
pub use transport::{PeerConnection, TransportListener};
pub use types::{DataType, NodeId, Priority, Rank, ReduceOp};
