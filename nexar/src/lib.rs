pub mod client;
pub mod cluster;
pub mod collective;
pub mod compression;
pub mod config;
pub mod device;
pub mod error;
pub mod memory;
pub mod protocol;
pub mod reduce;
mod reduce_simd;
pub(crate) mod reduce_types;
pub mod rpc;
pub mod transport;
pub mod types;

pub use client::{NexarClient, SyncClient};
pub use cluster::{SeedNode, WorkerNode};
pub use collective::{CollectiveGroup, CollectiveHandle};
pub use config::NexarConfig;
pub use device::{CpuAdapter, DeviceAdapter};
pub use error::{NexarError, Result};
pub use memory::{BufferPtr, BufferRef, Device, GlobalPtr, Host, MemorySpace};
pub use protocol::NexarMessage;
pub use transport::buffer_pool::PoolProfile;
pub use transport::{
    BulkTransport, PeerConnection, TaggedBulkTransport, TcpBulkTransport, TransportListener,
};
pub use types::{DataType, IoVec, NodeId, Priority, Rank, ReduceOp};
