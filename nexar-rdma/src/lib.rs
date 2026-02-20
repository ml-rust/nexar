//! RDMA and GPUDirect transport extensions for nexar.
//!
//! This crate provides InfiniBand/RoCE kernel-bypass transport as an extension
//! to nexar's QUIC-based communication. It uses the extension slot on
//! `PeerConnection` to attach RDMA state without requiring any `#[cfg]` flags
//! in the core nexar crate.
//!
//! # Features
//!
//! - **default** — RDMA transport via ibverbs (InfiniBand/RoCE)
//! - **gpudirect** — GPUDirect RDMA: NIC reads/writes GPU memory directly
//!
//! # Usage
//!
//! ```ignore
//! use nexar_rdma::ext::PeerConnectionRdmaExt;
//!
//! // After bootstrap, each rank calls establish_rdma_mesh:
//! nexar_rdma::bootstrap::establish_rdma_mesh(&client).await;
//!
//! // Send via RDMA with QUIC fallback:
//! peer.send_raw_rdma(data).await?;
//! ```

pub mod bootstrap;
pub mod client_ext;
#[cfg(feature = "gpudirect")]
pub mod cuda_adapter;
pub mod ext;
#[cfg(feature = "gpudirect")]
pub mod gpudirect;
pub mod rdma;

#[cfg(feature = "gpudirect")]
pub use cuda_adapter::CudaAdapter;
pub use ext::PeerConnectionRdmaExt;
pub use rdma::{RdmaConnection, RdmaContext, RdmaMemoryPool};

#[cfg(feature = "gpudirect")]
pub use client_ext::NexarClientRdmaExt;
#[cfg(feature = "gpudirect")]
pub use ext::PeerConnectionGpuDirectExt;
#[cfg(feature = "gpudirect")]
pub use gpudirect::{GpuDirectContext, GpuDirectPool, GpuDirectQp, GpuMr};
