//! Gradient compression for bandwidth-efficient allreduce.
//!
//! When network bandwidth is the bottleneck (e.g., cross-datacenter training),
//! compressed allreduce can reduce communication volume by 10-100x at the cost
//! of slight noise in gradient updates.
//!
//! # Available compressors
//!
//! - [`TopKCompressor`]: Keep the top K% of elements by magnitude. Best accuracy
//!   retention but O(n log n) due to sorting. Use for smaller tensors or when
//!   accuracy matters most.
//! - [`RandomKCompressor`]: Randomly sample K% of elements. O(n) and unbiased in
//!   expectation when combined with error feedback. Prefer for very large tensors.
//! - [`NoCompression`]: Identity pass-through. Useful as a baseline or when
//!   compression is conditionally disabled.
//!
//! # Usage with collectives
//!
//! Use [`crate::client::NexarClient::all_reduce_compressed`] (blocking) or
//! [`crate::client::NexarClient::all_reduce_compressed_nb`] (non-blocking). Both require:
//!
//! 1. A `&dyn Compressor` implementation.
//! 2. A `residual` buffer (same size as the tensor), zero-initialized on the
//!    first call and preserved across training steps. The residual accumulates
//!    compression error (error feedback) to maintain convergence.
//!
//! ```ignore
//! use nexar::compression::TopKCompressor;
//!
//! // Keep top 1% of gradients
//! let compressor = TopKCompressor::new(0.01);
//! let mut residual = vec![0u8; tensor_bytes];
//!
//! // Each training step:
//! unsafe {
//!     client.all_reduce_compressed(
//!         ptr, count, dtype, op,
//!         &compressor, &mut residual,
//!     ).await?;
//! }
//! ```
//!
//! # When to use compression
//!
//! - Cross-node allreduce over Ethernet (1-100 Gbps) — compression helps most
//! - Large gradient tensors where bandwidth dominates compute
//! - NOT recommended for intra-node (NVLink/PCIe) where bandwidth is abundant
//! - NOT recommended for very small tensors (compression overhead > savings)

pub mod none;
pub mod randomk;
pub mod topk;
pub mod traits;

pub use none::NoCompression;
pub use randomk::RandomKCompressor;
pub use topk::TopKCompressor;
pub use traits::{CompressedTensor, Compressor};
