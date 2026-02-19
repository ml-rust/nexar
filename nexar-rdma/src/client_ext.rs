//! Extension traits for `NexarClient` that add GPU-direct send/recv.

#[cfg(feature = "gpudirect")]
mod gpudirect_client {
    use crate::ext::PeerConnectionGpuDirectExt;
    use nexar::NexarClient;
    use nexar::error::{NexarError, Result};
    use nexar::types::Rank;

    /// Extension trait adding GPUDirect send/recv to `NexarClient`.
    pub trait NexarClientRdmaExt {
        /// Send raw bytes directly from GPU memory to a peer.
        fn send_bytes_gpu(
            &self,
            dest: Rank,
            gpu_ptr: u64,
            size: usize,
        ) -> impl std::future::Future<Output = Result<()>> + Send;

        /// Receive raw bytes directly into GPU memory from a peer.
        fn recv_bytes_gpu(
            &self,
            src: Rank,
            gpu_ptr: u64,
            size: usize,
        ) -> impl std::future::Future<Output = Result<()>> + Send;
    }

    impl NexarClientRdmaExt for NexarClient {
        async fn send_bytes_gpu(&self, dest: Rank, gpu_ptr: u64, size: usize) -> Result<()> {
            if dest >= self.world_size() {
                return Err(NexarError::InvalidRank {
                    rank: dest,
                    world_size: self.world_size(),
                });
            }
            let peer = self.peer(dest)?;
            peer.send_raw_gpu(gpu_ptr, size).await
        }

        async fn recv_bytes_gpu(&self, src: Rank, gpu_ptr: u64, size: usize) -> Result<()> {
            if src >= self.world_size() {
                return Err(NexarError::InvalidRank {
                    rank: src,
                    world_size: self.world_size(),
                });
            }
            let peer = self.peer(src)?;
            peer.recv_raw_gpu(gpu_ptr, size).await
        }
    }
}

#[cfg(feature = "gpudirect")]
pub use gpudirect_client::NexarClientRdmaExt;
