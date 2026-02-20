use crate::error::{NexarError, Result};
use crate::transport::buffer_pool::PooledBuf;
use crate::types::Rank;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::NexarClient;
use super::async_client::RawRecvSource;

impl NexarClient {
    /// Send raw bytes to a peer.
    ///
    /// Uses comm-aware send for split communicators. Always uses QUIC transport.
    /// For bulk data in collectives, prefer `send_bytes_best_effort` which
    /// auto-selects RDMA when available.
    pub async fn send_bytes(&self, dest: Rank, data: &[u8]) -> Result<()> {
        let peer = self.peer(dest)?;
        if self.comm_id == 0 {
            peer.send_raw(data).await
        } else {
            peer.send_raw_comm(self.comm_id, data).await
        }
    }

    /// Send raw bytes using the best available transport (RDMA if available, QUIC fallback).
    /// For split communicators, always uses QUIC (comm-id routing required).
    pub(crate) async fn send_bytes_best_effort(&self, dest: Rank, data: &[u8]) -> Result<()> {
        let peer = self.peer(dest)?;
        if self.comm_id == 0 {
            peer.send_raw_best_effort(data).await
        } else {
            // Split communicators need comm_id tagging, QUIC only.
            peer.send_raw_comm(self.comm_id, data).await
        }
    }

    /// Send raw bytes to a peer with a u64 tag.
    ///
    /// Tagged sends are always via QUIC (tags are part of the wire format).
    pub async fn send_bytes_tagged(&self, dest: Rank, tag: u64, data: &[u8]) -> Result<()> {
        let peer = self.peer(dest)?;
        peer.send_raw_tagged(tag, data).await
    }

    /// Receive tagged bytes using the best available transport.
    ///
    /// Tries `TaggedBulkTransport` (TCP sidecar) first, falling back to QUIC.
    pub(crate) async fn recv_bytes_tagged_best_effort(
        &self,
        src: Rank,
        tag: u64,
        expected_size: usize,
    ) -> Result<PooledBuf> {
        let peer = self.peer(src)?;
        let tagged_bulk: Option<std::sync::Arc<dyn crate::transport::TaggedBulkTransport>> = peer
            .extension::<std::sync::Arc<dyn crate::transport::TaggedBulkTransport>>()?
            .map(|b| std::sync::Arc::clone(&*b));
        if let Some(bulk) = tagged_bulk
            && let Ok(data) = bulk.recv_bulk_tagged(tag, expected_size).await
        {
            return Ok(PooledBuf::from_vec(
                data,
                std::sync::Arc::clone(&self._pool),
            ));
        }
        self.recv_bytes_tagged(src, tag).await
    }

    /// Send tagged bytes using the best available transport.
    pub(crate) async fn send_bytes_tagged_best_effort(
        &self,
        dest: Rank,
        tag: u64,
        data: &[u8],
    ) -> Result<()> {
        let peer = self.peer(dest)?;
        peer.send_raw_tagged_best_effort(tag, data).await
    }

    /// Receive tagged raw bytes from a peer.
    ///
    /// The tag channel is lazily created and cached for the lifetime of this
    /// (rank, tag) pair. This allows multi-round algorithms (like ring
    /// allreduce) to use the same channel across rounds without losing
    /// messages that arrive between rounds.
    pub async fn recv_bytes_tagged(&self, src: Rank, tag: u64) -> Result<PooledBuf> {
        let original_src = self.resolve_rank(src);
        let key = (original_src, tag);

        // Get or create the receiver for this (rank, tag) pair.
        let rx_arc = {
            let mut map = self.tagged_receivers.lock().await;
            if let Some(rx) = map.get(&key) {
                Arc::clone(rx)
            } else {
                let router = self
                    .routers
                    .get(&original_src)
                    .ok_or(NexarError::UnknownPeer { rank: src })?;
                let rx = router.register_tag(tag).await;
                let rx_arc = Arc::new(Mutex::new(rx));
                map.insert(key, Arc::clone(&rx_arc));
                rx_arc
            }
        };

        let mut rx = rx_arc.lock().await;
        rx.recv()
            .await
            .ok_or(NexarError::PeerDisconnected { rank: src })
    }

    /// Receive raw bytes using the best available transport.
    ///
    /// Tries `BulkTransport::recv_bulk` first (e.g., RDMA), falling back to QUIC.
    /// Only works for the default communicator (comm_id 0) and requires knowing
    /// the expected size.
    #[allow(dead_code)]
    pub(crate) async fn recv_bytes_best_effort(
        &self,
        src: Rank,
        expected_size: usize,
    ) -> Result<PooledBuf> {
        if self.comm_id == 0 {
            let peer = self.peer(src)?;
            // Try BulkTransport recv_bulk if available.
            let bulk: Option<std::sync::Arc<dyn crate::transport::BulkTransport>> = peer
                .extension::<std::sync::Arc<dyn crate::transport::BulkTransport>>()?
                .map(|b| std::sync::Arc::clone(&*b));
            if let Some(bulk) = bulk {
                match bulk.recv_bulk(expected_size).await {
                    Ok(data) => {
                        return Ok(PooledBuf::from_vec(
                            data,
                            std::sync::Arc::clone(&self._pool),
                        ));
                    }
                    Err(e) => {
                        tracing::warn!(
                            src,
                            expected_size,
                            error = %e,
                            "bulk transport recv failed, falling back to QUIC"
                        );
                    }
                }
            }
        }
        // Fallback to QUIC recv.
        self.recv_bytes(src).await
    }

    /// Receive raw bytes from a peer.
    ///
    /// Uses comm-aware recv for split communicators.
    pub async fn recv_bytes(&self, src: Rank) -> Result<PooledBuf> {
        match &self.raw_recv {
            RawRecvSource::Router => {
                let original_src = self.resolve_rank(src);
                let router = self
                    .routers
                    .get(&original_src)
                    .ok_or(NexarError::UnknownPeer { rank: src })?;
                router.recv_raw(original_src).await
            }
            RawRecvSource::Comm(channels) => {
                let rx = channels
                    .get(&src)
                    .ok_or(NexarError::UnknownPeer { rank: src })?;
                rx.lock()
                    .await
                    .recv()
                    .await
                    .ok_or(NexarError::PeerDisconnected { rank: src })
            }
        }
    }
}
