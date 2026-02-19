//! RDMA and GPUDirect mesh establishment for local bootstrap.

use crate::ext::PeerConnectionRdmaExt;
use crate::rdma::{RdmaContext, RdmaMemoryPool};
use nexar::NexarClient;
use std::sync::Arc;

/// Attempt to establish RDMA connections between all pairs in a local cluster.
///
/// Uses the extension slot on `PeerConnection` to attach RDMA state.
/// If RDMA device initialization fails (no IB hardware), logs a warning
/// and continues with QUIC-only transport.
pub async fn establish_rdma_mesh(clients: &[NexarClient]) {
    let ctx: Arc<RdmaContext> = match RdmaContext::new(None) {
        Ok(ctx) => Arc::new(ctx),
        Err(e) => {
            tracing::warn!("RDMA not available, using QUIC-only transport: {e}");
            return;
        }
    };

    let pool = match RdmaMemoryPool::new(&ctx, 16, 4 * 1024 * 1024) {
        Ok(pool) => pool,
        Err(e) => {
            tracing::warn!("RDMA memory pool creation failed: {e}");
            return;
        }
    };

    let n = clients.len();
    if n < 2 {
        return;
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let rank_i = clients[i].rank();
            let rank_j = clients[j].rank();

            let prepared_i = match ctx.prepare_connection(rank_j) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("RDMA prepare for {rank_i}->{rank_j} failed: {e}");
                    continue;
                }
            };
            let prepared_j = match ctx.prepare_connection(rank_i) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("RDMA prepare for {rank_j}->{rank_i} failed: {e}");
                    continue;
                }
            };

            let ep_i = prepared_i.endpoint();
            let ep_j = prepared_j.endpoint();

            let rdma_i = match prepared_i.complete(ep_j) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("RDMA handshake {rank_i}->{rank_j} failed: {e}");
                    continue;
                }
            };
            let rdma_j = match prepared_j.complete(ep_i) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("RDMA handshake {rank_j}->{rank_i} failed: {e}");
                    continue;
                }
            };

            // Attach RDMA connections via the extension slot.
            if let Ok(peer) = clients[i].peer(rank_j) {
                peer.set_rdma(rdma_i, Arc::clone(&pool));
            }
            if let Ok(peer) = clients[j].peer(rank_i) {
                peer.set_rdma(rdma_j, Arc::clone(&pool));
            }
        }
    }

    tracing::info!("RDMA mesh established for {} nodes", n);
}

/// Attempt to establish GPUDirect RDMA connections between all pairs.
#[cfg(feature = "gpudirect")]
pub fn establish_gpudirect_mesh(clients: &[NexarClient]) {
    use crate::ext::PeerConnectionGpuDirectExt;
    use crate::gpudirect::{GpuDirectContext, GpuDirectPool};

    let n = clients.len();
    if n < 2 {
        return;
    }

    let ctx = match GpuDirectContext::new(None) {
        Ok(ctx) => Arc::new(ctx),
        Err(e) => {
            tracing::warn!("GPUDirect not available: {e}");
            return;
        }
    };

    for i in 0..n {
        for j in (i + 1)..n {
            let rank_i = clients[i].rank();
            let rank_j = clients[j].rank();

            let prepared_i = match ctx.prepare_qp() {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("GPUDirect QP prepare for {rank_i}->{rank_j} failed: {e}");
                    continue;
                }
            };
            let prepared_j = match ctx.prepare_qp() {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("GPUDirect QP prepare for {rank_j}->{rank_i} failed: {e}");
                    continue;
                }
            };

            let ep_i = prepared_i.endpoint();
            let ep_j = prepared_j.endpoint();

            let qp_i = match prepared_i.complete(ep_j) {
                Ok(qp) => qp,
                Err(e) => {
                    tracing::warn!("GPUDirect handshake {rank_i}->{rank_j} failed: {e}");
                    continue;
                }
            };
            let qp_j = match prepared_j.complete(ep_i) {
                Ok(qp) => qp,
                Err(e) => {
                    tracing::warn!("GPUDirect handshake {rank_j}->{rank_i} failed: {e}");
                    continue;
                }
            };

            let empty_pool_i = Arc::new(GpuDirectPool::empty(Arc::clone(&ctx)));
            let empty_pool_j = Arc::new(GpuDirectPool::empty(Arc::clone(&ctx)));

            if let Ok(peer) = clients[i].peer(rank_j) {
                peer.set_gpudirect(qp_i, empty_pool_i);
            }
            if let Ok(peer) = clients[j].peer(rank_i) {
                peer.set_gpudirect(qp_j, empty_pool_j);
            }
        }
    }

    tracing::info!("GPUDirect RDMA mesh established for {n} nodes");
}
