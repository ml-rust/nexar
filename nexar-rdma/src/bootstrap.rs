//! RDMA and GPUDirect mesh establishment.
//!
//! Two variants:
//! - `establish_rdma_mesh` / `establish_gpudirect_mesh`: multi-node, exchanges
//!   endpoint info over QUIC via `NexarClient::send_bytes()`/`recv_bytes()`.
//! - `establish_rdma_mesh_local` / `establish_gpudirect_mesh_local`: in-process,
//!   for tests where all clients share a process.

use crate::ext::PeerConnectionRdmaExt;
use crate::rdma::{RdmaContext, RdmaMemoryPool};
use nexar::NexarClient;
use nexar::error::Result;
use std::sync::Arc;

/// Size of a serialized `QueuePairEndpoint`: lid(2) + num(4) + has_gid(1) + gid(16) = 23.
const QP_ENDPOINT_SIZE: usize = 23;

/// Serialize an ibverbs `QueuePairEndpoint` to bytes.
fn serialize_qp_endpoint(ep: &ibverbs::QueuePairEndpoint) -> [u8; QP_ENDPOINT_SIZE] {
    let mut buf = [0u8; QP_ENDPOINT_SIZE];
    buf[0..2].copy_from_slice(&ep.lid.to_le_bytes());
    buf[2..6].copy_from_slice(&ep.num.to_le_bytes());
    if let Some(gid) = &ep.gid {
        buf[6] = 1;
        // Reconstruct raw bytes from the two u64 components (big-endian, network order).
        buf[7..15].copy_from_slice(&gid.subnet_prefix().to_be_bytes());
        buf[15..23].copy_from_slice(&gid.interface_id().to_be_bytes());
    }
    buf
}

/// Deserialize an ibverbs `QueuePairEndpoint` from bytes.
fn deserialize_qp_endpoint(buf: &[u8]) -> ibverbs::QueuePairEndpoint {
    let lid = u16::from_le_bytes([buf[0], buf[1]]);
    let num = u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]);
    let gid = if buf[6] == 1 {
        let mut raw = [0u8; 16];
        raw.copy_from_slice(&buf[7..23]);
        // Safety: ibverbs::Gid is a #[repr(transparent)] wrapper around [u8; 16].
        Some(unsafe { std::mem::transmute::<[u8; 16], ibverbs::Gid>(raw) })
    } else {
        None
    };
    ibverbs::QueuePairEndpoint { num, lid, gid }
}

/// Establish RDMA connections to all peers via QUIC endpoint exchange.
///
/// Called collectively by all ranks. Uses QUIC send/recv to exchange QP endpoint
/// data, then completes the RDMA handshake.
///
/// Lower-ranked peer sends first to avoid deadlock.
pub async fn establish_rdma_mesh(client: &NexarClient) -> Result<()> {
    let ctx: Arc<RdmaContext> = match RdmaContext::new(None) {
        Ok(ctx) => Arc::new(ctx),
        Err(e) => {
            tracing::warn!("RDMA not available, using QUIC-only transport: {e}");
            return Ok(());
        }
    };

    let pool = match RdmaMemoryPool::new(&ctx, 16, 4 * 1024 * 1024) {
        Ok(pool) => pool,
        Err(e) => {
            tracing::warn!("RDMA memory pool creation failed: {e}");
            return Ok(());
        }
    };

    let my_rank = client.rank();
    let world = client.world_size();

    // Connect to each peer in sorted order.
    for peer_rank in 0..world {
        if peer_rank == my_rank {
            continue;
        }

        let prepared = match ctx.prepare_connection(peer_rank) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("RDMA prepare for {my_rank}->{peer_rank} failed: {e}");
                continue;
            }
        };

        let local_ep = prepared.endpoint();
        let local_bytes = serialize_qp_endpoint(&local_ep);

        // Lower rank sends first to avoid deadlock.
        let remote_bytes = if my_rank < peer_rank {
            client.send_bytes(peer_rank, &local_bytes).await?;
            let buf = client.recv_bytes(peer_rank).await?;
            buf.to_vec()
        } else {
            let buf = client.recv_bytes(peer_rank).await?;
            client.send_bytes(peer_rank, &local_bytes).await?;
            buf.to_vec()
        };

        let remote_ep = deserialize_qp_endpoint(&remote_bytes);

        let rdma_conn = match prepared.complete(remote_ep) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("RDMA handshake {my_rank}->{peer_rank} failed: {e}");
                continue;
            }
        };

        if let Ok(peer) = client.peer(peer_rank) {
            peer.set_rdma(rdma_conn, Arc::clone(&pool));
        }
    }

    tracing::info!("RDMA mesh established for rank {my_rank}");
    Ok(())
}

/// In-process RDMA mesh establishment for testing (no network exchange needed).
///
/// Takes all clients and connects them pairwise by directly exchanging endpoints
/// in memory.
pub async fn establish_rdma_mesh_local(clients: &[NexarClient]) {
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

/// Establish GPUDirect RDMA connections to all peers via QUIC endpoint exchange.
///
/// Called collectively by all ranks. Lower-ranked peer sends first.
#[cfg(feature = "gpudirect")]
pub async fn establish_gpudirect_mesh(client: &NexarClient) -> Result<()> {
    use crate::ext::PeerConnectionGpuDirectExt;
    use crate::gpudirect::{GpuDirectContext, GpuDirectEndpoint, GpuDirectPool};

    let ctx = match GpuDirectContext::new(None) {
        Ok(ctx) => Arc::new(ctx),
        Err(e) => {
            tracing::warn!("GPUDirect not available: {e}");
            return Ok(());
        }
    };

    let my_rank = client.rank();
    let world = client.world_size();

    for peer_rank in 0..world {
        if peer_rank == my_rank {
            continue;
        }

        let prepared = match ctx.prepare_qp() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("GPUDirect QP prepare for {my_rank}->{peer_rank} failed: {e}");
                continue;
            }
        };

        let local_ep = prepared.endpoint();
        let local_bytes = local_ep.to_bytes();

        let remote_bytes = if my_rank < peer_rank {
            client.send_bytes(peer_rank, &local_bytes).await?;
            let buf = client.recv_bytes(peer_rank).await?;
            buf.to_vec()
        } else {
            let buf = client.recv_bytes(peer_rank).await?;
            client.send_bytes(peer_rank, &local_bytes).await?;
            buf.to_vec()
        };

        let remote_ep =
            GpuDirectEndpoint::from_bytes(remote_bytes[..20].try_into().map_err(|_| {
                nexar::NexarError::DecodeFailed(format!(
                    "GPUDirect endpoint too short: {} bytes",
                    remote_bytes.len()
                ))
            })?);

        let qp = match prepared.complete(remote_ep) {
            Ok(qp) => qp,
            Err(e) => {
                tracing::warn!("GPUDirect handshake {my_rank}->{peer_rank} failed: {e}");
                continue;
            }
        };

        let empty_pool = Arc::new(GpuDirectPool::empty(Arc::clone(&ctx)));
        if let Ok(peer) = client.peer(peer_rank) {
            peer.set_gpudirect(qp, empty_pool);
        }
    }

    tracing::info!("GPUDirect RDMA mesh established for rank {my_rank}");
    Ok(())
}

/// In-process GPUDirect mesh establishment for testing.
#[cfg(feature = "gpudirect")]
pub fn establish_gpudirect_mesh_local(clients: &[NexarClient]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qp_endpoint_roundtrip_with_gid() {
        let raw: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        // Safety: ibverbs::Gid is a wrapper around [u8; 16].
        let gid: ibverbs::Gid = unsafe { std::mem::transmute(raw) };
        let ep = ibverbs::QueuePairEndpoint {
            num: 0x12345678,
            lid: 0xABCD,
            gid: Some(gid),
        };
        let bytes = serialize_qp_endpoint(&ep);
        let ep2 = deserialize_qp_endpoint(&bytes);
        assert_eq!(ep2.num, ep.num);
        assert_eq!(ep2.lid, ep.lid);
        let gid2 = ep2.gid.unwrap();
        assert_eq!(gid2.subnet_prefix(), ep.gid.unwrap().subnet_prefix());
        assert_eq!(gid2.interface_id(), ep.gid.unwrap().interface_id());
    }

    #[test]
    fn test_qp_endpoint_roundtrip_no_gid() {
        let ep = ibverbs::QueuePairEndpoint {
            num: 42,
            lid: 7,
            gid: None,
        };
        let bytes = serialize_qp_endpoint(&ep);
        let ep2 = deserialize_qp_endpoint(&bytes);
        assert_eq!(ep2.num, 42);
        assert_eq!(ep2.lid, 7);
        assert!(ep2.gid.is_none());
    }
}
