use crate::client::NexarClient;
use crate::cluster::{SeedNode, WorkerNode};
use crate::device::DeviceAdapter;
use crate::error::{NexarError, Result};
use crate::transport::PeerConnection;
use crate::transport::tls::make_client_config_mtls;
use crate::types::Rank;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

impl NexarClient {
    /// Bootstrap a cluster: start a seed node and connect workers.
    ///
    /// This is a convenience for tests and simple deployments where
    /// all nodes run in the same process (each as a tokio task).
    ///
    /// The seed generates an ephemeral CA during formation. Workers receive
    /// CA-signed certificates via the Welcome message and use them for
    /// mutual TLS on the P2P mesh.
    pub async fn bootstrap_local(
        world_size: u32,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Result<Vec<NexarClient>> {
        let seed_addr: SocketAddr = "127.0.0.1:0".parse().expect("hardcoded socket addr");
        let seed = SeedNode::bind(seed_addr, world_size)?;
        let seed_addr = seed.local_addr();

        // Spawn seed.
        let seed_handle = tokio::spawn(async move { seed.form_cluster().await });

        // Spawn workers and collect their results.
        let mut worker_handles = Vec::new();
        for _ in 0..world_size {
            worker_handles.push(tokio::spawn(WorkerNode::connect(seed_addr)));
        }

        let (_map, _seed_conns) = seed_handle
            .await
            .map_err(|e| NexarError::Transport(format!("seed task panicked: {e}")))??;

        let mut workers: Vec<WorkerNode> = Vec::new();
        for h in worker_handles {
            workers.push(
                h.await
                    .map_err(|e| NexarError::Transport(format!("worker task panicked: {e}")))??,
            );
        }

        build_mesh(workers, adapter).await
    }
}

/// Establish a full mesh of P2P connections between workers using mutual TLS.
///
/// All (i, j) pairs are connected concurrently using `futures::future::try_join_all`,
/// reducing wall-clock time from O(N²) sequential handshakes to O(N) (limited by
/// the node with the most connections).
///
/// Each worker's CA-signed certificate (received via Welcome) is used for
/// both the mesh listener and outgoing connections. All peers verify each
/// other against the cluster CA.
async fn build_mesh(
    workers: Vec<WorkerNode>,
    adapter: Arc<dyn DeviceAdapter>,
) -> Result<Vec<NexarClient>> {
    let n = workers.len();
    if n == 1 {
        // Single-node: no peers needed.
        let w = workers
            .into_iter()
            .next()
            .expect("workers vec confirmed non-empty by n==1 check");
        return Ok(vec![NexarClient::new(
            w.rank,
            w.world_size,
            HashMap::new(),
            adapter,
        )]);
    }

    // All workers share the same CA cert (from the same cluster formation).
    let ca_cert_der = rustls::pki_types::CertificateDer::from(workers[0].ca_cert.clone());

    // Bind an mTLS listener for each worker on a random port.
    let mut listeners = Vec::new();
    let mut listen_addrs = Vec::new();
    for w in &workers {
        let cert = rustls::pki_types::CertificateDer::from(w.node_cert.clone());
        let key = rustls::pki_types::PrivateKeyDer::try_from(w.node_key.clone())
            .map_err(|e| NexarError::Tls(format!("parse node key for rank {}: {e}", w.rank)))?;
        let listener = crate::transport::TransportListener::bind_with_mtls(
            "127.0.0.1:0".parse().expect("hardcoded socket addr"),
            cert,
            key,
            &ca_cert_der,
        )?;
        listen_addrs.push(listener.local_addr());
        listeners.push(listener);
    }

    // Build per-worker mTLS client configs.
    let mut client_configs = Vec::new();
    for w in &workers {
        let cert = rustls::pki_types::CertificateDer::from(w.node_cert.clone());
        let key = rustls::pki_types::PrivateKeyDer::try_from(w.node_key.clone())
            .map_err(|e| NexarError::Tls(format!("parse node key for rank {}: {e}", w.rank)))?;
        client_configs.push(make_client_config_mtls(cert, key, &ca_cert_der)?);
    }

    // Wrap listeners in Arc so they can be shared across concurrent tasks.
    let listeners: Vec<Arc<crate::transport::TransportListener>> =
        listeners.into_iter().map(Arc::new).collect();

    // Connect all (i, j) pairs concurrently.
    let mut pair_futures = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let rank_i = workers[i].rank;
            let rank_j = workers[j].rank;
            let addr_j = listen_addrs[j];
            let config_i = client_configs[i].clone();
            let listener_j = Arc::clone(&listeners[j]);

            pair_futures.push(tokio::spawn(async move {
                let mut endpoint =
                    quinn::Endpoint::client("0.0.0.0:0".parse().expect("hardcoded socket addr"))
                        .map_err(|e| NexarError::Transport(format!("mesh client: {e}")))?;
                endpoint.set_default_client_config(config_i);

                let accept_fut = listener_j.accept();
                let connect_fut = endpoint.connect(addr_j, "localhost");

                let connect_connecting =
                    connect_fut.map_err(|e| NexarError::Transport(format!("mesh connect: {e}")))?;

                let (accepted, connected) = tokio::try_join!(accept_fut, async {
                    connect_connecting
                        .await
                        .map_err(|e| NexarError::Transport(format!("mesh handshake: {e}")))
                })?;

                Ok::<_, NexarError>((
                    i,
                    j,
                    rank_i,
                    rank_j,
                    PeerConnection::new(rank_j, connected),
                    PeerConnection::new(rank_i, accepted),
                ))
            }));
        }
    }

    let mut all_peers: Vec<HashMap<Rank, PeerConnection>> =
        (0..n).map(|_| HashMap::new()).collect();

    for handle in pair_futures {
        let (i, j, rank_i, rank_j, conn_ij, conn_ji) = handle
            .await
            .map_err(|e| NexarError::Transport(format!("mesh task panicked: {e}")))??;
        all_peers[i].insert(rank_j, conn_ij);
        all_peers[j].insert(rank_i, conn_ji);
    }

    let mut clients = Vec::new();
    for (idx, peers) in all_peers.into_iter().enumerate() {
        clients.push(NexarClient::new(
            workers[idx].rank,
            workers[idx].world_size,
            peers,
            Arc::clone(&adapter),
        ));
    }

    // Sort by rank.
    clients.sort_by_key(|c| c.rank());

    // Optionally establish RDMA connections over the QUIC mesh.
    #[cfg(feature = "rdma")]
    establish_rdma_mesh(&mut clients).await;

    // Optionally establish GPUDirect RDMA connections.
    #[cfg(feature = "gpudirect")]
    establish_gpudirect_mesh(&mut clients);

    Ok(clients)
}

/// Attempt to establish RDMA connections between all pairs.
///
/// Uses the existing QUIC mesh to exchange `QueuePairEndpoint` data.
/// If RDMA device initialization fails (no IB hardware), logs a warning
/// and continues with QUIC-only transport.
///
/// RDMA connections are attached to the underlying `PeerConnection`s for
/// bulk data offload. Control messages always remain on QUIC.
#[cfg(feature = "rdma")]
async fn establish_rdma_mesh(clients: &mut [NexarClient]) {
    use crate::transport::rdma::{RdmaContext, RdmaMemoryPool};

    // Try to open the RDMA device. If it fails, RDMA is simply not available.
    let ctx: std::sync::Arc<RdmaContext> = match RdmaContext::new(None) {
        Ok(ctx) => std::sync::Arc::new(ctx),
        Err(e) => {
            tracing::warn!("RDMA not available, using QUIC-only transport: {e}");
            return;
        }
    };

    // Create a shared memory pool (16 buffers of 4 MiB each).
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

    // For each pair (i, j), prepare RDMA connections and exchange endpoints
    // via QUIC control messages. This is done sequentially since RDMA
    // handshake is per-QP and relatively fast.
    //
    // In a real multi-node deployment, endpoints would be exchanged via
    // the existing QUIC control channel between peers. For local bootstrap,
    // we can do it directly since we have access to all clients.
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

            // Attach RDMA connections to the PeerConnections.
            // We need mutable access to the inner PeerConnection, which is behind an Arc.
            // Since we're in bootstrap (single-threaded at this point), we use Arc::get_mut.
            if let Some(peer) = clients[i]
                .peers
                .get_mut(&rank_j)
                .and_then(|p| std::sync::Arc::get_mut(p))
            {
                peer.set_rdma(rdma_i, std::sync::Arc::clone(&pool));
            }
            if let Some(peer) = clients[j]
                .peers
                .get_mut(&rank_i)
                .and_then(|p| std::sync::Arc::get_mut(p))
            {
                peer.set_rdma(rdma_j, std::sync::Arc::clone(&pool));
            }
        }
    }

    tracing::info!("RDMA mesh established for {} nodes", n);
}

/// Attempt to establish GPUDirect RDMA connections between all pairs.
///
/// Creates a `GpuDirectContext` (raw FFI to ibverbs-sys), prepares QPs for
/// each peer pair, exchanges endpoints directly (local bootstrap), and
/// attaches the completed QPs to PeerConnections.
///
/// If GPUDirect initialization fails (no IB hardware or no nvidia-peermem),
/// logs a warning and continues without GPUDirect.
#[cfg(feature = "gpudirect")]
fn establish_gpudirect_mesh(clients: &mut [NexarClient]) {
    use crate::transport::rdma::GpuDirectContext;

    let n = clients.len();
    if n < 2 {
        return;
    }

    let ctx = match GpuDirectContext::new(None) {
        Ok(ctx) => std::sync::Arc::new(ctx),
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

            // We don't create a GpuDirectPool here because that requires
            // pre-allocated GPU memory. The pool will be set up by the caller
            // when they allocate GPU buffers for collective operations.
            // For now, attach the QPs without a pool — send_raw_gpu will
            // fall back to staged transfers until a pool is provided.

            // Attach QPs to PeerConnections.
            let empty_pool_i = std::sync::Arc::new(crate::transport::rdma::GpuDirectPool::empty(
                std::sync::Arc::clone(&ctx),
            ));
            let empty_pool_j = std::sync::Arc::new(crate::transport::rdma::GpuDirectPool::empty(
                std::sync::Arc::clone(&ctx),
            ));

            if let Some(peer) = clients[i]
                .peers
                .get_mut(&rank_j)
                .and_then(|p| std::sync::Arc::get_mut(p))
            {
                peer.set_gpudirect(qp_i, empty_pool_i);
            }
            if let Some(peer) = clients[j]
                .peers
                .get_mut(&rank_i)
                .and_then(|p| std::sync::Arc::get_mut(p))
            {
                peer.set_gpudirect(qp_j, empty_pool_j);
            }
        }
    }

    tracing::info!("GPUDirect RDMA mesh established for {n} nodes");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuAdapter;

    #[tokio::test]
    async fn test_bootstrap_single_node() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = NexarClient::bootstrap_local(1, adapter).await.unwrap();
        assert_eq!(clients.len(), 1);
        assert_eq!(clients[0].rank(), 0);
        assert_eq!(clients[0].world_size(), 1);
    }

    #[tokio::test]
    async fn test_bootstrap_two_nodes() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = NexarClient::bootstrap_local(2, adapter).await.unwrap();
        assert_eq!(clients.len(), 2);
        assert_eq!(clients[0].rank(), 0);
        assert_eq!(clients[1].rank(), 1);
        assert_eq!(clients[0].world_size(), 2);
    }

    #[tokio::test]
    async fn test_bootstrap_four_nodes() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = NexarClient::bootstrap_local(4, adapter).await.unwrap();
        assert_eq!(clients.len(), 4);
        for (i, c) in clients.iter().enumerate() {
            assert_eq!(c.rank() as usize, i);
            assert_eq!(c.world_size(), 4);
        }
    }

    #[tokio::test]
    async fn test_send_recv_two_nodes() {
        let adapter = Arc::new(CpuAdapter::new());
        let clients = NexarClient::bootstrap_local(2, adapter).await.unwrap();

        // We need to run send and recv concurrently.
        let send_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut recv_buf: Vec<f32> = vec![0.0; 4];
        let size = send_data.len() * std::mem::size_of::<f32>();

        // Move clients into Arcs for sharing across tasks.
        let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();
        let c0 = Arc::clone(&clients[0]);
        let c1 = Arc::clone(&clients[1]);

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_buf.as_mut_ptr() as u64;

        let send_task =
            tokio::spawn(async move { unsafe { c0.send(send_ptr, size, 1, 42).await } });
        let recv_task =
            tokio::spawn(async move { unsafe { c1.recv(recv_ptr, size, 0, 42).await } });

        send_task.await.unwrap().unwrap();
        recv_task.await.unwrap().unwrap();

        assert_eq!(recv_buf, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
