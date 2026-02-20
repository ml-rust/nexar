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
        let seed = SeedNode::bind_local(seed_addr, world_size)?;
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
            .map_err(|e| NexarError::transport_with_source("seed task panicked", e))??;

        let mut workers: Vec<WorkerNode> = Vec::new();
        for h in worker_handles {
            workers.push(
                h.await
                    .map_err(|e| NexarError::transport_with_source("worker task panicked", e))??,
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
                        .map_err(|e| NexarError::transport_with_source("mesh client", e))?;
                endpoint.set_default_client_config(config_i);

                let accept_fut = listener_j.accept();
                let connect_fut = endpoint.connect(addr_j, "localhost");

                let connect_connecting = connect_fut
                    .map_err(|e| NexarError::transport_with_source("mesh connect", e))?;

                let (accepted, connected) = tokio::try_join!(accept_fut, async {
                    connect_connecting
                        .await
                        .map_err(|e| NexarError::transport_with_source("mesh handshake", e))
                })?;

                let conn_ij = PeerConnection::new(rank_j, connected);
                let conn_ji = PeerConnection::new(rank_i, accepted);

                // Pre-open streams to reduce first-send latency.
                tokio::join!(conn_ij.warm_stream_pool(), conn_ji.warm_stream_pool());

                Ok::<_, NexarError>((i, j, rank_i, rank_j, conn_ij, conn_ji))
            }));
        }
    }

    let mut all_peers: Vec<HashMap<Rank, PeerConnection>> =
        (0..n).map(|_| HashMap::new()).collect();

    for handle in pair_futures {
        let (i, j, rank_i, rank_j, conn_ij, conn_ji) = handle
            .await
            .map_err(|e| NexarError::transport_with_source("mesh task panicked", e))??;
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

    // Establish TCP bulk sidecar connections between each peer pair.
    establish_tcp_sidecars(&clients).await?;

    Ok(clients)
}

/// Establish TCP bulk sidecar connections between all peer pairs.
///
/// For each (i, j) pair with i < j: rank i listens, rank j connects.
/// The resulting `TcpBulkTransport` is attached as a `TaggedBulkTransport`
/// extension so `send_raw_tagged_best_effort` / `recv_bytes_tagged_best_effort`
/// automatically use the fast path.
///
/// Respects `NexarConfig::enable_tcp_bulk_sidecar` — skips entirely if disabled.
/// Emits a warning when establishing unencrypted sidecars on non-loopback addresses.
async fn establish_tcp_sidecars(clients: &[NexarClient]) -> Result<()> {
    use crate::transport::TaggedBulkTransport;
    use crate::transport::tcp_bulk::{tcp_bulk_accept, tcp_bulk_connect, tcp_bulk_listen};

    let n = clients.len();
    if n <= 1 {
        return Ok(());
    }

    // Check config from the first client (all share the same config in a cluster).
    let config = clients[0].config();
    if !config.enable_tcp_bulk_sidecar {
        tracing::info!("TCP bulk sidecar disabled by config");
        return Ok(());
    }

    if !config.encrypt_bulk_transport {
        tracing::warn!(
            "TCP bulk sidecar is UNENCRYPTED. Tensor data will be sent in plaintext. \
             Set NEXAR_ENCRYPT_BULK_TRANSPORT=true or config.encrypt_bulk_transport=true \
             to enable TLS on the bulk transport."
        );
    }

    // For each pair (i, j) with i < j: i listens, j connects.
    for i in 0..n {
        for j in (i + 1)..n {
            let rank_j = clients[j].rank();
            let rank_i = clients[i].rank();

            let bind_addr = std::net::SocketAddr::V4(std::net::SocketAddrV4::new(
                std::net::Ipv4Addr::LOCALHOST,
                0,
            ));
            let (listener, addr) = tcp_bulk_listen(bind_addr).await?;

            let (transport_i, transport_j) =
                tokio::try_join!(tcp_bulk_accept(&listener), tcp_bulk_connect(addr),)?;

            // Attach as TaggedBulkTransport to the peer connections.
            let peer_ij = clients[i].peer(rank_j)?;
            peer_ij.add_extension(Arc::new(transport_i) as Arc<dyn TaggedBulkTransport>)?;

            let peer_ji = clients[j].peer(rank_i)?;
            peer_ji.add_extension(Arc::new(transport_j) as Arc<dyn TaggedBulkTransport>)?;
        }
    }

    Ok(())
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
