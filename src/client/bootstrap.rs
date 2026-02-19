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
        let seed_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
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
        let w = workers.into_iter().next().unwrap();
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
    // Each listener uses the worker's CA-signed cert and requires client certs.
    let mut listeners = Vec::new();
    let mut listen_addrs = Vec::new();
    for w in &workers {
        let cert = rustls::pki_types::CertificateDer::from(w.node_cert.clone());
        let key = rustls::pki_types::PrivateKeyDer::try_from(w.node_key.clone())
            .map_err(|e| NexarError::Tls(format!("parse node key for rank {}: {e}", w.rank)))?;
        let listener = crate::transport::TransportListener::bind_with_mtls(
            "127.0.0.1:0".parse().unwrap(),
            cert,
            key,
            &ca_cert_der,
        )?;
        listen_addrs.push(listener.local_addr());
        listeners.push(listener);
    }

    // Each worker needs connections to all other workers.
    // Worker i will accept connections from workers j > i,
    // and connect to workers j < i.
    // Build per-worker mTLS client configs.
    let mut client_configs = Vec::new();
    for w in &workers {
        let cert = rustls::pki_types::CertificateDer::from(w.node_cert.clone());
        let key = rustls::pki_types::PrivateKeyDer::try_from(w.node_key.clone())
            .map_err(|e| NexarError::Tls(format!("parse node key for rank {}: {e}", w.rank)))?;
        client_configs.push(make_client_config_mtls(cert, key, &ca_cert_der)?);
    }

    let mut all_peers: Vec<HashMap<Rank, PeerConnection>> =
        (0..n).map(|_| HashMap::new()).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let rank_i = workers[i].rank;
            let rank_j = workers[j].rank;
            let addr_j = listen_addrs[j];

            // Worker i connects to worker j using i's mTLS client config.
            let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap())
                .map_err(|e| NexarError::Transport(format!("mesh client: {e}")))?;
            endpoint.set_default_client_config(client_configs[i].clone());

            let accept_fut = listeners[j].accept();
            let connect_fut = endpoint.connect(addr_j, "localhost");

            let connect_connecting =
                connect_fut.map_err(|e| NexarError::Transport(format!("mesh connect: {e}")))?;

            let (accepted, connected) = tokio::try_join!(accept_fut, async {
                connect_connecting
                    .await
                    .map_err(|e| NexarError::Transport(format!("mesh handshake: {e}")))
            })?;

            all_peers[i].insert(rank_j, PeerConnection::new(rank_j, connected));
            all_peers[j].insert(rank_i, PeerConnection::new(rank_i, accepted));
        }
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
    Ok(clients)
}
