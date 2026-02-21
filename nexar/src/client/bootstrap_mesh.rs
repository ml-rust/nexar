//! Mesh establishment helpers for bootstrap: TLS setup, two-phase connection,
//! and TCP sidecar attachment.

use crate::client::NexarClient;
use crate::cluster::WorkerNode;
use crate::error::{NexarError, Result};
use crate::transport::PeerConnection;
use crate::transport::tls::make_client_config_mtls;
use crate::types::Rank;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

/// Prepare TLS infrastructure: listeners, listen addresses, and client configs.
pub(super) fn prepare_tls_infra(
    workers: &[WorkerNode],
) -> Result<(
    Vec<crate::transport::TransportListener>,
    Vec<SocketAddr>,
    Vec<quinn::ClientConfig>,
)> {
    let ca_cert_der = rustls::pki_types::CertificateDer::from(workers[0].ca_cert.clone());

    let mut listeners = Vec::new();
    let mut listen_addrs = Vec::new();
    for w in workers {
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

    let mut client_configs = Vec::new();
    for w in workers {
        let cert = rustls::pki_types::CertificateDer::from(w.node_cert.clone());
        let key = rustls::pki_types::PrivateKeyDer::try_from(w.node_key.clone())
            .map_err(|e| NexarError::Tls(format!("parse node key for rank {}: {e}", w.rank)))?;
        client_configs.push(make_client_config_mtls(cert, key, &ca_cert_der)?);
    }

    Ok((listeners, listen_addrs, client_configs))
}

/// Two-phase connection establishment that avoids the connection swapping bug.
///
/// The old approach had each (i,j) pair task call `accept()` on listener j concurrently.
/// When multiple pairs share the same listener j, their `accept()` calls race and can
/// receive connections from the wrong peer, causing data to route to wrong PeerRouters.
///
/// Fix: Phase 1 spawns all outgoing connects. Phase 2 runs one accept loop per listener
/// (concurrently across listeners, sequentially within each). Phase 3 matches accepted
/// connections to connectors by comparing remote_addr == connector local_addr.
pub(super) async fn establish_connections(
    workers: &[WorkerNode],
    pairs: &[(usize, usize)],
    listen_addrs: &[SocketAddr],
    client_configs: &[quinn::ClientConfig],
    listeners: &[crate::transport::TransportListener],
) -> Result<(
    Vec<HashMap<Rank, PeerConnection>>,
    Vec<Vec<quinn::Endpoint>>,
)> {
    let n = workers.len();

    // Count expected incoming connections per listener.
    let mut accept_counts: Vec<usize> = vec![0; n];
    for &(_i, j) in pairs {
        accept_counts[j] += 1;
    }

    // Phase 1: Spawn all outgoing connects concurrently.
    let mut connect_handles = Vec::new();
    for &(i, j) in pairs {
        let addr_j = listen_addrs[j];
        let config_i = client_configs[i].clone();

        connect_handles.push(tokio::spawn(async move {
            let mut endpoint =
                quinn::Endpoint::client("127.0.0.1:0".parse().expect("hardcoded socket addr"))
                    .map_err(|e| NexarError::transport_with_source("mesh client", e))?;
            endpoint.set_default_client_config(config_i);

            let local_addr = endpoint
                .local_addr()
                .map_err(|e| NexarError::transport_with_source("endpoint local_addr", e))?;

            let connecting = endpoint
                .connect(addr_j, "localhost")
                .map_err(|e| NexarError::transport_with_source("mesh connect", e))?;

            let conn = connecting
                .await
                .map_err(|e| NexarError::transport_with_source("mesh handshake", e))?;

            Ok::<_, NexarError>((i, j, conn, endpoint, local_addr))
        }));
    }

    // Phase 2: Accept all expected connections per listener.
    let connect_fut = async {
        let mut results = Vec::new();
        for h in connect_handles {
            results
                .push(h.await.map_err(|e| {
                    NexarError::transport_with_source("connect task panicked", e)
                })??);
        }
        Ok::<_, NexarError>(results)
    };

    let accept_fut = async {
        let mut futs = Vec::new();
        for j in 0..n {
            let count = accept_counts[j];
            if count == 0 {
                continue;
            }
            let listener = &listeners[j];
            futs.push(async move {
                let mut accepted = Vec::new();
                for _ in 0..count {
                    let conn = listener.accept().await?;
                    let remote = conn.remote_address();
                    accepted.push((j, conn, remote));
                }
                Ok::<_, NexarError>(accepted)
            });
        }
        let results = futures::future::try_join_all(futs).await?;
        Ok::<_, NexarError>(results.into_iter().flatten().collect::<Vec<_>>())
    };

    // Run connects and accepts concurrently.
    let (connect_results, accepted_conns) = tokio::try_join!(connect_fut, accept_fut)?;

    // Phase 3: Match accepted connections to connectors by address.
    let mut accept_map: HashMap<(usize, SocketAddr), quinn::Connection> = HashMap::new();
    for (j, conn, remote) in accepted_conns {
        accept_map.insert((j, remote), conn);
    }

    let mut all_peers: Vec<HashMap<Rank, PeerConnection>> =
        (0..n).map(|_| HashMap::new()).collect();
    let mut all_endpoints: Vec<Vec<quinn::Endpoint>> = (0..n).map(|_| Vec::new()).collect();

    for (i, j, connected, endpoint, local_addr) in connect_results {
        let rank_i = workers[i].rank;
        let rank_j = workers[j].rank;

        let accepted = accept_map.remove(&(j, local_addr)).ok_or_else(|| {
            NexarError::transport(format!(
                "no accepted connection on listener {j} from {local_addr} \
                     (pair ({i},{j}), ranks ({rank_i},{rank_j}))"
            ))
        })?;

        let conn_ij = PeerConnection::new(rank_j, connected);
        let conn_ji = PeerConnection::new(rank_i, accepted);

        tokio::join!(conn_ij.warm_stream_pool(), conn_ji.warm_stream_pool());

        all_peers[i].insert(rank_j, conn_ij);
        all_peers[j].insert(rank_i, conn_ji);
        all_endpoints[i].push(endpoint);
    }

    Ok((all_peers, all_endpoints))
}

/// Establish TCP bulk sidecar connections between all peer pairs.
///
/// For each (i, j) pair with i < j: rank i listens, rank j connects.
/// Respects `NexarConfig::enable_tcp_bulk_sidecar` — skips entirely if disabled.
pub(super) async fn establish_tcp_sidecars(clients: &[NexarClient]) -> Result<()> {
    use crate::transport::TaggedBulkTransport;
    use crate::transport::tcp_bulk::{
        tcp_bulk_accept, tcp_bulk_accept_tls, tcp_bulk_connect, tcp_bulk_connect_tls,
        tcp_bulk_listen,
    };
    use crate::transport::tls::{
        make_bulk_tls_client_config_insecure, make_bulk_tls_server_config_insecure,
    };

    let n = clients.len();
    if n <= 1 {
        return Ok(());
    }

    let config = clients[0].config();
    if !config.enable_tcp_bulk_sidecar {
        tracing::info!("TCP bulk sidecar disabled by config");
        return Ok(());
    }

    let encrypt = config.encrypt_bulk_transport;
    if !encrypt {
        tracing::warn!(
            "TCP bulk sidecar is UNENCRYPTED. Tensor data will be sent in plaintext. \
             Set NEXAR_ENCRYPT_BULK_TRANSPORT=true or config.encrypt_bulk_transport=true \
             to enable TLS on the bulk transport."
        );
    }

    let tls_server = if encrypt {
        Some(make_bulk_tls_server_config_insecure()?)
    } else {
        None
    };
    let tls_client = if encrypt {
        Some(make_bulk_tls_client_config_insecure()?)
    } else {
        None
    };

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
                if let (Some(sc), Some(cc)) = (&tls_server, &tls_client) {
                    tokio::try_join!(
                        tcp_bulk_accept_tls(&listener, Arc::clone(sc)),
                        tcp_bulk_connect_tls(addr, Arc::clone(cc)),
                    )?
                } else {
                    tokio::try_join!(tcp_bulk_accept(&listener), tcp_bulk_connect(addr),)?
                };

            let peer_ij = clients[i].peer(rank_j)?;
            peer_ij.add_extension(Arc::new(transport_i) as Arc<dyn TaggedBulkTransport>)?;

            let peer_ji = clients[j].peer(rank_i)?;
            peer_ji.add_extension(Arc::new(transport_j) as Arc<dyn TaggedBulkTransport>)?;
        }
    }

    Ok(())
}
