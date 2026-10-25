//! Mesh formation for one rank. In-process and multi-process bootstraps share it.
//!
//! 1. Bind this rank's mTLS listener and, if needed, its TCP sidecar listener.
//! 2. Report the advertised address to the seed and wait for the peer table.
//! 3. Dial higher-ranked neighbors, accept lower-ranked ones, attach sidecars.
//! 4. Enter the seed's ready barrier.

use crate::client::NexarClient;
use crate::client::bootstrap_peer::{MeshPlan, connect_links};
use crate::client::bootstrap_sidecar::Sidecar;
use crate::cluster::WorkerNode;
use crate::cluster::sparse::{TopologyStrategy, build_neighbors, build_routing_table};
use crate::config::NexarConfig;
use crate::device::DeviceAdapter;
use crate::error::{NexarError, Result};
use crate::transport::buffer_pool::PoolProfile;
use crate::transport::tls::make_client_config_mtls;
use crate::transport::{PeerConnection, TaggedBulkTransport, TransportListener};
use crate::types::Rank;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;

/// Where this rank listens and what it advertises.
pub(super) struct MeshAddrs {
    pub bind: SocketAddr,
    pub advertise: Option<SocketAddr>,
}

/// An error and whether the seed still needs to hear about it.
struct MeshError {
    error: NexarError,
    report: bool,
}

fn local(error: NexarError) -> MeshError {
    MeshError {
        error,
        report: true,
    }
}

fn from_seed(error: NexarError) -> MeshError {
    MeshError {
        error,
        report: false,
    }
}

/// Build this rank's part of the mesh and return its client.
///
/// A local error goes to the seed, which aborts the other ranks.
pub(super) async fn join_mesh(
    worker: &mut WorkerNode,
    addrs: &MeshAddrs,
    adapter: Arc<dyn DeviceAdapter>,
    config: NexarConfig,
) -> Result<NexarClient> {
    match build_client(worker, addrs, adapter, config).await {
        Ok(client) => {
            worker.ready().await?;
            Ok(client)
        }
        Err(MeshError { error, report }) => {
            if report {
                worker.report_failure(&error).await;
            }
            Err(error)
        }
    }
}

async fn build_client(
    worker: &mut WorkerNode,
    addrs: &MeshAddrs,
    adapter: Arc<dyn DeviceAdapter>,
    config: NexarConfig,
) -> std::result::Result<NexarClient, MeshError> {
    let (rank, world) = (worker.rank, worker.world_size);
    let ca = CertificateDer::from(worker.ca_cert.clone());
    let cert = CertificateDer::from(worker.node_cert.clone());
    let key = PrivateKeyDer::try_from(worker.node_key.clone()).map_err(|e| {
        local(NexarError::Tls(format!(
            "parse node key for rank {rank}: {e}"
        )))
    })?;

    let listener =
        TransportListener::bind_with_mtls(addrs.bind, cert.clone(), key.clone_key(), &ca)
            .map_err(local)?;
    let advertised = advertised_addr(addrs, listener.local_addr(), worker).map_err(local)?;

    let neighbors = mesh_neighbors(&config.topology, rank, world);
    let lower: Vec<Rank> = neighbors.iter().copied().filter(|&r| r < rank).collect();
    let higher: Vec<Rank> = neighbors.iter().copied().filter(|&r| r > rank).collect();
    let sidecar = Sidecar::prepare(
        &config,
        addrs.bind,
        advertised,
        !higher.is_empty(),
        (&cert, &key, &ca),
    )
    .await
    .map_err(local)?;

    let peers = worker
        .announce_mesh(advertised.to_string())
        .await
        .map_err(from_seed)?;
    let mut table = HashMap::new();
    for (r, addr) in peers {
        let parsed: SocketAddr = addr.parse().map_err(|e| {
            from_seed(NexarError::DecodeFailed(format!(
                "seed sent invalid address '{addr}' for rank {r}: {e}"
            )))
        })?;
        table.insert(r, parsed);
    }

    let plan = MeshPlan {
        rank,
        table: &table,
        lower: &lower,
        higher: &higher,
        quic_client: make_client_config_mtls(cert, key, &ca).map_err(local)?,
        sidecar: &sidecar,
    };
    let deadline = worker.deadline();
    let timeout_ms = worker.join_timeout().as_millis() as u64;
    let links = tokio::select! {
        links = tokio::time::timeout_at(deadline, connect_links(&plan, &listener)) => match links {
            Ok(Ok(links)) => links,
            Ok(Err(e)) => return Err(local(e)),
            Err(_) => return Err(local(NexarError::JoinTimeout { stage: "peer connections", timeout_ms })),
        },
        e = worker.next_abort() => return Err(from_seed(e)),
    };

    let mut peer_conns = HashMap::new();
    for (r, conn) in links.conns {
        peer_conns.insert(r, PeerConnection::new(r, conn));
    }
    futures::future::join_all(peer_conns.values().map(|p| p.warm_stream_pool())).await;

    let sparse = !matches!(config.topology, TopologyStrategy::FullMesh);
    let topology = config.topology.clone();
    let mut client = NexarClient::new_with_config(
        rank,
        world,
        peer_conns,
        adapter,
        PoolProfile::Training,
        config,
    );
    // The mesh is complete: refuse further inbound connections on this endpoint.
    let endpoint = listener.endpoint().clone();
    endpoint.set_server_config(None);
    client.mesh.add_endpoint(endpoint);
    if sparse {
        let table = Arc::new(build_routing_table(&topology, rank, world));
        client.setup_relay(table).await;
    }
    for (r, transport) in links.sidecars {
        client
            .peer(r)
            .and_then(|p| p.add_extension(Arc::new(transport) as Arc<dyn TaggedBulkTransport>))
            .map_err(local)?;
    }
    Ok(client)
}

/// The address peers dial: the advertise IP, else the bind IP, else the IP the
/// seed observes. The port is the advertise port, else the bound port.
fn advertised_addr(
    addrs: &MeshAddrs,
    bound: SocketAddr,
    worker: &WorkerNode,
) -> Result<SocketAddr> {
    let host = addrs
        .advertise
        .map(|a| a.ip())
        .filter(|ip| !ip.is_unspecified())
        .or(Some(addrs.bind.ip()).filter(|ip| !ip.is_unspecified()))
        .or(worker.observed_ip)
        .ok_or_else(|| NexarError::BootstrapAborted {
            reason: format!(
                "rank {} has no address to advertise: set ClusterConfig::advertise_addr",
                worker.rank
            ),
        })?;
    let port = addrs
        .advertise
        .map(|a| a.port())
        .filter(|&p| p != 0)
        .unwrap_or(bound.port());
    Ok(SocketAddr::new(host, port))
}

/// Ranks this rank links to directly. Symmetric: `j` is a neighbor of `i`
/// when either rank's topology lists the other.
pub(super) fn mesh_neighbors(
    topology: &TopologyStrategy,
    rank: Rank,
    world: u32,
) -> BTreeSet<Rank> {
    let mut set: BTreeSet<Rank> = build_neighbors(topology, rank, world)
        .into_iter()
        .filter(|&r| r != rank && r < world)
        .collect();
    if !matches!(topology, TopologyStrategy::FullMesh) {
        for other in (0..world).filter(|&r| r != rank) {
            if build_neighbors(topology, other, world).contains(&rank) {
                set.insert(other);
            }
        }
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_neighbors_full_mesh() {
        let n = mesh_neighbors(&TopologyStrategy::FullMesh, 1, 4);
        assert_eq!(n.into_iter().collect::<Vec<_>>(), vec![0, 2, 3]);
    }

    #[test]
    fn test_mesh_neighbors_symmetric() {
        let topo = TopologyStrategy::KRegular { degree: 2 };
        for i in 0..6 {
            for j in mesh_neighbors(&topo, i, 6) {
                assert!(mesh_neighbors(&topo, j, 6).contains(&i), "{i} -> {j}");
            }
        }
    }
}
