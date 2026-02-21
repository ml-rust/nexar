use crate::client::NexarClient;
use crate::client::bootstrap_mesh::{
    establish_connections, establish_tcp_sidecars, prepare_tls_infra,
};
use crate::cluster::PendingJoin;
use crate::cluster::elastic::{ElasticBootstrap, ElasticConfig, ElasticManager};
use crate::cluster::sparse::{TopologyStrategy, build_neighbors, build_routing_table};
use crate::cluster::{SeedNode, WorkerNode};
use crate::config::NexarConfig;
use crate::device::DeviceAdapter;
use crate::error::{NexarError, Result};
use crate::types::Rank;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

impl NexarClient {
    /// Bootstrap a cluster: start a seed node and connect workers.
    ///
    /// This is a convenience for tests and simple deployments where
    /// all nodes run in the same process (each as a tokio task).
    pub async fn bootstrap_local(
        world_size: u32,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Result<Vec<NexarClient>> {
        let workers = spawn_seed_and_workers(world_size).await?;
        build_mesh(workers, adapter).await
    }

    /// Like [`bootstrap_local`], but with a custom configuration.
    pub async fn bootstrap_local_with_config(
        world_size: u32,
        adapter: Arc<dyn DeviceAdapter>,
        config: crate::config::NexarConfig,
    ) -> Result<Vec<NexarClient>> {
        let workers = spawn_seed_and_workers(world_size).await?;
        build_mesh_with_config(workers, adapter, config).await
    }

    /// Bootstrap a cluster with elastic scaling support.
    pub async fn bootstrap_elastic(
        initial_world: u32,
        elastic_config: ElasticConfig,
        nexar_config: NexarConfig,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Result<ElasticBootstrap> {
        let seed_addr: SocketAddr = "127.0.0.1:0".parse().expect("hardcoded socket addr");
        let seed = SeedNode::bind_local(seed_addr, initial_world)?;
        let seed_addr = seed.local_addr();

        let seed = Arc::new(seed);
        let seed_for_formation = Arc::clone(&seed);
        let seed_handle = tokio::spawn(async move { seed_for_formation.form_cluster().await });

        let mut worker_handles = Vec::new();
        for _ in 0..initial_world {
            worker_handles.push(tokio::spawn(WorkerNode::connect(seed_addr)));
        }

        let seed_result = seed_handle
            .await
            .map_err(|e| NexarError::transport_with_source("seed task panicked", e))??;

        let mut workers: Vec<WorkerNode> = Vec::new();
        for h in worker_handles {
            workers.push(
                h.await
                    .map_err(|e| NexarError::transport_with_source("worker task panicked", e))??,
            );
        }

        // Save credentials before building mesh.
        let creds: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = workers
            .iter()
            .map(|w| (w.ca_cert.clone(), w.node_cert.clone(), w.node_key.clone()))
            .collect();

        let clients = build_mesh_with_config(workers, adapter, nexar_config.clone()).await?;

        let pending_joins = Arc::new(std::sync::Mutex::new(Vec::<PendingJoin>::new()));

        let ca = Arc::new(seed_result.ca);
        let next_rank = Arc::new(std::sync::atomic::AtomicU32::new(seed_result.next_rank));
        let cluster_map = Arc::new(std::sync::Mutex::new(seed_result.map));
        let max_world = elastic_config.max_world_size;

        let pj = Arc::clone(&pending_joins);
        let ca2 = Arc::clone(&ca);
        let nr = Arc::clone(&next_rank);
        let cm = Arc::clone(&cluster_map);
        tokio::spawn(async move {
            let _ = seed.accept_elastic(ca2, nr, max_world, pj, cm).await;
        });

        let mut managers = Vec::new();
        for (client, (ca_cert, node_cert, node_key)) in clients.into_iter().zip(creds) {
            managers.push(ElasticManager::new(
                client,
                elastic_config.clone(),
                nexar_config.clone(),
                ca_cert,
                node_cert,
                node_key,
                Arc::clone(&pending_joins),
                Some(seed_addr),
            ));
        }

        Ok(ElasticBootstrap {
            managers,
            seed_addr,
        })
    }
}

/// Spawn seed + workers and collect the worker nodes.
async fn spawn_seed_and_workers(world_size: u32) -> Result<Vec<WorkerNode>> {
    let seed_addr: SocketAddr = "127.0.0.1:0".parse().expect("hardcoded socket addr");
    let seed = SeedNode::bind_local(seed_addr, world_size)?;
    let seed_addr = seed.local_addr();

    let seed_handle = tokio::spawn(async move { seed.form_cluster().await });

    let mut worker_handles = Vec::new();
    for _ in 0..world_size {
        worker_handles.push(tokio::spawn(WorkerNode::connect(seed_addr)));
    }

    let _seed_result = seed_handle
        .await
        .map_err(|e| NexarError::transport_with_source("seed task panicked", e))??;

    let mut workers: Vec<WorkerNode> = Vec::new();
    for h in worker_handles {
        workers.push(
            h.await
                .map_err(|e| NexarError::transport_with_source("worker task panicked", e))??,
        );
    }

    Ok(workers)
}

/// Establish a full mesh of P2P connections between workers using mutual TLS.
async fn build_mesh(
    workers: Vec<WorkerNode>,
    adapter: Arc<dyn DeviceAdapter>,
) -> Result<Vec<NexarClient>> {
    let n = workers.len();
    if n == 1 {
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

    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    let (listeners, listen_addrs, client_configs) = prepare_tls_infra(&workers)?;
    let (all_peers, all_endpoints) =
        establish_connections(&workers, &pairs, &listen_addrs, &client_configs, &listeners).await?;

    let mut clients = Vec::new();
    for (idx, (peers, endpoints)) in all_peers.into_iter().zip(all_endpoints).enumerate() {
        let mut client = NexarClient::new(
            workers[idx].rank,
            workers[idx].world_size,
            peers,
            Arc::clone(&adapter),
        );
        client._endpoints = endpoints;
        clients.push(client);
    }

    clients.sort_by_key(|c| c.rank());
    establish_tcp_sidecars(&clients).await?;
    Ok(clients)
}

/// Like [`build_mesh`], but with a custom configuration for each client.
async fn build_mesh_with_config(
    workers: Vec<WorkerNode>,
    adapter: Arc<dyn DeviceAdapter>,
    config: crate::config::NexarConfig,
) -> Result<Vec<NexarClient>> {
    let is_sparse = !matches!(config.topology, TopologyStrategy::FullMesh);

    if is_sparse {
        return build_sparse_mesh_with_config(workers, adapter, config).await;
    }

    let n = workers.len();
    if n == 1 {
        let w = workers
            .into_iter()
            .next()
            .expect("workers vec confirmed non-empty by n==1 check");
        return Ok(vec![NexarClient::new_with_config(
            w.rank,
            w.world_size,
            HashMap::new(),
            adapter,
            crate::transport::buffer_pool::PoolProfile::Training,
            config,
        )]);
    }

    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    let (listeners, listen_addrs, client_configs) = prepare_tls_infra(&workers)?;
    let (all_peers, all_endpoints) =
        establish_connections(&workers, &pairs, &listen_addrs, &client_configs, &listeners).await?;

    let mut clients = Vec::new();
    for (idx, (peers, endpoints)) in all_peers.into_iter().zip(all_endpoints).enumerate() {
        let mut client = NexarClient::new_with_config(
            workers[idx].rank,
            workers[idx].world_size,
            peers,
            Arc::clone(&adapter),
            crate::transport::buffer_pool::PoolProfile::Training,
            config.clone(),
        );
        client._endpoints = endpoints;
        clients.push(client);
    }

    clients.sort_by_key(|c| c.rank());
    establish_tcp_sidecars(&clients).await?;
    Ok(clients)
}

/// Build a sparse mesh where each node only connects to its neighbors.
async fn build_sparse_mesh_with_config(
    workers: Vec<WorkerNode>,
    adapter: Arc<dyn DeviceAdapter>,
    config: crate::config::NexarConfig,
) -> Result<Vec<NexarClient>> {
    let n = workers.len();
    let world_size = n as u32;

    if n == 1 {
        let w = workers
            .into_iter()
            .next()
            .expect("workers vec confirmed non-empty by n==1 check");
        return Ok(vec![NexarClient::new_with_config(
            w.rank,
            w.world_size,
            HashMap::new(),
            adapter,
            crate::transport::buffer_pool::PoolProfile::Training,
            config,
        )]);
    }

    // Compute neighbor sets for all ranks.
    let neighbor_sets: Vec<HashSet<Rank>> = workers
        .iter()
        .map(|w| build_neighbors(&config.topology, w.rank, world_size))
        .collect();

    // Determine which (i, j) pairs need connections (only neighbors).
    let mut connection_pairs_set: HashSet<(usize, usize)> = HashSet::new();
    for (i, neighbors) in neighbor_sets.iter().enumerate().take(n) {
        for &neighbor_rank in neighbors {
            let j = neighbor_rank as usize;
            if i < j {
                connection_pairs_set.insert((i, j));
            } else if j < i {
                connection_pairs_set.insert((j, i));
            }
        }
    }
    let pairs: Vec<(usize, usize)> = connection_pairs_set.into_iter().collect();

    let (listeners, listen_addrs, client_configs) = prepare_tls_infra(&workers)?;
    let (all_peers, all_endpoints) =
        establish_connections(&workers, &pairs, &listen_addrs, &client_configs, &listeners).await?;

    let mut clients = Vec::new();
    for (idx, (peers, endpoints)) in all_peers.into_iter().zip(all_endpoints).enumerate() {
        let rank = workers[idx].rank;
        let mut client = NexarClient::new_with_config(
            rank,
            workers[idx].world_size,
            peers,
            Arc::clone(&adapter),
            crate::transport::buffer_pool::PoolProfile::Training,
            config.clone(),
        );
        client._endpoints = endpoints;

        let rt = Arc::new(build_routing_table(&config.topology, rank, world_size));
        client.setup_relay(rt).await;

        clients.push(client);
    }

    clients.sort_by_key(|c| c.rank());
    establish_tcp_sidecars(&clients).await?;
    Ok(clients)
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

        let send_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut recv_buf: Vec<f32> = vec![0.0; 4];
        let size = send_data.len() * std::mem::size_of::<f32>();

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
