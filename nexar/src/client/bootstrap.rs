use crate::client::NexarClient;
use crate::client::bootstrap_cluster::{Credentials, join_cluster};
use crate::cluster::elastic::{ElasticBootstrap, ElasticConfig, ElasticManager};
use crate::cluster::{ClusterConfig, FormClusterResult, PendingJoin, SeedNode};
use crate::config::NexarConfig;
use crate::device::DeviceAdapter;
use crate::error::{NexarError, Result};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

impl NexarClient {
    /// Bootstrap a cluster: start a seed node and connect workers.
    ///
    /// All ranks run in this process, each as a tokio task, on loopback. Ranks
    /// form the mesh through the same per-rank path as
    /// [`NexarClient::bootstrap_join`]. Uses [`NexarConfig::from_env`].
    pub async fn bootstrap_local(
        world_size: u32,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Result<Vec<NexarClient>> {
        Self::bootstrap_local_with_config(world_size, adapter, NexarConfig::from_env()).await
    }

    /// Like [`NexarClient::bootstrap_local`], but with a custom configuration.
    pub async fn bootstrap_local_with_config(
        world_size: u32,
        adapter: Arc<dyn DeviceAdapter>,
        config: NexarConfig,
    ) -> Result<Vec<NexarClient>> {
        let local = form_local(world_size, adapter, config).await?;
        Ok(local.members.into_iter().map(|(c, _)| c).collect())
    }

    /// Bootstrap a cluster with elastic scaling support.
    pub async fn bootstrap_elastic(
        initial_world: u32,
        elastic_config: ElasticConfig,
        nexar_config: NexarConfig,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Result<ElasticBootstrap> {
        let local = form_local(initial_world, adapter, nexar_config.clone()).await?;
        let seed_addr = local.seed.local_addr();

        let pending_joins = Arc::new(std::sync::Mutex::new(Vec::<PendingJoin>::new()));
        let ca = Arc::new(local.formed.ca);
        let next_rank = Arc::new(std::sync::atomic::AtomicU32::new(local.formed.next_rank));
        let cluster_map = Arc::new(std::sync::Mutex::new(local.formed.map));
        let max_world = elastic_config.max_world_size;

        let pj = Arc::clone(&pending_joins);
        let seed = local.seed;
        tokio::spawn(async move {
            let _ = seed
                .accept_elastic(ca, next_rank, max_world, pj, cluster_map)
                .await;
        });

        let managers = local
            .members
            .into_iter()
            .map(|(client, creds)| {
                ElasticManager::new(
                    client,
                    elastic_config.clone(),
                    nexar_config.clone(),
                    creds.ca_cert,
                    creds.node_cert,
                    creds.node_key,
                    Arc::clone(&pending_joins),
                    Some(seed_addr),
                )
            })
            .collect();

        Ok(ElasticBootstrap {
            managers,
            seed_addr,
        })
    }
}

/// An in-process cluster: the seed, its formation result, and every rank in rank order.
struct LocalCluster {
    seed: Arc<SeedNode>,
    formed: FormClusterResult,
    members: Vec<(NexarClient, Credentials)>,
}

/// Run a loopback seed and `world_size` ranks as tasks of this runtime.
async fn form_local(
    world_size: u32,
    adapter: Arc<dyn DeviceAdapter>,
    config: NexarConfig,
) -> Result<LocalCluster> {
    let loopback = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0);
    let seed = Arc::new(SeedNode::bind_local(loopback, world_size)?);
    let seed_addr = seed.local_addr();
    let forming = Arc::clone(&seed);
    let seed_task = tokio::spawn(async move { forming.form_cluster().await });

    let cfg = ClusterConfig::new(world_size)
        .with_bind_addr(loopback)
        .with_nexar(config);
    let rank_tasks: Vec<_> = (0..world_size)
        .map(|_| {
            let cfg = cfg.clone();
            let adapter = Arc::clone(&adapter);
            tokio::spawn(async move { join_cluster(seed_addr, &cfg, None, adapter).await })
        })
        .collect();

    let formed = seed_task
        .await
        .map_err(|e| NexarError::transport_with_source("seed task panicked", e))??;
    let mut members = Vec::with_capacity(rank_tasks.len());
    for task in rank_tasks {
        members.push(
            task.await
                .map_err(|e| NexarError::transport_with_source("rank task panicked", e))??,
        );
    }
    members.sort_by_key(|(c, _)| c.rank());
    Ok(LocalCluster {
        seed,
        formed,
        members,
    })
}

/// Two ranks on loopback, QUIC only, with `depth` messages per receive key.
#[cfg(test)]
pub(crate) async fn local_pair(depth: usize) -> (NexarClient, NexarClient) {
    local_pair_with(NexarConfig {
        recv_queue_depth: depth,
        ..NexarConfig::default()
    })
    .await
}

/// Two ranks on loopback with `config`, QUIC only.
#[cfg(test)]
pub(crate) async fn local_pair_with(mut config: NexarConfig) -> (NexarClient, NexarClient) {
    config.enable_tcp_bulk_sidecar = false;
    let adapter = Arc::new(crate::device::CpuAdapter::new());
    let mut clients = NexarClient::bootstrap_local_with_config(2, adapter, config)
        .await
        .expect("bootstrap two local ranks");
    clients.sort_by_key(|c| c.rank());
    let b = clients.pop().expect("rank 1");
    let a = clients.pop().expect("rank 0");
    (a, b)
}

/// Shut down both ranks of a [`local_pair`].
#[cfg(test)]
pub(crate) async fn shutdown_pair(a: NexarClient, b: NexarClient) {
    let (ra, rb) = tokio::join!(a.shutdown(), b.shutdown());
    ra.expect("rank 0 shutdown");
    rb.expect("rank 1 shutdown");
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
