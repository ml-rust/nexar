//! Multi-process bootstrap: one process per rank, joined through a seed.

use crate::client::NexarClient;
use crate::client::bootstrap_mesh::{MeshAddrs, join_mesh};
use crate::cluster::worker::JoinRequest;
use crate::cluster::{ClusterConfig, NodeCredentials, SeedNode, WorkerNode};
use crate::device::DeviceAdapter;
use crate::error::{NexarError, Result};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

/// How long the seed process waits for its seed to abort the others after its own join fails.
const SEED_WIND_DOWN: Duration = Duration::from_secs(10);

impl NexarClient {
    /// Bootstrap rank 0 of a multi-process cluster.
    ///
    /// Binds the seed on `seed_addr`, joins it as rank 0, and returns once all
    /// `cfg.world_size` ranks formed the mesh. Every other process calls
    /// [`NexarClient::bootstrap_join`] with the same `seed_addr`.
    ///
    /// A non-loopback address without a token is an error: see [`ClusterConfig`].
    pub async fn bootstrap_seed(
        seed_addr: SocketAddr,
        cfg: &ClusterConfig,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Result<NexarClient> {
        let seed = SeedNode::from_config(seed_addr, cfg)?;
        Self::bootstrap_seed_with(seed, cfg, adapter).await
    }

    /// Like [`NexarClient::bootstrap_seed`], with a seed the caller already bound.
    ///
    /// Bind with [`SeedNode::from_config`] and read [`SeedNode::local_addr`]
    /// to learn the port when `seed_addr` uses port 0.
    pub async fn bootstrap_seed_with(
        seed: SeedNode,
        cfg: &ClusterConfig,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Result<NexarClient> {
        cfg.checked_token(seed.local_addr())?;
        if seed.world_size() != cfg.world_size {
            return Err(NexarError::BootstrapAborted {
                reason: format!(
                    "seed expects world size {} and the config says {}; use the same world_size",
                    seed.world_size(),
                    cfg.world_size
                ),
            });
        }
        if let Some(rank) = cfg.requested_rank
            && rank != 0
        {
            return Err(NexarError::RankConflict {
                rank,
                detail: "the seed process always holds rank 0; leave requested_rank unset or 0"
                    .into(),
            });
        }
        let bound = seed.local_addr();
        if cfg.world_size > 1 && cfg.advertise_host().is_none() && bound.ip().is_unspecified() {
            return Err(NexarError::BootstrapAborted {
                reason: format!(
                    "the seed binds {bound} and rank 0 has no address peers can dial; \
                     set ClusterConfig::advertise_addr or bind_addr to this host's IP"
                ),
            });
        }
        let mut own = cfg.clone();
        own.requested_rank = Some(0);
        let own_seed = dialable(bound);

        let formed = seed.form_cluster();
        tokio::pin!(formed);
        let joined = join_cluster(own_seed, &own, seed.token(), adapter);
        tokio::pin!(joined);
        tokio::select! {
            result = &mut formed => {
                result?;
                Ok(joined.await?.0)
            }
            result = &mut joined => match result {
                Ok((client, _)) => {
                    formed.await?;
                    Ok(client)
                }
                Err(e) => {
                    let _ = tokio::time::timeout(SEED_WIND_DOWN, &mut formed).await;
                    Err(e)
                }
            }
        }
    }

    /// Bootstrap a non-seed rank of a multi-process cluster.
    ///
    /// Connects to the seed at `seed_addr`, retrying until `cfg.join_timeout`,
    /// then forms the mesh. Returns once every rank is ready.
    pub async fn bootstrap_join(
        seed_addr: SocketAddr,
        cfg: &ClusterConfig,
        adapter: Arc<dyn DeviceAdapter>,
    ) -> Result<NexarClient> {
        let token = cfg.checked_token(seed_addr)?;
        Ok(join_cluster(seed_addr, cfg, token, adapter).await?.0)
    }
}

/// Join the seed at `seed_addr` and build this rank's mesh.
pub(super) async fn join_cluster(
    seed_addr: SocketAddr,
    cfg: &ClusterConfig,
    token: Option<Vec<u8>>,
    adapter: Arc<dyn DeviceAdapter>,
) -> Result<(NexarClient, NodeCredentials)> {
    let req = JoinRequest {
        token,
        requested_rank: cfg.requested_rank,
        advertise_host: cfg.advertise_host(),
        timeout: cfg.join_timeout,
    };
    let mut worker = WorkerNode::join(seed_addr, &req).await?;
    if worker.world_size != cfg.world_size {
        let err = NexarError::BootstrapAborted {
            reason: format!(
                "seed expects world size {} and rank {} was configured with {}; use the same world_size on every process",
                worker.world_size, worker.rank, cfg.world_size
            ),
        };
        worker.report_failure(&err).await;
        worker.close().await;
        return Err(err);
    }
    let creds = worker.credentials();
    let addrs = MeshAddrs {
        bind: cfg.bind_addr,
        advertise: cfg.advertise_addr,
    };
    let client = join_mesh(&mut worker, &addrs, adapter, cfg.nexar.clone()).await;
    worker.close().await;
    Ok((client?, creds))
}

/// `addr` with an unspecified IP replaced by loopback of the same family.
fn dialable(addr: SocketAddr) -> SocketAddr {
    match addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), addr.port())
        }
        IpAddr::V6(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv6Addr::LOCALHOST.into(), addr.port())
        }
        _ => addr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::sparse::TopologyStrategy;
    use crate::config::NexarConfig;
    use crate::device::CpuAdapter;
    use crate::transport::TaggedBulkTransport;
    use crate::types::{DataType, Rank, ReduceOp};
    use std::future::Future;
    use std::thread::JoinHandle;

    const TOKEN: &str = "bootstrap-test-token";
    const SIDECAR_TAG: u64 = 0xB007;

    /// What one rank observed after the bootstrap.
    #[derive(Debug)]
    struct Outcome {
        rank: Rank,
        sum: f32,
        from_prev: f32,
        sidecar_from_prev: Option<Vec<u8>>,
        direct_peers: usize,
    }

    fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    fn cfg(world: u32) -> ClusterConfig {
        ClusterConfig::new(world)
            .with_bind_addr(loopback())
            .with_cluster_token(TOKEN)
            .with_join_timeout(Duration::from_secs(30))
            .with_nexar(NexarConfig::default())
    }

    fn adapter() -> Arc<dyn DeviceAdapter> {
        Arc::new(CpuAdapter::new())
    }

    /// Run `f` on its own thread with its own tokio runtime.
    fn on_thread<F, Fut, T>(f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = T>,
        T: Send + 'static,
    {
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap()
                .block_on(f())
        })
    }

    /// Allreduce, a tagged ring send/recv, and a sidecar frame to the next rank.
    async fn exercise(client: NexarClient) -> Result<Outcome> {
        let client = Arc::new(client);
        let (rank, n) = (client.rank(), client.world_size());
        let mut data = vec![(rank + 1) as f32; 256];
        unsafe {
            client
                .all_reduce(
                    data.as_mut_ptr() as u64,
                    data.len(),
                    DataType::F32,
                    ReduceOp::Sum,
                )
                .await?;
        }
        let (next, prev) = ((rank + 1) % n, (rank + n - 1) % n);
        let payload = [rank as f32; 16];
        let mut got = vec![-1.0f32; 16];
        let sender = Arc::clone(&client);
        let send_ptr = payload.as_ptr() as u64;
        let send = tokio::spawn(async move { unsafe { sender.send(send_ptr, 64, next, 7).await } });
        unsafe { client.recv(got.as_mut_ptr() as u64, 64, prev, 7).await? };
        send.await
            .map_err(|e| NexarError::transport_with_source("send task", e))??;

        let sidecar = |r: Rank| -> Result<Option<Arc<dyn TaggedBulkTransport>>> {
            Ok(client
                .peer(r)?
                .extension::<Arc<dyn TaggedBulkTransport>>()?
                .map(|t| Arc::clone(&*t)))
        };
        let mut sidecar_from_prev = None;
        if let (Some(to_next), Some(from_prev)) = (sidecar(next)?, sidecar(prev)?) {
            to_next
                .send_bulk_tagged(SIDECAR_TAG, &[rank as u8; 32])
                .await?;
            sidecar_from_prev = Some(from_prev.recv_bulk_tagged(SIDECAR_TAG, 32).await?);
        }
        client.barrier().await?;
        let direct_peers = client.peers.len();
        let client = Arc::into_inner(client).expect("no other client handle is left");
        client.shutdown().await?;
        Ok(Outcome {
            rank,
            sum: data[0],
            from_prev: got[0],
            sidecar_from_prev,
            direct_peers,
        })
    }

    /// Bind a seed on loopback and run rank 0 on its own thread.
    fn start_seed(cfg: ClusterConfig) -> (SocketAddr, JoinHandle<Result<Outcome>>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = on_thread(move || async move {
            let seed = SeedNode::from_config(loopback(), &cfg)?;
            tx.send(seed.local_addr()).unwrap();
            let client = NexarClient::bootstrap_seed_with(seed, &cfg, adapter()).await?;
            exercise(client).await
        });
        (rx.recv().unwrap(), handle)
    }

    fn start_worker(seed: SocketAddr, cfg: ClusterConfig) -> JoinHandle<Result<Outcome>> {
        on_thread(move || async move {
            let client = NexarClient::bootstrap_join(seed, &cfg, adapter()).await?;
            exercise(client).await
        })
    }

    fn check_ring(outcomes: &[Outcome], n: u32) {
        let mut ranks: Vec<Rank> = outcomes.iter().map(|o| o.rank).collect();
        ranks.sort();
        assert_eq!(ranks, (0..n).collect::<Vec<_>>());
        for o in outcomes {
            assert_eq!(o.sum, (n * (n + 1) / 2) as f32, "rank {}", o.rank);
            assert_eq!(
                o.from_prev,
                ((o.rank + n - 1) % n) as f32,
                "rank {}",
                o.rank
            );
        }
    }

    fn run_cluster(n: u32, cfg: ClusterConfig) -> Vec<Outcome> {
        let (seed, rank0) = start_seed(cfg.clone());
        let workers: Vec<_> = (1..n).map(|_| start_worker(seed, cfg.clone())).collect();
        let mut outcomes = vec![rank0.join().unwrap().unwrap()];
        for w in workers {
            outcomes.push(w.join().unwrap().unwrap());
        }
        check_ring(&outcomes, n);
        outcomes
    }

    #[test]
    fn test_two_ranks_with_tls_sidecar() {
        let outcomes = run_cluster(2, cfg(2));
        for o in &outcomes {
            let prev = (o.rank + 1) % 2;
            assert_eq!(
                o.sidecar_from_prev,
                Some(vec![prev as u8; 32]),
                "rank {}",
                o.rank
            );
        }
    }

    #[test]
    fn test_four_ranks_plaintext_sidecar() {
        let mut c = cfg(4);
        c.nexar.encrypt_bulk_transport = false;
        let outcomes = run_cluster(4, c);
        for o in &outcomes {
            let prev = (o.rank + 3) % 4;
            assert_eq!(o.sidecar_from_prev, Some(vec![prev as u8; 32]));
            assert_eq!(o.direct_peers, 3);
        }
    }

    #[test]
    fn test_four_ranks_without_sidecar() {
        let mut c = cfg(4);
        c.nexar.enable_tcp_bulk_sidecar = false;
        for o in run_cluster(4, c) {
            assert_eq!(o.sidecar_from_prev, None);
        }
    }

    #[test]
    fn test_sparse_ring_topology() {
        let mut c = cfg(6);
        c.nexar.topology = TopologyStrategy::KRegular { degree: 2 };
        for o in run_cluster(6, c) {
            assert_eq!(o.direct_peers, 2, "rank {}", o.rank);
            assert!(o.sidecar_from_prev.is_some(), "rank {}", o.rank);
        }
    }

    #[test]
    fn test_requested_ranks_are_honored() {
        let (seed, rank0) = start_seed(cfg(4));
        let w3 = start_worker(seed, cfg(4).with_requested_rank(3));
        let w1 = start_worker(seed, cfg(4).with_requested_rank(1));
        let free = start_worker(seed, cfg(4));
        assert_eq!(rank0.join().unwrap().unwrap().rank, 0);
        assert_eq!(w3.join().unwrap().unwrap().rank, 3);
        assert_eq!(w1.join().unwrap().unwrap().rank, 1);
        assert_eq!(free.join().unwrap().unwrap().rank, 2);
    }

    #[test]
    fn test_duplicate_requested_rank_is_rejected() {
        let (seed, rank0) = start_seed(cfg(3));
        let (tx, rx) = std::sync::mpsc::channel();
        for _ in 0..2 {
            let tx = tx.clone();
            on_thread(move || async move {
                let result = async {
                    let c = cfg(3).with_requested_rank(1);
                    let client = NexarClient::bootstrap_join(seed, &c, adapter()).await?;
                    exercise(client).await
                }
                .await;
                tx.send(result).unwrap();
            });
        }
        // The winner waits for a third rank, so the rejected claim finishes first.
        let lost = rx.recv().unwrap();
        assert!(
            matches!(lost, Err(NexarError::RankConflict { rank: 1, .. })),
            "{lost:?}"
        );
        // The seed keeps accepting after the rejection: a third worker completes the cluster.
        let c = start_worker(seed, cfg(3));
        assert_eq!(rx.recv().unwrap().unwrap().rank, 1);
        assert_eq!(c.join().unwrap().unwrap().rank, 2);
        assert_eq!(rank0.join().unwrap().unwrap().rank, 0);
    }

    #[test]
    fn test_token_mismatch_is_rejected() {
        let (seed, rank0) = start_seed(cfg(2));
        let bad = start_worker(seed, cfg(2).with_cluster_token("wrong"));
        assert!(matches!(
            bad.join().unwrap(),
            Err(NexarError::ClusterTokenMismatch)
        ));
        let good = start_worker(seed, cfg(2));
        assert_eq!(good.join().unwrap().unwrap().rank, 1);
        assert_eq!(rank0.join().unwrap().unwrap().rank, 0);
    }

    #[test]
    fn test_join_timeout_aborts_every_side() {
        let (tx, rx) = std::sync::mpsc::channel();
        let seed = on_thread(move || async move {
            let seed = SeedNode::from_config(loopback(), &cfg(3))?
                .with_formation_timeout(Duration::from_secs(2));
            tx.send(seed.local_addr()).unwrap();
            seed.form_cluster().await.map(|_| ())
        });
        let seed_addr = rx.recv().unwrap();
        let workers: Vec<_> = (0..2).map(|_| start_worker(seed_addr, cfg(3))).collect();
        let expect = |r: Result<()>| {
            assert!(
                matches!(
                    r,
                    Err(NexarError::ClusterFormationTimeout {
                        joined: 2,
                        expected: 3
                    })
                ),
                "{r:?}"
            )
        };
        expect(seed.join().unwrap());
        for w in workers {
            expect(w.join().unwrap().map(|_| ()));
        }
    }

    #[test]
    fn test_garbage_connections_do_not_stop_formation() {
        let (seed, rank0) = start_seed(cfg(2));
        on_thread(move || async move {
            let udp = std::net::UdpSocket::bind(loopback()).unwrap();
            udp.send_to(b"not a quic packet", seed).unwrap();
            let mut endpoint = quinn::Endpoint::client(loopback()).unwrap();
            endpoint.set_default_client_config(
                crate::transport::tls::make_bootstrap_client_config().unwrap(),
            );
            // A connection that sends garbage on its control stream.
            let garbage = endpoint.connect(seed, "localhost").unwrap().await.unwrap();
            let (mut send, mut recv) = garbage.open_bi().await.unwrap();
            send.write_all(&[0xFF; 64]).await.unwrap();
            let reply = crate::cluster::control::read_control(&mut recv)
                .await
                .unwrap();
            assert!(
                matches!(reply, crate::protocol::NexarMessage::Abort { .. }),
                "{reply:?}"
            );
            // A connection that never sends a Hello stays open during formation.
            let _silent = endpoint.connect(seed, "localhost").unwrap().await.unwrap();
            let good = NexarClient::bootstrap_join(seed, &cfg(2), adapter())
                .await
                .unwrap();
            exercise(good).await.unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(rank0.join().unwrap().unwrap().rank, 0);
    }

    #[test]
    fn test_world_size_mismatch_is_rejected() {
        let (seed, rank0) = start_seed(cfg(2).with_join_timeout(Duration::from_secs(5)));
        let wrong = start_worker(seed, cfg(3));
        assert!(matches!(
            wrong.join().unwrap(),
            Err(NexarError::BootstrapAborted { .. })
        ));
        assert!(rank0.join().unwrap().is_err());
    }
}
