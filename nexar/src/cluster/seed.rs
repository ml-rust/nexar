use crate::cluster::ClusterConfig;
use crate::cluster::seed_join::{HELLO_TIMEOUT, Joiner, MAX_HANDSHAKES};
use crate::cluster::topology::ClusterMap;
use crate::error::{NexarError, Result};
use crate::protocol::{AbortReason, NexarMessage};
use crate::transport::TransportListener;
use crate::transport::tls::ClusterCa;
use crate::types::Rank;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::Instant;

/// Result of cluster formation, including the CA for elastic scaling.
pub struct FormClusterResult {
    /// Mesh address of every rank.
    pub map: ClusterMap,
    /// Bootstrap connection to every rank, in rank order.
    pub connections: Vec<quinn::Connection>,
    pub ca: ClusterCa,
    pub next_rank: Rank,
}

/// The seed node orchestrates cluster formation.
///
/// It accepts workers, assigns ranks, issues per-node mTLS certificates from an
/// ephemeral CA, collects every rank's mesh address, and broadcasts the peer
/// table. It does not route data.
///
/// # Security
///
/// Workers do not verify the seed's bootstrap certificate. The cluster token
/// is the only defense against a stranger who joins or poses as the seed.
pub struct SeedNode {
    pub(super) listener: TransportListener,
    pub(super) expected_world_size: u32,
    pub(super) formation_timeout: Duration,
    /// Token every Hello must carry. `None`: no check (loopback only).
    pub(super) cluster_token: Option<Vec<u8>>,
    /// Permits for handshakes in flight. Caps the tasks a connection flood can start.
    pub(super) handshakes: Arc<Semaphore>,
    /// Time a connection has to send its Hello.
    pub(super) hello_timeout: Duration,
}

impl SeedNode {
    /// Create a seed node bound to the given address.
    ///
    /// If `NEXAR_CLUSTER_TOKEN` is set, workers must present the same token in
    /// their Hello. If it is not set, the seed generates a random ASCII hex
    /// token. The seed never logs it: read it with [`SeedNode::cluster_token`]
    /// and set it as `NEXAR_CLUSTER_TOKEN` on every worker over a trusted channel.
    pub fn bind(addr: SocketAddr, expected_world_size: u32) -> Result<Self> {
        let cluster_token = match std::env::var(crate::cluster::CLUSTER_TOKEN_ENV) {
            Ok(t) if !t.is_empty() => {
                tracing::info!("using cluster token from NEXAR_CLUSTER_TOKEN env var");
                t.into_bytes()
            }
            _ => {
                // The first 32 bytes of a fresh CSPRNG-backed key serve as the token.
                let ephemeral_key = rcgen::KeyPair::generate()
                    .map_err(|e| NexarError::Tls(format!("generate cluster token: {e}")))?;
                let token: Vec<u8> = ephemeral_key.serialize_der().into_iter().take(32).collect();
                let hex_token: String = token.iter().map(|b| format!("{b:02x}")).collect();
                tracing::warn!(
                    "NEXAR_CLUSTER_TOKEN not set: the seed generated a random cluster token; \
                     read it with SeedNode::cluster_token and set NEXAR_CLUSTER_TOKEN to it on every worker"
                );
                hex_token.into_bytes()
            }
        };
        Self::bind_with_token(addr, expected_world_size, Some(cluster_token))
    }

    /// Create a seed node without a token check, for a loopback address only.
    ///
    /// A non-loopback `addr` returns [`NexarError::TokenRequired`]: use
    /// [`SeedNode::bind_with_token`] or [`SeedNode::from_config`] with a token.
    pub fn bind_local(addr: SocketAddr, expected_world_size: u32) -> Result<Self> {
        Self::bind_with_token(addr, expected_world_size, None)
    }

    /// Create a seed node that checks `token`.
    ///
    /// `None` or an empty token disables the check. That is legal only on a
    /// loopback `addr`: a non-loopback `addr` without a token returns
    /// [`NexarError::TokenRequired`] before the seed binds.
    pub fn bind_with_token(
        addr: SocketAddr,
        expected_world_size: u32,
        token: Option<Vec<u8>>,
    ) -> Result<Self> {
        let token = token.filter(|t| !t.is_empty());
        if token.is_none() && !addr.ip().is_loopback() {
            return Err(NexarError::TokenRequired { addr });
        }
        Ok(Self {
            listener: TransportListener::bind(addr)?,
            expected_world_size,
            formation_timeout: crate::cluster::DEFAULT_JOIN_TIMEOUT,
            cluster_token: token,
            handshakes: Arc::new(Semaphore::new(MAX_HANDSHAKES)),
            hello_timeout: HELLO_TIMEOUT,
        })
    }

    /// Create the seed of a multi-process cluster.
    ///
    /// Takes the world size, token, and join timeout from `cfg`. A
    /// non-loopback `addr` without a token is an error.
    pub fn from_config(addr: SocketAddr, cfg: &ClusterConfig) -> Result<Self> {
        let token = cfg.checked_token(addr)?;
        Ok(Self::bind_with_token(addr, cfg.world_size, token)?
            .with_formation_timeout(cfg.join_timeout))
    }

    /// Set the cluster formation timeout.
    pub fn with_formation_timeout(mut self, timeout: Duration) -> Self {
        self.formation_timeout = timeout;
        self
    }

    /// Get the local address the seed is listening on.
    pub fn local_addr(&self) -> SocketAddr {
        self.listener.local_addr()
    }

    /// The number of ranks this seed waits for.
    pub fn world_size(&self) -> u32 {
        self.expected_world_size
    }

    /// The token this seed checks. `None`: no check (loopback only).
    ///
    /// Workers must present this token. It is a secret: never log it.
    pub fn cluster_token(&self) -> Option<&[u8]> {
        self.cluster_token.as_deref()
    }

    /// The token this seed checks, for the seed process's own join.
    pub(crate) fn token(&self) -> Option<Vec<u8>> {
        self.cluster_token.clone()
    }

    /// Form the cluster: join, mesh address exchange, and ready barrier.
    ///
    /// 1. Accepts workers until `world_size` valid Hellos arrive. A bad Hello
    ///    gets an `Abort` reply, and the seed keeps accepting.
    /// 2. Honors requested ranks and assigns the rest in arrival order.
    /// 3. Sends each worker a Welcome with its rank and a CA-signed leaf certificate.
    /// 4. Collects every rank's `MeshListen` and broadcasts the `PeerTable`.
    /// 5. Waits until every rank enters the ready barrier, then releases them.
    ///
    /// On timeout every worker gets an `Abort`, and the seed returns
    /// [`NexarError::ClusterFormationTimeout`]. When a rank reports a failure,
    /// the seed aborts the others and returns that rank's error.
    pub async fn form_cluster(&self) -> Result<FormClusterResult> {
        let deadline = Instant::now() + self.formation_timeout;
        let ca = ClusterCa::generate()?;
        let (mut joiners, mut events) = self.collect_joins(deadline).await?;
        let _late = self.reject_late_joins();

        if let Err(e) = self.send_welcomes(&ca, &mut joiners).await {
            let reason = AbortReason::Malformed {
                detail: format!("seed failed to send Welcome: {e}"),
            };
            abort_all(&mut joiners, reason).await;
            return Err(e);
        }

        let addrs = self
            .collect_stage(&mut joiners, &mut events, deadline, Stage::MeshListen)
            .await?;
        let mut peers: Vec<(Rank, String)> = addrs.into_iter().collect();
        peers.sort_by_key(|(r, _)| *r);
        let table = NexarMessage::PeerTable {
            peers: peers.clone(),
        };
        self.broadcast(&mut joiners, &table, false).await?;

        self.collect_stage(&mut joiners, &mut events, deadline, Stage::Ready)
            .await?;
        let release = NexarMessage::BarrierAck {
            epoch: 0,
            comm_id: crate::cluster::worker::BOOTSTRAP_COMM_ID,
        };
        self.broadcast(&mut joiners, &release, true).await?;

        let mut map = ClusterMap::new();
        for (rank, addr) in peers {
            map.add_peer(rank, addr);
        }
        tracing::info!(world_size = self.expected_world_size, "cluster formed");
        Ok(FormClusterResult {
            map,
            connections: joiners.into_iter().map(|j| j.conn).collect(),
            ca,
            next_rank: self.expected_world_size,
        })
    }
}

/// A formation step that needs one message from every rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    MeshListen,
    Ready,
}

impl Stage {
    pub(super) fn name(self) -> &'static str {
        match self {
            Stage::MeshListen => "MeshListen",
            Stage::Ready => "ready barrier",
        }
    }
}

/// Send `Abort` to every joiner concurrently and wait until each acknowledges it.
pub(super) async fn abort_all(joiners: &mut [Joiner], reason: AbortReason) {
    let futs = joiners
        .iter_mut()
        .map(|j| crate::cluster::control::send_abort(&mut j.send, reason.clone()));
    futures::future::join_all(futs).await;
}

/// A node waiting to join the cluster at the next elastic checkpoint.
#[derive(Debug, Clone)]
pub struct PendingJoin {
    pub rank: Rank,
    pub listen_addr: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_seed_bind() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let seed = SeedNode::bind(addr, 2).unwrap();
        assert_ne!(seed.local_addr().port(), 0);
    }

    #[tokio::test]
    async fn test_from_config_requires_token_off_loopback() {
        let cfg = ClusterConfig::new(2).with_cluster_token("");
        let addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
        assert!(matches!(
            SeedNode::from_config(addr, &cfg),
            Err(NexarError::TokenRequired { .. })
        ));
        let seed = SeedNode::from_config(addr, &cfg.with_cluster_token("t")).unwrap();
        assert_eq!(seed.world_size(), 2);
    }

    #[tokio::test]
    async fn test_bind_local_rejects_non_loopback() {
        let addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
        match SeedNode::bind_local(addr, 2) {
            Err(NexarError::TokenRequired { addr: a }) => assert_eq!(a, addr),
            other => panic!("expected TokenRequired, got {:?}", other.err()),
        }
        let seed = SeedNode::bind_local("127.0.0.1:0".parse().unwrap(), 2).unwrap();
        assert!(seed.cluster_token().is_none());
    }

    #[tokio::test]
    async fn test_bind_with_token_requires_token_off_loopback() {
        let addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
        assert!(matches!(
            SeedNode::bind_with_token(addr, 2, None),
            Err(NexarError::TokenRequired { .. })
        ));
        assert!(matches!(
            SeedNode::bind_with_token(addr, 2, Some(Vec::new())),
            Err(NexarError::TokenRequired { .. })
        ));
        let seed = SeedNode::bind_with_token(addr, 2, Some(b"t".to_vec())).unwrap();
        assert_eq!(seed.cluster_token(), Some(&b"t"[..]));
    }

    #[tokio::test]
    async fn test_bind_always_has_token() {
        let seed = SeedNode::bind("0.0.0.0:0".parse().unwrap(), 2).unwrap();
        assert!(seed.cluster_token().is_some_and(|t| !t.is_empty()));
    }
}
