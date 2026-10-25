use crate::config::NexarConfig;
use crate::error::{NexarError, Result};
use crate::types::Rank;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

/// Environment variable read when [`ClusterConfig::cluster_token`] is `None`.
pub const CLUSTER_TOKEN_ENV: &str = "NEXAR_CLUSTER_TOKEN";

/// Default time a process waits for the whole cluster to form.
pub const DEFAULT_JOIN_TIMEOUT: Duration = Duration::from_secs(60);

/// Settings for one process that joins a multi-process cluster.
///
/// Pass it to [`NexarClient::bootstrap_seed`](crate::NexarClient::bootstrap_seed)
/// on the rank-0 process and to
/// [`NexarClient::bootstrap_join`](crate::NexarClient::bootstrap_join) on every other process.
///
/// # Security
///
/// The bootstrap connection to the seed does not authenticate the seed's
/// certificate. The cluster token is the only defense against a stranger who
/// joins the cluster or poses as the seed. A bind, seed, or advertise address
/// that is not loopback therefore requires a token.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Number of ranks in the cluster. Every process must use the same value.
    pub world_size: u32,
    /// Local address of this rank's mesh listener (QUIC on UDP, sidecar on TCP).
    /// Port 0 picks a free port.
    pub bind_addr: SocketAddr,
    /// Address that peers dial to reach this rank. `None`: the bind address.
    /// An unspecified bind IP is replaced by the IP the seed observes.
    /// Port 0 means the bound port.
    pub advertise_addr: Option<SocketAddr>,
    /// Rank this process asks for. `None`: the seed assigns one in arrival order.
    pub requested_rank: Option<Rank>,
    /// Shared secret for the bootstrap handshake. `None`: [`CLUSTER_TOKEN_ENV`].
    pub cluster_token: Option<String>,
    /// Time limit for the whole formation: join, mesh, and ready barrier.
    pub join_timeout: Duration,
    /// Runtime config of the client that the bootstrap returns.
    pub nexar: NexarConfig,
}

impl ClusterConfig {
    /// Config for a cluster of `world_size` ranks that binds every interface on a free port.
    pub fn new(world_size: u32) -> Self {
        Self {
            world_size,
            bind_addr: SocketAddr::new(IpAddr::from([0, 0, 0, 0]), 0),
            advertise_addr: None,
            requested_rank: None,
            cluster_token: None,
            join_timeout: DEFAULT_JOIN_TIMEOUT,
            nexar: NexarConfig::from_env(),
        }
    }

    /// Set [`ClusterConfig::bind_addr`].
    pub fn with_bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = addr;
        self
    }

    /// Set [`ClusterConfig::advertise_addr`].
    pub fn with_advertise_addr(mut self, addr: SocketAddr) -> Self {
        self.advertise_addr = Some(addr);
        self
    }

    /// Set [`ClusterConfig::requested_rank`].
    pub fn with_requested_rank(mut self, rank: Rank) -> Self {
        self.requested_rank = Some(rank);
        self
    }

    /// Set [`ClusterConfig::cluster_token`].
    pub fn with_cluster_token(mut self, token: impl Into<String>) -> Self {
        self.cluster_token = Some(token.into());
        self
    }

    /// Set [`ClusterConfig::join_timeout`].
    pub fn with_join_timeout(mut self, timeout: Duration) -> Self {
        self.join_timeout = timeout;
        self
    }

    /// Set [`ClusterConfig::nexar`].
    pub fn with_nexar(mut self, nexar: NexarConfig) -> Self {
        self.nexar = nexar;
        self
    }

    /// The token from the config, else from [`CLUSTER_TOKEN_ENV`]. Empty counts as unset.
    pub fn resolved_token(&self) -> Option<Vec<u8>> {
        match &self.cluster_token {
            Some(t) if !t.is_empty() => Some(t.clone().into_bytes()),
            Some(_) => None,
            None => std::env::var(CLUSTER_TOKEN_ENV)
                .ok()
                .filter(|t| !t.is_empty())
                .map(String::into_bytes),
        }
    }

    /// Check the config for a process that talks to the seed at `seed_addr`.
    ///
    /// Returns the token to present. Errors when the world is empty, the
    /// requested rank is out of range, or a non-loopback address has no token.
    pub(crate) fn checked_token(&self, seed_addr: SocketAddr) -> Result<Option<Vec<u8>>> {
        if self.world_size == 0 {
            return Err(NexarError::BootstrapAborted {
                reason: "world_size is 0; set ClusterConfig::world_size to at least 1".into(),
            });
        }
        if let Some(rank) = self.requested_rank
            && rank >= self.world_size
        {
            return Err(NexarError::InvalidRank {
                rank,
                world_size: self.world_size,
            });
        }
        let token = self.resolved_token();
        if token.is_none() {
            let exposed = [Some(seed_addr), Some(self.bind_addr), self.advertise_addr]
                .into_iter()
                .flatten()
                .find(|a| !a.ip().is_loopback());
            if let Some(addr) = exposed {
                return Err(NexarError::TokenRequired { addr });
            }
        }
        Ok(token)
    }

    /// The IP this rank tells the seed it advertises, if the config fixes one.
    pub(crate) fn advertise_host(&self) -> Option<IpAddr> {
        self.advertise_addr
            .map(|a| a.ip())
            .filter(|ip| !ip.is_unspecified())
            .or(Some(self.bind_addr.ip()).filter(|ip| !ip.is_unspecified()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    #[test]
    fn test_loopback_needs_no_token() {
        let cfg = ClusterConfig::new(2)
            .with_bind_addr(loopback())
            .with_cluster_token("");
        assert!(cfg.checked_token(loopback()).unwrap().is_none());
    }

    #[test]
    fn test_non_loopback_requires_token() {
        let cfg = ClusterConfig::new(2).with_cluster_token("");
        let err = cfg.checked_token(loopback()).unwrap_err();
        assert!(matches!(err, NexarError::TokenRequired { .. }), "{err}");
        let cfg = cfg.with_cluster_token("s3cret");
        assert_eq!(
            cfg.checked_token(loopback()).unwrap(),
            Some(b"s3cret".to_vec())
        );
    }

    #[test]
    fn test_requested_rank_out_of_range() {
        let cfg = ClusterConfig::new(2)
            .with_bind_addr(loopback())
            .with_requested_rank(2);
        assert!(matches!(
            cfg.checked_token(loopback()),
            Err(NexarError::InvalidRank { rank: 2, .. })
        ));
    }

    #[test]
    fn test_advertise_host_precedence() {
        let cfg = ClusterConfig::new(2);
        assert_eq!(cfg.advertise_host(), None);
        let cfg = cfg.with_bind_addr("10.0.0.5:7000".parse().unwrap());
        assert_eq!(cfg.advertise_host(), Some("10.0.0.5".parse().unwrap()));
        let cfg = cfg.with_advertise_addr("192.0.2.9:0".parse().unwrap());
        assert_eq!(cfg.advertise_host(), Some("192.0.2.9".parse().unwrap()));
    }
}
