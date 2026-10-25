use crate::cluster::control::{
    ControlEvent, control_channel, spawn_reader, write_control, write_final,
};
use crate::error::{NexarError, Result};
use crate::protocol::{AbortReason, NexarMessage};
use crate::transport::tls::make_bootstrap_client_config;
use crate::types::{PROTOCOL_VERSION, Rank};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;

/// `comm_id` of the ready barrier that ends cluster formation.
pub(crate) const BOOTSTRAP_COMM_ID: u64 = u64::MAX;

/// Longest single connect attempt to the seed before a retry.
const CONNECT_ATTEMPT: Duration = Duration::from_secs(10);
/// Largest pause between connect attempts.
const MAX_BACKOFF: Duration = Duration::from_secs(2);

/// What a worker presents to the seed.
pub(crate) struct JoinRequest {
    pub token: Option<Vec<u8>>,
    pub requested_rank: Option<Rank>,
    pub advertise_host: Option<IpAddr>,
    pub timeout: Duration,
}

/// A worker that completed the Hello/Welcome handshake with the seed.
///
/// Holds the rank, the mTLS credentials, and the open control stream to the
/// seed. Dropping it closes the seed connection.
pub struct WorkerNode {
    pub rank: Rank,
    pub world_size: u32,
    /// `(rank, mesh address)` for every rank. Filled from the seed's `PeerTable`.
    pub peers: Vec<(Rank, String)>,
    pub seed_conn: quinn::Connection,
    /// DER-encoded cluster CA certificate (trust anchor for mesh mTLS).
    pub ca_cert: Vec<u8>,
    /// DER-encoded leaf certificate for this node, signed by the cluster CA.
    pub node_cert: Vec<u8>,
    /// DER-encoded private key for this node's leaf certificate.
    pub node_key: Vec<u8>,
    /// This worker's IP address as the seed observes it.
    pub observed_ip: Option<IpAddr>,
    session: SeedSession,
}

/// The control stream to the seed and the formation deadline.
struct SeedSession {
    send: quinn::SendStream,
    events: mpsc::Receiver<ControlEvent<()>>,
    reader: tokio::task::JoinHandle<()>,
    _endpoint: quinn::Endpoint,
    deadline: Instant,
    timeout: Duration,
}

impl WorkerNode {
    /// Connect to the seed, complete the handshake, and receive a rank.
    ///
    /// Presents the token from `NEXAR_CLUSTER_TOKEN` and waits at most the
    /// default join timeout for the Welcome. The bootstrap connection does not
    /// verify the seed's certificate. The cluster token is the only defense.
    /// Use [`NexarClient::bootstrap_join`](crate::NexarClient::bootstrap_join)
    /// to also build the mesh.
    pub async fn connect(seed_addr: SocketAddr) -> Result<Self> {
        let req = JoinRequest {
            token: std::env::var(crate::cluster::CLUSTER_TOKEN_ENV)
                .ok()
                .filter(|t| !t.is_empty())
                .map(String::into_bytes),
            requested_rank: None,
            advertise_host: None,
            timeout: crate::cluster::DEFAULT_JOIN_TIMEOUT,
        };
        Self::join(seed_addr, &req).await
    }

    /// Connect to the seed, send Hello, and wait for Welcome or Abort.
    pub(crate) async fn join(seed_addr: SocketAddr, req: &JoinRequest) -> Result<Self> {
        let deadline = Instant::now() + req.timeout;
        let mut endpoint = quinn::Endpoint::client(unspecified_for(seed_addr))
            .map_err(|e| NexarError::transport_with_source("bind bootstrap client", e))?;
        endpoint.set_default_client_config(make_bootstrap_client_config()?);

        let conn = connect_with_retry(&endpoint, seed_addr, deadline, req.timeout).await?;
        let (mut send, recv) = conn
            .open_bi()
            .await
            .map_err(|e| NexarError::transport_with_source("open control stream to seed", e))?;
        let hello = NexarMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            capabilities: 0,
            cluster_token: req.token.clone().unwrap_or_default(),
            listen_addr: req
                .advertise_host
                .map(|ip| ip.to_string())
                .unwrap_or_default(),
            requested_rank: req.requested_rank,
        };
        write_control(&mut send, &hello).await?;

        let (tx, events) = control_channel();
        let reader = spawn_reader((), recv, tx);
        let mut session = SeedSession {
            send,
            events,
            reader,
            _endpoint: endpoint,
            deadline,
            timeout: req.timeout,
        };

        let welcome = match session.next("Welcome").await {
            Ok(msg) => msg,
            Err(e) => {
                close_seed_conn(&conn, &session._endpoint).await;
                return Err(e);
            }
        };
        match welcome {
            NexarMessage::Welcome {
                rank,
                world_size,
                peers,
                ca_cert,
                node_cert,
                node_key,
                observed_addr,
            } => Ok(WorkerNode {
                rank,
                world_size,
                peers,
                seed_conn: conn,
                ca_cert,
                node_cert,
                node_key,
                observed_ip: observed_addr.parse().ok(),
                session,
            }),
            other => {
                close_seed_conn(&conn, &session._endpoint).await;
                Err(NexarError::DecodeFailed(format!(
                    "expected Welcome from the seed, got {other:?}"
                )))
            }
        }
    }

    /// Close the seed connection and wait until the close leaves the endpoint.
    ///
    /// The seed then sees the close at once instead of waiting for an ACK.
    pub(crate) async fn close(&mut self) {
        self.session.reader.abort();
        close_seed_conn(&self.seed_conn, &self.session._endpoint).await;
    }

    /// The instant at which cluster formation times out for this worker.
    pub(crate) fn deadline(&self) -> Instant {
        self.session.deadline
    }

    /// The join timeout this worker runs with.
    pub(crate) fn join_timeout(&self) -> Duration {
        self.session.timeout
    }

    /// Report the mesh listener address and wait for the full peer table.
    pub(crate) async fn announce_mesh(&mut self, addr: String) -> Result<Vec<(Rank, String)>> {
        let msg = NexarMessage::MeshListen {
            rank: self.rank,
            addr,
        };
        write_control(&mut self.session.send, &msg).await?;
        match self.session.next("PeerTable").await? {
            NexarMessage::PeerTable { mut peers } => {
                peers.sort_by_key(|(r, _)| *r);
                let complete = peers.len() == self.world_size as usize
                    && peers.iter().enumerate().all(|(i, (r, _))| *r as usize == i);
                if !complete {
                    return Err(NexarError::DecodeFailed(format!(
                        "seed sent a peer table without every rank of {}: {peers:?}",
                        self.world_size
                    )));
                }
                self.peers = peers.clone();
                Ok(peers)
            }
            other => Err(NexarError::DecodeFailed(format!(
                "expected PeerTable from the seed, got {other:?}"
            ))),
        }
    }

    /// Wait until the seed sends something. Returns the error that ends formation.
    ///
    /// Cancel-safe: callers race it against mesh setup to stop on an `Abort`.
    pub(crate) async fn next_abort(&mut self) -> NexarError {
        match self.session.events.recv().await {
            Some((_, Ok(NexarMessage::Abort { reason }))) => reason.into_error(),
            Some((_, Ok(other))) => NexarError::DecodeFailed(format!(
                "unexpected {other:?} from the seed during mesh setup"
            )),
            Some((_, Err(e))) => lost_seed(e),
            None => lost_seed("reader stopped"),
        }
    }

    /// Tell the seed that this rank failed, so it aborts the other ranks.
    pub(crate) async fn report_failure(&mut self, err: &NexarError) {
        let reason = AbortReason::PeerFailed {
            rank: self.rank,
            reason: err.to_string(),
        };
        let _ = write_final(&mut self.session.send, &NexarMessage::Abort { reason }).await;
    }

    /// Enter the ready barrier and wait until every rank is ready.
    pub(crate) async fn ready(&mut self) -> Result<()> {
        let msg = NexarMessage::Barrier {
            epoch: 0,
            comm_id: BOOTSTRAP_COMM_ID,
        };
        write_control(&mut self.session.send, &msg).await?;
        match self.session.next("ready barrier").await? {
            NexarMessage::BarrierAck {
                comm_id: BOOTSTRAP_COMM_ID,
                ..
            } => Ok(()),
            other => Err(NexarError::DecodeFailed(format!(
                "expected the ready BarrierAck from the seed, got {other:?}"
            ))),
        }
    }
}

impl SeedSession {
    /// The next message from the seed before the deadline. `Abort` becomes its error.
    async fn next(&mut self, stage: &'static str) -> Result<NexarMessage> {
        let event = tokio::time::timeout_at(self.deadline, self.events.recv())
            .await
            .map_err(|_| NexarError::JoinTimeout {
                stage,
                timeout_ms: self.timeout.as_millis() as u64,
            })?;
        match event {
            Some((_, Ok(NexarMessage::Abort { reason }))) => Err(reason.into_error()),
            Some((_, Ok(msg))) => Ok(msg),
            Some((_, Err(e))) => Err(lost_seed(e)),
            None => Err(lost_seed("reader stopped")),
        }
    }
}

impl Drop for WorkerNode {
    fn drop(&mut self) {
        self.session.reader.abort();
        self.seed_conn.close(0u32.into(), b"worker done");
    }
}

/// Longest wait for a close frame to leave the bootstrap endpoint.
const CLOSE_FLUSH: Duration = Duration::from_secs(1);

async fn close_seed_conn(conn: &quinn::Connection, endpoint: &quinn::Endpoint) {
    conn.close(0u32.into(), b"worker done");
    let _ = tokio::time::timeout(CLOSE_FLUSH, endpoint.wait_idle()).await;
}

fn lost_seed(e: impl std::fmt::Display) -> NexarError {
    NexarError::BootstrapAborted {
        reason: format!("lost the seed connection: {e}"),
    }
}

/// The wildcard address of `addr`'s family with port 0.
pub(crate) fn unspecified_for(addr: SocketAddr) -> SocketAddr {
    let ip = if addr.is_ipv4() {
        IpAddr::V4(Ipv4Addr::UNSPECIFIED)
    } else {
        IpAddr::V6(Ipv6Addr::UNSPECIFIED)
    };
    SocketAddr::new(ip, 0)
}

/// Connect to the seed, retrying until `deadline`. The seed can start after the worker.
async fn connect_with_retry(
    endpoint: &quinn::Endpoint,
    seed_addr: SocketAddr,
    deadline: Instant,
    timeout: Duration,
) -> Result<quinn::Connection> {
    let mut backoff = Duration::from_millis(100);
    let mut last_err = String::from("no attempt finished");
    loop {
        let now = Instant::now();
        if now >= deadline {
            tracing::warn!("seed {seed_addr} unreachable: {last_err}");
            return Err(NexarError::JoinTimeout {
                stage: "seed connection",
                timeout_ms: timeout.as_millis() as u64,
            });
        }
        let attempt_end = deadline.min(now + CONNECT_ATTEMPT);
        let connecting = endpoint
            .connect(seed_addr, "localhost")
            .map_err(|e| NexarError::transport_with_source("connect to seed", e))?;
        match tokio::time::timeout_at(attempt_end, connecting).await {
            Ok(Ok(conn)) => return Ok(conn),
            Ok(Err(e)) => last_err = e.to_string(),
            Err(_) => last_err = "handshake timed out".into(),
        }
        tracing::debug!("seed {seed_addr} not reachable yet: {last_err}");
        tokio::time::sleep(backoff.min(deadline.saturating_duration_since(Instant::now()))).await;
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}
