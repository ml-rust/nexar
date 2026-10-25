//! Join phase of cluster formation: accept workers, check each Hello, assign ranks.

use crate::cluster::SeedNode;
use crate::cluster::control::{
    ControlEvent, control_channel, read_control, send_abort, spawn_reader, tokens_match,
};
use crate::cluster::seed::abort_all;
use crate::error::{NexarError, Result};
use crate::protocol::{AbortReason, NexarMessage};
use crate::types::{PROTOCOL_VERSION, Rank};
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::time::Instant;

/// Time a new connection has to finish the QUIC handshake and send its Hello.
pub(super) const HELLO_TIMEOUT: Duration = Duration::from_secs(10);

/// Handshakes that run at once. The seed refuses further connections until one finishes.
///
/// A refused worker retries, so a flood of stalled connections delays a join by
/// at most [`HELLO_TIMEOUT`] and cannot stop it.
pub(super) const MAX_HANDSHAKES: usize = 64;

/// Messages from joiners' control streams, keyed by joiner id.
pub(super) type JoinEvents = mpsc::Receiver<ControlEvent<u64>>;

/// A worker whose Hello the seed accepted.
pub(super) struct Joiner {
    /// Arrival order. Keys this joiner's control events.
    pub id: u64,
    pub conn: quinn::Connection,
    pub send: quinn::SendStream,
    pub requested: Option<Rank>,
    /// The IP the worker advertises, if it named one.
    pub advertised: Option<IpAddr>,
    /// The worker's IP as this seed observes it.
    pub observed: IpAddr,
    /// Assigned rank. Valid after [`SeedNode::collect_joins`] returns.
    pub rank: Rank,
}

/// A connection that sent a valid Hello.
struct Handshake {
    conn: quinn::Connection,
    send: quinn::SendStream,
    recv: quinn::RecvStream,
    requested: Option<Rank>,
    advertised: Option<IpAddr>,
}

impl SeedNode {
    /// Accept workers until `world_size` valid Hellos arrive, then assign ranks.
    ///
    /// Each connection runs its handshake in its own task, so a stalled or
    /// garbage connection cannot block the others. Returns the joiners in rank
    /// order and the channel that carries their later control messages.
    pub(super) async fn collect_joins(
        &self,
        deadline: Instant,
    ) -> Result<(Vec<Joiner>, JoinEvents)> {
        let expected = self.expected_world_size;
        let (hs_tx, mut hs_rx) = mpsc::channel::<Handshake>(MAX_HANDSHAKES);
        let (ev_tx, mut ev_rx) = control_channel();
        let gate = self.handshake_gate();
        let mut joiners: HashMap<u64, Joiner> = HashMap::new();
        let mut claimed: HashSet<Rank> = HashSet::new();
        let mut next_id = 0u64;

        while (joiners.len() as u32) < expected {
            tokio::select! {
                incoming = self.listener.endpoint().accept() => {
                    let incoming = incoming
                        .ok_or_else(|| NexarError::transport("seed endpoint closed"))?;
                    gate.spawn(incoming, hs_tx.clone());
                }
                Some(mut hs) = hs_rx.recv() => {
                    if let Some(rank) = hs.requested {
                        let detail = if rank >= expected {
                            Some(format!(
                                "out of range for world size {expected}; request a rank below {expected}"
                            ))
                        } else if claimed.contains(&rank) {
                            Some("another worker already holds it; give each process a distinct requested_rank".into())
                        } else {
                            None
                        };
                        if let Some(detail) = detail {
                            tracing::warn!(rank, "bootstrap: rejected a Hello: {detail}");
                            tokio::spawn(async move {
                                send_abort(&mut hs.send, AbortReason::RankConflict { rank, detail }).await;
                                drop(hs.conn);
                            });
                            continue;
                        }
                        claimed.insert(rank);
                    }
                    let id = next_id;
                    next_id += 1;
                    spawn_reader(id, hs.recv, ev_tx.clone());
                    let observed = hs.conn.remote_address().ip();
                    joiners.insert(id, Joiner {
                        id,
                        conn: hs.conn,
                        send: hs.send,
                        requested: hs.requested,
                        advertised: hs.advertised,
                        observed,
                        rank: 0,
                    });
                    tracing::info!(joined = joiners.len(), expected, "worker joined");
                }
                Some((id, msg)) = ev_rx.recv() => {
                    // A joiner that speaks or disconnects before its Welcome leaves.
                    if let Some(j) = joiners.remove(&id) {
                        if let Some(rank) = j.requested {
                            claimed.remove(&rank);
                        }
                        let why = match msg {
                            Ok(m) => format!("sent {m:?} before its Welcome"),
                            Err(e) => e.to_string(),
                        };
                        tracing::warn!(observed = %j.observed, "bootstrap: dropped a joiner: {why}");
                        j.conn.close(0u32.into(), b"left before welcome");
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    let joined = joiners.len() as u32;
                    let mut all: Vec<Joiner> = joiners.into_values().collect();
                    abort_all(&mut all, AbortReason::FormationTimeout { joined, expected }).await;
                    return Err(NexarError::ClusterFormationTimeout { joined, expected });
                }
            }
        }

        let mut ordered: Vec<Joiner> = joiners.into_values().collect();
        ordered.sort_by_key(|j| j.id);
        assign_ranks(&mut ordered, expected);
        ordered.sort_by_key(|j| j.rank);
        Ok((ordered, ev_rx))
    }
}

/// Rejects Hellos that arrive after the join phase. Stops when dropped.
pub(super) struct LateJoinGuard(tokio::task::JoinHandle<()>);

impl Drop for LateJoinGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl SeedNode {
    /// Answer every Hello that arrives after the join phase with `Abort`, so a
    /// surplus process learns the cluster is full instead of waiting.
    pub(super) fn reject_late_joins(&self) -> LateJoinGuard {
        let endpoint = self.listener.endpoint().clone();
        let world = self.expected_world_size;
        let gate = self.handshake_gate();
        LateJoinGuard(tokio::spawn(async move {
            let (tx, mut rx) = mpsc::channel::<Handshake>(MAX_HANDSHAKES);
            loop {
                tokio::select! {
                    incoming = endpoint.accept() => {
                        let Some(incoming) = incoming else { return };
                        gate.spawn(incoming, tx.clone());
                    }
                    Some(mut hs) = rx.recv() => {
                        let detail = format!("the cluster already has all {world} ranks; start at most world_size processes");
                        let reason = match hs.requested {
                            Some(rank) => AbortReason::RankConflict { rank, detail },
                            None => AbortReason::Malformed { detail },
                        };
                        tokio::spawn(async move {
                            send_abort(&mut hs.send, reason).await;
                            drop(hs.conn);
                        });
                    }
                }
            }
        }))
    }
}

/// Give each joiner without a requested rank the lowest free rank, in arrival order.
fn assign_ranks(joiners: &mut [Joiner], world_size: u32) {
    let taken: HashSet<Rank> = joiners.iter().filter_map(|j| j.requested).collect();
    let mut free = (0..world_size).filter(|r| !taken.contains(r));
    for j in joiners.iter_mut() {
        j.rank = match j.requested {
            Some(r) => r,
            None => free.next().unwrap_or(world_size),
        };
    }
}

/// What a task needs to run handshakes: the shared permit pool and the Hello rules.
struct HandshakeGate {
    permits: Arc<Semaphore>,
    token: Option<Vec<u8>>,
    timeout: Duration,
}

impl HandshakeGate {
    /// Run `incoming`'s handshake in its own task, or refuse it when
    /// [`MAX_HANDSHAKES`] handshakes already run.
    fn spawn(&self, incoming: quinn::Incoming, tx: mpsc::Sender<Handshake>) {
        let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() else {
            tracing::warn!("bootstrap: too many handshakes in flight, refused a connection");
            incoming.refuse();
            return;
        };
        tokio::spawn(handshake(
            incoming,
            self.token.clone(),
            self.timeout,
            tx,
            permit,
        ));
    }
}

impl SeedNode {
    fn handshake_gate(&self) -> HandshakeGate {
        HandshakeGate {
            permits: Arc::clone(&self.handshakes),
            token: self.cluster_token.clone(),
            timeout: self.hello_timeout,
        }
    }
}

/// Finish the QUIC handshake, read the Hello, and check it.
///
/// Sends a valid Hello to `tx`. Answers a bad Hello with `Abort`. Drops a
/// connection that fails or stalls, with a warning. Holds `_permit` until done.
async fn handshake(
    incoming: quinn::Incoming,
    token: Option<Vec<u8>>,
    timeout: Duration,
    tx: mpsc::Sender<Handshake>,
    _permit: OwnedSemaphorePermit,
) {
    let opened = tokio::time::timeout(timeout, async {
        let conn = incoming
            .await
            .map_err(|e| NexarError::transport_with_source("bootstrap handshake", e))?;
        let (send, mut recv) = conn
            .accept_bi()
            .await
            .map_err(|e| NexarError::transport_with_source("accept control stream", e))?;
        let hello = read_control(&mut recv).await;
        Ok::<_, NexarError>((conn, send, recv, hello))
    })
    .await;

    let (conn, mut send, recv, hello) = match opened {
        Ok(Ok(parts)) => parts,
        Ok(Err(e)) => {
            tracing::warn!("bootstrap: dropped a connection: {e}");
            return;
        }
        Err(_) => {
            tracing::warn!("bootstrap: dropped a connection that sent no Hello in time");
            return;
        }
    };
    match check_hello(hello, token.as_deref()) {
        Ok((requested, advertised)) => {
            let _ = tx
                .send(Handshake {
                    conn,
                    send,
                    recv,
                    requested,
                    advertised,
                })
                .await;
        }
        Err(reason) => {
            tracing::warn!(remote = %conn.remote_address(), "bootstrap: rejected a Hello: {reason:?}");
            send_abort(&mut send, reason).await;
        }
    }
}

/// Validate a Hello. Returns the requested rank and the advertised IP.
fn check_hello(
    msg: Result<NexarMessage>,
    token: Option<&[u8]>,
) -> std::result::Result<(Option<Rank>, Option<IpAddr>), AbortReason> {
    let msg = msg.map_err(|e| AbortReason::Malformed {
        detail: format!("unreadable Hello (protocol version {PROTOCOL_VERSION} expected): {e}"),
    })?;
    let NexarMessage::Hello {
        protocol_version,
        cluster_token,
        listen_addr,
        requested_rank,
        ..
    } = msg
    else {
        return Err(AbortReason::Malformed {
            detail: format!("expected Hello, got {msg:?}"),
        });
    };
    if protocol_version != PROTOCOL_VERSION {
        return Err(AbortReason::ProtocolMismatch {
            seed: PROTOCOL_VERSION,
        });
    }
    if let Some(expected) = token
        && !tokens_match(&cluster_token, expected)
    {
        return Err(AbortReason::TokenMismatch);
    }
    let advertised = if listen_addr.is_empty() {
        None
    } else {
        Some(
            listen_addr
                .parse::<IpAddr>()
                .map_err(|e| AbortReason::Malformed {
                    detail: format!("advertised address '{listen_addr}' is not an IP address: {e}"),
                })?,
        )
    };
    Ok((requested_rank, advertised))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hello(token: &[u8], rank: Option<Rank>, addr: &str) -> Result<NexarMessage> {
        Ok(NexarMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            capabilities: 0,
            cluster_token: token.to_vec(),
            listen_addr: addr.into(),
            requested_rank: rank,
        })
    }

    #[tokio::test]
    async fn test_handshake_flood_does_not_stop_a_join() {
        use crate::cluster::worker::{JoinRequest, WorkerNode};
        use crate::transport::tls::make_bootstrap_client_config;

        let mut seed = SeedNode::bind_local("127.0.0.1:0".parse().unwrap(), 2).unwrap();
        seed.handshakes = Arc::new(Semaphore::new(4));
        seed.hello_timeout = Duration::from_millis(500);
        let permits = Arc::clone(&seed.handshakes);
        let addr = seed.local_addr();
        let seed = Arc::new(seed);
        let collect = {
            let seed = Arc::clone(&seed);
            tokio::spawn(async move {
                seed.collect_joins(Instant::now() + Duration::from_secs(30))
                    .await
            })
        };

        // Twelve connections that never send a Hello: three times the permits.
        let mut flood = quinn::Endpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
        flood.set_default_client_config(make_bootstrap_client_config().unwrap());
        let held: Vec<_> = (0..12)
            .map(|_| {
                let connecting = flood.connect(addr, "localhost").unwrap();
                tokio::spawn(connecting)
            })
            .collect();
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            permits.available_permits(),
            0,
            "the flood fills the permits"
        );

        let req = JoinRequest {
            token: None,
            requested_rank: None,
            advertise_host: None,
            timeout: Duration::from_secs(30),
        };
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let req = JoinRequest { token: None, ..req };
                tokio::spawn(async move { WorkerNode::join(addr, &req).await })
            })
            .collect();

        let (joiners, _events) = tokio::time::timeout(Duration::from_secs(20), collect)
            .await
            .expect("formation waits out the flood")
            .unwrap()
            .unwrap();
        assert_eq!(joiners.len(), 2);
        for w in workers {
            w.abort();
        }
        for h in held {
            h.abort();
        }
    }

    #[test]
    fn test_check_hello_accepts_valid() {
        let (rank, ip) = check_hello(hello(b"t", Some(1), "10.0.0.2"), Some(b"t")).unwrap();
        assert_eq!(rank, Some(1));
        assert_eq!(ip, Some("10.0.0.2".parse().unwrap()));
        assert_eq!(
            check_hello(hello(b"", None, ""), None).unwrap(),
            (None, None)
        );
    }

    #[test]
    fn test_check_hello_rejects() {
        assert_eq!(
            check_hello(hello(b"x", None, ""), Some(b"t")),
            Err(AbortReason::TokenMismatch)
        );
        assert!(matches!(
            check_hello(hello(b"", None, "not-an-ip"), None),
            Err(AbortReason::Malformed { .. })
        ));
        let old = Ok(NexarMessage::Hello {
            protocol_version: 1,
            capabilities: 0,
            cluster_token: vec![],
            listen_addr: String::new(),
            requested_rank: None,
        });
        assert_eq!(
            check_hello(old, None),
            Err(AbortReason::ProtocolMismatch {
                seed: PROTOCOL_VERSION
            })
        );
        assert!(matches!(
            check_hello(Ok(NexarMessage::NodeLeft { rank: 0 }), None),
            Err(AbortReason::Malformed { .. })
        ));
    }
}
