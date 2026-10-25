//! Formation steps after the join phase: Welcome, mesh address exchange, ready barrier.

use crate::cluster::SeedNode;
use crate::cluster::control::{write_control, write_final};
use crate::cluster::seed::{Stage, abort_all};
use crate::cluster::seed_join::{JoinEvents, Joiner};
use crate::error::{NexarError, Result};
use crate::protocol::{AbortReason, NexarMessage};
use crate::transport::tls::ClusterCa;
use crate::types::Rank;
use std::collections::HashMap;
use std::net::SocketAddr;
use tokio::time::Instant;

impl SeedNode {
    /// Send each joiner its rank and a leaf certificate signed by `ca`.
    ///
    /// The certificate names `localhost`, the advertised IP, and the observed IP,
    /// so peers can verify it under any of the addresses they dial.
    pub(super) async fn send_welcomes(&self, ca: &ClusterCa, joiners: &mut [Joiner]) -> Result<()> {
        let ca_cert = ca.cert_der().to_vec();
        for j in joiners.iter_mut() {
            let mut sans = vec!["localhost".to_string(), j.observed.to_string()];
            if let Some(ip) = j.advertised {
                sans.push(ip.to_string());
            }
            sans.sort();
            sans.dedup();
            let san_refs: Vec<&str> = sans.iter().map(String::as_str).collect();
            let (cert, key) = ca.issue_cert_multi(&san_refs)?;
            let welcome = NexarMessage::Welcome {
                rank: j.rank,
                world_size: self.expected_world_size,
                peers: Vec::new(),
                ca_cert: ca_cert.clone(),
                node_cert: cert.to_vec(),
                node_key: key.secret_der().to_vec(),
                observed_addr: j.observed.to_string(),
            };
            write_control(&mut j.send, &welcome).await.map_err(|e| {
                NexarError::ConnectionFailed {
                    rank: j.rank,
                    reason: format!("send Welcome: {e}"),
                }
            })?;
        }
        Ok(())
    }

    /// Wait for one `stage` message from every joiner.
    ///
    /// For [`Stage::MeshListen`] the result maps each rank to its mesh address.
    /// An `Abort`, an unexpected message, a lost connection, or the deadline
    /// aborts every joiner and returns the matching error.
    pub(super) async fn collect_stage(
        &self,
        joiners: &mut [Joiner],
        events: &mut JoinEvents,
        deadline: Instant,
        stage: Stage,
    ) -> Result<HashMap<Rank, String>> {
        let ranks: HashMap<u64, Rank> = joiners.iter().map(|j| (j.id, j.rank)).collect();
        let mut got: HashMap<Rank, String> = HashMap::new();
        while got.len() < joiners.len() {
            let event = tokio::select! {
                ev = events.recv() => ev,
                _ = tokio::time::sleep_until(deadline) => {
                    let joined = got.len() as u32;
                    let expected = self.expected_world_size;
                    tracing::warn!(joined, expected, "formation timed out at {}", stage.name());
                    abort_all(joiners, AbortReason::FormationTimeout { joined, expected }).await;
                    return Err(NexarError::ClusterFormationTimeout { joined, expected });
                }
            };
            let Some((id, msg)) = event else {
                return Err(NexarError::transport("seed event channel closed"));
            };
            let Some(&rank) = ranks.get(&id) else {
                continue;
            };
            let failure = match (stage, msg) {
                (Stage::MeshListen, Ok(NexarMessage::MeshListen { rank: r, addr }))
                    if r == rank =>
                {
                    match addr.parse::<SocketAddr>() {
                        Ok(_) => {
                            got.insert(rank, addr);
                            continue;
                        }
                        Err(e) => AbortReason::PeerFailed {
                            rank,
                            reason: format!("advertised mesh address '{addr}' is invalid: {e}"),
                        },
                    }
                }
                (Stage::Ready, Ok(NexarMessage::Barrier { .. })) => {
                    got.insert(rank, String::new());
                    continue;
                }
                (_, Ok(NexarMessage::Abort { reason })) => reason,
                (_, Ok(other)) => AbortReason::PeerFailed {
                    rank,
                    reason: format!("sent {other:?} during {}", stage.name()),
                },
                (_, Err(e)) => AbortReason::PeerFailed {
                    rank,
                    reason: format!("lost its seed connection during {}: {e}", stage.name()),
                },
            };
            tracing::warn!("aborting cluster formation: {failure:?}");
            abort_all(joiners, failure.clone()).await;
            return Err(failure.into_error());
        }
        Ok(got)
    }

    /// Send `msg` to every joiner concurrently. `last` finishes each stream
    /// and waits until the worker acknowledges the bytes.
    pub(super) async fn broadcast(
        &self,
        joiners: &mut [Joiner],
        msg: &NexarMessage,
        last: bool,
    ) -> Result<()> {
        let sends = joiners.iter_mut().map(|j| async move {
            let sent = if last {
                write_final(&mut j.send, msg).await
            } else {
                write_control(&mut j.send, msg).await
            };
            sent.map_err(|e| NexarError::ConnectionFailed {
                rank: j.rank,
                reason: format!("send to rank {}: {e}", j.rank),
            })
        });
        let results = futures::future::join_all(sends).await;
        if let Some(err) = results.into_iter().find_map(|r| r.err()) {
            let reason = match &err {
                NexarError::ConnectionFailed { rank, reason } => AbortReason::PeerFailed {
                    rank: *rank,
                    reason: reason.clone(),
                },
                other => AbortReason::Malformed {
                    detail: other.to_string(),
                },
            };
            if !last {
                abort_all(joiners, reason).await;
            }
            return Err(err);
        }
        Ok(())
    }
}
