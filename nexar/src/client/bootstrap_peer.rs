//! QUIC peer connections for one rank: dial higher ranks, accept lower ranks.
//!
//! Each accepted connection is tied to a rank by the connector's `PeerHello`,
//! never by comparing socket addresses. The same message carries the
//! connector's TCP sidecar address.

use crate::client::bootstrap_sidecar::Sidecar;
use crate::cluster::control::{read_control, send_abort, write_control};
use crate::error::{NexarError, Result};
use crate::protocol::{AbortReason, NexarMessage};
use crate::transport::{TcpBulkTransport, TransportListener};
use crate::types::Rank;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::Duration;

/// Time an inbound connection has to finish mTLS and send its `PeerHello`.
const PEER_HELLO_TIMEOUT: Duration = Duration::from_secs(10);

/// What one rank needs to connect to its neighbors.
pub(super) struct MeshPlan<'a> {
    pub rank: Rank,
    /// Mesh address of every rank.
    pub table: &'a HashMap<Rank, SocketAddr>,
    /// Neighbors with a lower rank. They dial this rank.
    pub lower: &'a [Rank],
    /// Neighbors with a higher rank. This rank dials them.
    pub higher: &'a [Rank],
    pub quic_client: quinn::ClientConfig,
    pub sidecar: &'a Sidecar,
}

/// Established links of one rank.
pub(super) struct Links {
    pub conns: HashMap<Rank, quinn::Connection>,
    pub sidecars: HashMap<Rank, TcpBulkTransport>,
}

/// Connect to every neighbor: QUIC both ways, then sidecars where enabled.
pub(super) async fn connect_links(
    plan: &MeshPlan<'_>,
    listener: &TransportListener,
) -> Result<Links> {
    let (offer, tls) = plan.sidecar.offer();
    let outbound = futures::future::try_join_all(
        plan.higher
            .iter()
            .map(|&peer| dial_peer(plan, listener.endpoint(), peer, &offer, tls)),
    );
    let inbound = async {
        let accepted = accept_peers(plan, listener).await?;
        let dials = accepted.iter().filter_map(|(peer, (_, addr))| {
            addr.map(|addr| async move {
                let t = plan.sidecar.connect_to(plan.rank, *peer, addr).await?;
                Ok::<_, NexarError>((*peer, t))
            })
        });
        let sidecars = futures::future::try_join_all(dials).await?;
        Ok::<_, NexarError>((accepted, sidecars))
    };
    let sidecars_in = async {
        if plan.sidecar.enabled() && !plan.higher.is_empty() {
            plan.sidecar.accept_from(plan.higher).await
        } else {
            Ok(HashMap::new())
        }
    };
    let (dialed, (accepted, sidecars_out), mut sidecars) =
        tokio::try_join!(outbound, inbound, sidecars_in)?;

    let mut conns: HashMap<Rank, quinn::Connection> = dialed.into_iter().collect();
    conns.extend(accepted.into_iter().map(|(r, (c, _))| (r, c)));
    sidecars.extend(sidecars_out);
    Ok(Links { conns, sidecars })
}

/// Dial the higher rank `peer`, send `PeerHello`, and check its `PeerHelloAck`.
async fn dial_peer(
    plan: &MeshPlan<'_>,
    endpoint: &quinn::Endpoint,
    peer: Rank,
    offer: &str,
    tls: bool,
) -> Result<(Rank, quinn::Connection)> {
    let addr = *plan
        .table
        .get(&peer)
        .ok_or(NexarError::UnknownPeer { rank: peer })?;
    let fail = |reason: String| NexarError::ConnectionFailed { rank: peer, reason };
    let conn = endpoint
        .connect_with(plan.quic_client.clone(), addr, &addr.ip().to_string())
        .map_err(|e| fail(format!("dial {addr}: {e}")))?
        .await
        .map_err(|e| fail(format!("mTLS handshake with {addr}: {e}")))?;
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| fail(format!("open handshake stream to {addr}: {e}")))?;
    let hello = NexarMessage::PeerHello {
        rank: plan.rank,
        sidecar_addr: offer.to_string(),
        sidecar_tls: tls,
    };
    write_control(&mut send, &hello)
        .await
        .map_err(|e| fail(format!("send PeerHello to {addr}: {e}")))?;
    let reply = read_control(&mut recv)
        .await
        .map_err(|e| fail(format!("read PeerHelloAck from {addr}: {e}")))?;
    match reply {
        NexarMessage::PeerHelloAck { rank } if rank == peer => {}
        NexarMessage::PeerHelloAck { rank } => {
            return Err(fail(format!("{addr} answered as rank {rank}")));
        }
        NexarMessage::Abort { reason } => return Err(reason.into_error()),
        other => return Err(fail(format!("expected PeerHelloAck, got {other:?}"))),
    }
    let _ = send.finish();
    Ok((peer, conn))
}

/// Accepted connection plus the sidecar address its connector offered.
type Accepted = HashMap<Rank, (quinn::Connection, Option<SocketAddr>)>;

/// Accept one connection from each lower rank.
///
/// A connection that fails, stalls, or names an unexpected rank is dropped with
/// a warning. A sidecar config mismatch is an error for both ranks.
async fn accept_peers(plan: &MeshPlan<'_>, listener: &TransportListener) -> Result<Accepted> {
    let expected: HashSet<Rank> = plan.lower.iter().copied().collect();
    let mut got: Accepted = HashMap::new();
    let mut pending = FuturesUnordered::new();
    while got.len() < expected.len() {
        tokio::select! {
            incoming = listener.endpoint().accept() => {
                let incoming = incoming.ok_or_else(|| NexarError::transport("mesh endpoint closed"))?;
                pending.push(read_peer_hello(incoming));
            }
            Some(hello) = pending.next() => {
                let (conn, mut send, rank, addr, tls) = match hello {
                    Ok(parts) => parts,
                    Err(e) => {
                        tracing::warn!("mesh: dropped an inbound connection: {e}");
                        continue;
                    }
                };
                if !expected.contains(&rank) || got.contains_key(&rank) {
                    tracing::warn!(rank, me = plan.rank, "mesh: dropped an inbound connection from an unexpected rank");
                    conn.close(0u32.into(), b"unexpected rank");
                    continue;
                }
                let sidecar = match plan.sidecar.check_offer(plan.rank, rank, &addr, tls) {
                    Ok(s) => s,
                    Err(e) => {
                        let reason = AbortReason::PeerFailed { rank: plan.rank, reason: e.to_string() };
                        send_abort(&mut send, reason).await;
                        return Err(e);
                    }
                };
                let ack = NexarMessage::PeerHelloAck { rank: plan.rank };
                write_control(&mut send, &ack).await.map_err(|e| NexarError::ConnectionFailed {
                    rank,
                    reason: format!("send PeerHelloAck: {e}"),
                })?;
                let _ = send.finish();
                got.insert(rank, (conn, sidecar));
            }
        }
    }
    Ok(got)
}

/// Finish mTLS on `incoming` and read the connector's `PeerHello`.
async fn read_peer_hello(
    incoming: quinn::Incoming,
) -> Result<(quinn::Connection, quinn::SendStream, Rank, String, bool)> {
    tokio::time::timeout(PEER_HELLO_TIMEOUT, async {
        let conn = incoming
            .await
            .map_err(|e| NexarError::transport_with_source("mesh mTLS handshake", e))?;
        let (send, mut recv) = conn
            .accept_bi()
            .await
            .map_err(|e| NexarError::transport_with_source("accept handshake stream", e))?;
        match read_control(&mut recv).await? {
            NexarMessage::PeerHello {
                rank,
                sidecar_addr,
                sidecar_tls,
            } => Ok((conn, send, rank, sidecar_addr, sidecar_tls)),
            other => Err(NexarError::DecodeFailed(format!(
                "expected PeerHello, got {other:?}"
            ))),
        }
    })
    .await
    .map_err(|_| NexarError::transport("no PeerHello in time"))?
}
