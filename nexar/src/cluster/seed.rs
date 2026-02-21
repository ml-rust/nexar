use crate::cluster::topology::ClusterMap;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::{decode_message, encode_message};
use crate::protocol::header::HEADER_SIZE;
use crate::transport::TransportListener;
use crate::transport::tls::ClusterCa;
use crate::types::{PROTOCOL_VERSION, Priority, Rank};
use quinn::RecvStream;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

/// Read and decode a single framed message from a QUIC recv stream.
async fn read_framed_message(recv: &mut RecvStream) -> Result<NexarMessage> {
    let mut header_buf = [0u8; HEADER_SIZE];
    recv.read_exact(&mut header_buf)
        .await
        .map_err(|e| NexarError::transport_with_source("read message header", e))?;
    let payload_len =
        u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]) as usize;
    let mut payload = vec![0u8; payload_len];
    recv.read_exact(&mut payload)
        .await
        .map_err(|e| NexarError::transport_with_source("read message payload", e))?;

    let mut full_buf = Vec::with_capacity(HEADER_SIZE + payload_len);
    full_buf.extend_from_slice(&header_buf);
    full_buf.extend_from_slice(&payload);
    let (_, msg) = decode_message(&full_buf)?;
    Ok(msg)
}

/// Result of cluster formation, including the CA for elastic scaling.
pub struct FormClusterResult {
    pub map: ClusterMap,
    pub connections: Vec<quinn::Connection>,
    pub ca: ClusterCa,
    pub next_rank: Rank,
}

/// The seed node orchestrates cluster formation.
///
/// It listens for incoming workers, assigns ranks, and distributes the
/// peer table once enough workers have joined. During formation, it
/// generates an ephemeral CA and issues per-node certificates for
/// mutual TLS on the P2P mesh.
pub struct SeedNode {
    listener: TransportListener,
    expected_world_size: u32,
    formation_timeout: Duration,
    /// Cluster token for authenticating workers during bootstrap.
    /// Always `Some` in production (auto-generated if env var is unset).
    /// `None` only for loopback testing via `bind_local`.
    cluster_token: Option<Vec<u8>>,
}

impl SeedNode {
    /// Create a seed node bound to the given address.
    ///
    /// If the `NEXAR_CLUSTER_TOKEN` environment variable is set, workers must
    /// present the same token in their Hello message to be accepted. If the
    /// variable is not set, a random token is auto-generated and logged so
    /// that workers can be configured with it.
    pub fn bind(addr: SocketAddr, expected_world_size: u32) -> Result<Self> {
        let listener = TransportListener::bind(addr)?;
        let cluster_token = match std::env::var("NEXAR_CLUSTER_TOKEN") {
            Ok(t) if !t.is_empty() => {
                tracing::info!("using cluster token from NEXAR_CLUSTER_TOKEN env var");
                t.into_bytes()
            }
            _ => {
                // Generate a random 32-byte token using rcgen's CSPRNG-backed
                // key generation. We take the first 32 bytes of the serialized
                // ephemeral key as a cryptographically random token.
                let ephemeral_key = rcgen::KeyPair::generate()
                    .map_err(|e| NexarError::Tls(format!("generate cluster token: {e}")))?;
                let der = ephemeral_key.serialize_der();
                let token: Vec<u8> = der.into_iter().take(32).collect();
                let hex_token: String = token.iter().map(|b| format!("{b:02x}")).collect();
                tracing::warn!(
                    "NEXAR_CLUSTER_TOKEN not set — auto-generated token: {hex_token}. \
                     Set NEXAR_CLUSTER_TOKEN={hex_token} on workers to authenticate."
                );
                token
            }
        };
        Ok(Self {
            listener,
            expected_world_size,
            formation_timeout: Duration::from_secs(60),
            cluster_token: Some(cluster_token),
        })
    }

    /// Create a seed node for loopback testing (no token enforcement).
    ///
    /// Skips cluster token generation/validation since all connections
    /// are on localhost within the same process.
    pub fn bind_local(addr: SocketAddr, expected_world_size: u32) -> Result<Self> {
        let listener = TransportListener::bind(addr)?;
        Ok(Self {
            listener,
            expected_world_size,
            formation_timeout: Duration::from_secs(60),
            cluster_token: None,
        })
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

    /// Wait for all expected workers to join and distribute the peer table.
    ///
    /// Generates an ephemeral cluster CA and issues a CA-signed leaf
    /// certificate to each worker via the Welcome message. Workers use
    /// these credentials for mutual TLS on the P2P mesh.
    ///
    /// Returns the cluster map and the list of QUIC connections (one per worker),
    /// in rank order (rank 0 = seed itself conceptually, ranks 1..N = workers).
    ///
    /// The seed does NOT assign itself a rank in the communicator — it is purely
    /// a coordination node. Workers get ranks 0..world_size-1.
    pub async fn form_cluster(&self) -> Result<FormClusterResult> {
        // Generate ephemeral CA for this cluster's lifetime.
        let ca = ClusterCa::generate()?;
        let ca_cert_der = ca.cert_der();

        let mut map = ClusterMap::new();
        let mut connections: Vec<(
            Rank,
            quinn::Connection,
            quinn::SendStream,
            quinn::RecvStream,
        )> = Vec::new();
        let mut next_rank: Rank = 0;

        let deadline = tokio::time::Instant::now() + self.formation_timeout;

        while next_rank < self.expected_world_size {
            let conn = tokio::select! {
                result = self.listener.accept() => result?,
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(NexarError::ClusterFormationTimeout {
                        joined: next_rank,
                        expected: self.expected_world_size,
                    });
                }
            };

            // Accept the first bidirectional stream (control channel).
            let (send, mut recv) = conn
                .accept_bi()
                .await
                .map_err(|e| NexarError::transport_with_source("accept bi from new worker", e))?;

            // Read Hello message.
            let msg = read_framed_message(&mut recv).await?;

            let worker_listen_addr = match msg {
                NexarMessage::Hello {
                    protocol_version,
                    cluster_token: token,
                    listen_addr,
                    ..
                } => {
                    if protocol_version != PROTOCOL_VERSION {
                        return Err(NexarError::ProtocolMismatch {
                            local: PROTOCOL_VERSION,
                            remote: protocol_version,
                        });
                    }
                    if let Some(expected) = &self.cluster_token
                        && token.as_slice() != expected.as_slice()
                    {
                        return Err(NexarError::ClusterTokenMismatch);
                    }
                    listen_addr
                }
                other => {
                    return Err(NexarError::DecodeFailed(format!(
                        "expected Hello, got {other:?}"
                    )));
                }
            };

            let rank = next_rank;
            let addr = if worker_listen_addr.is_empty() {
                conn.remote_address().to_string()
            } else {
                worker_listen_addr
            };
            map.add_peer(rank, addr);
            connections.push((rank, conn, send, recv));
            next_rank += 1;

            tracing::info!(
                "worker joined: rank={rank}, total={next_rank}/{}",
                self.expected_world_size
            );
        }

        // Build peer table.
        let peers = map.alive_peers();

        // Send Welcome to each worker with their assigned rank and mTLS credentials.
        let mut conns = Vec::with_capacity(connections.len());
        for (rank, conn, mut send, _recv) in connections {
            // Issue a unique leaf certificate for this worker.
            let (node_cert, node_key) = ca.issue_cert("localhost")?;

            let welcome = NexarMessage::Welcome {
                rank,
                world_size: self.expected_world_size,
                peers: peers.clone(),
                ca_cert: ca_cert_der.to_vec(),
                node_cert: node_cert.to_vec(),
                node_key: node_key.secret_der().to_vec(),
            };
            let buf = encode_message(&welcome, Priority::Critical)?;
            send.write_all(&buf).await.map_err(|e| {
                NexarError::transport_with_source(format!("send welcome to rank {rank}"), e)
            })?;
            conns.push(conn);
        }

        Ok(FormClusterResult {
            map,
            connections: conns,
            ca,
            next_rank,
        })
    }

    /// Background loop accepting new Hello messages from joining nodes after formation.
    ///
    /// Assigns new ranks, issues certs, sends Welcome, and pushes `PendingJoin`
    /// entries for the `ElasticManager` to process at the next checkpoint.
    pub async fn accept_elastic(
        &self,
        ca: Arc<ClusterCa>,
        next_rank: Arc<std::sync::atomic::AtomicU32>,
        max_world_size: u32,
        pending_joins: Arc<std::sync::Mutex<Vec<PendingJoin>>>,
        cluster_map: Arc<std::sync::Mutex<ClusterMap>>,
    ) -> Result<()> {
        let ca_cert_der = ca.cert_der();

        loop {
            let conn = match self.listener.accept().await {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("elastic accept error: {e}");
                    continue;
                }
            };

            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!("elastic accept_bi error: {e}");
                    continue;
                }
            };

            // Read Hello.
            let msg = match read_framed_message(&mut recv).await {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!("elastic join: failed to read hello: {e}");
                    continue;
                }
            };

            let listen_addr = match msg {
                NexarMessage::Hello {
                    protocol_version,
                    cluster_token: token,
                    listen_addr,
                    ..
                } => {
                    if protocol_version != PROTOCOL_VERSION {
                        tracing::warn!("elastic join: protocol mismatch");
                        continue;
                    }
                    if let Some(expected) = &self.cluster_token
                        && token.as_slice() != expected.as_slice()
                    {
                        tracing::warn!("elastic join: token mismatch");
                        continue;
                    }
                    listen_addr
                }
                _ => continue,
            };

            let rank = next_rank.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            if max_world_size > 0 && rank >= max_world_size {
                next_rank.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!("elastic join rejected: max world size {max_world_size} reached");
                continue;
            }

            // Resolve the address: use listen_addr if provided, else fall back to
            // the QUIC connection's remote address.
            let resolved_addr = if listen_addr.is_empty() {
                conn.remote_address().to_string()
            } else {
                listen_addr
            };

            // Issue cert for new node.
            let (node_cert, node_key) = match ca.issue_cert("localhost") {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!("elastic join: cert issue failed: {e}");
                    continue;
                }
            };

            // Get current peers from cluster map.
            let peers = {
                let map = cluster_map.lock().unwrap_or_else(|p| p.into_inner());
                map.alive_peers()
            };

            // Add to cluster map.
            {
                let mut map = cluster_map.lock().unwrap_or_else(|p| p.into_inner());
                map.add_peer(rank, resolved_addr.clone());
            }

            let welcome = NexarMessage::Welcome {
                rank,
                world_size: rank + 1, // Temporary; real world size set at checkpoint
                peers,
                ca_cert: ca_cert_der.as_ref().to_vec(),
                node_cert: node_cert.as_ref().to_vec(),
                node_key: node_key.secret_der().to_vec(),
            };

            let buf = match encode_message(&welcome, Priority::Critical) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("elastic join: encode welcome failed: {e}");
                    continue;
                }
            };
            if send.write_all(&buf).await.is_err() {
                continue;
            }

            pending_joins
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(PendingJoin {
                    rank,
                    listen_addr: resolved_addr,
                });

            tracing::info!("elastic join: rank {rank} queued for next checkpoint");
        }
    }
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
}
