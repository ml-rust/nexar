use crate::cluster::topology::ClusterMap;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::{decode_message, encode_message};
use crate::protocol::header::HEADER_SIZE;
use crate::transport::TransportListener;
use crate::transport::tls::ClusterCa;
use crate::types::{PROTOCOL_VERSION, Priority, Rank};
use std::net::SocketAddr;
use std::time::Duration;

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
}

impl SeedNode {
    /// Create a seed node bound to the given address.
    pub fn bind(addr: SocketAddr, expected_world_size: u32) -> Result<Self> {
        let listener = TransportListener::bind(addr)?;
        Ok(Self {
            listener,
            expected_world_size,
            formation_timeout: Duration::from_secs(60),
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
    pub async fn form_cluster(&self) -> Result<(ClusterMap, Vec<quinn::Connection>)> {
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
                .map_err(|e| NexarError::Transport(format!("accept bi from new worker: {e}")))?;

            // Read Hello message.
            let mut header_buf = [0u8; HEADER_SIZE];
            recv.read_exact(&mut header_buf)
                .await
                .map_err(|e| NexarError::Transport(format!("read hello header: {e}")))?;
            let payload_len =
                u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]])
                    as usize;
            let mut payload = vec![0u8; payload_len];
            recv.read_exact(&mut payload)
                .await
                .map_err(|e| NexarError::Transport(format!("read hello payload: {e}")))?;

            let mut full_buf = Vec::with_capacity(HEADER_SIZE + payload_len);
            full_buf.extend_from_slice(&header_buf);
            full_buf.extend_from_slice(&payload);
            let (_, msg) = decode_message(&full_buf)?;

            match msg {
                NexarMessage::Hello {
                    protocol_version, ..
                } => {
                    if protocol_version != PROTOCOL_VERSION {
                        return Err(NexarError::ProtocolMismatch {
                            local: PROTOCOL_VERSION,
                            remote: protocol_version,
                        });
                    }
                }
                other => {
                    return Err(NexarError::DecodeFailed(format!(
                        "expected Hello, got {other:?}"
                    )));
                }
            }

            let rank = next_rank;
            let addr = conn.remote_address().to_string();
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
            send.write_all(&buf)
                .await
                .map_err(|e| NexarError::Transport(format!("send welcome to rank {rank}: {e}")))?;
            conns.push(conn);
        }

        Ok((map, conns))
    }
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
