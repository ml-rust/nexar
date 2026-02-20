use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::protocol::codec::{decode_message, encode_message};
use crate::protocol::header::HEADER_SIZE;
use crate::transport::tls::make_bootstrap_client_config;
use crate::types::{PROTOCOL_VERSION, Priority, Rank};
use std::net::SocketAddr;

/// Result of connecting to the seed node and completing the handshake.
pub struct WorkerNode {
    pub rank: Rank,
    pub world_size: u32,
    pub peers: Vec<(Rank, String)>,
    pub seed_conn: quinn::Connection,
    /// DER-encoded cluster CA certificate (trust anchor for mesh mTLS).
    pub ca_cert: Vec<u8>,
    /// DER-encoded leaf certificate for this node, signed by the cluster CA.
    pub node_cert: Vec<u8>,
    /// DER-encoded private key for this node's leaf certificate.
    pub node_key: Vec<u8>,
}

impl WorkerNode {
    /// Connect to the seed node, complete the handshake, and receive rank assignment.
    ///
    /// The initial connection uses insecure TLS (bootstrap). After receiving
    /// the Welcome message with CA-signed credentials, all subsequent mesh
    /// connections use mutual TLS.
    pub async fn connect(seed_addr: SocketAddr) -> Result<Self> {
        let client_config = make_bootstrap_client_config()?;

        // Bind a local UDP socket.
        let bind_addr: SocketAddr = "0.0.0.0:0".parse().expect("hardcoded socket addr");
        let mut endpoint = quinn::Endpoint::client(bind_addr)
            .map_err(|e| NexarError::transport_with_source("bind client", e))?;
        endpoint.set_default_client_config(client_config);

        let conn = endpoint
            .connect(seed_addr, "localhost")
            .map_err(|e| NexarError::transport_with_source("connect to seed", e))?
            .await
            .map_err(|e| NexarError::ConnectionFailed {
                rank: 0,
                reason: format!("QUIC handshake: {e}"),
            })?;

        // Open the first bidirectional stream and send Hello.
        let (mut send, mut recv) = conn
            .open_bi()
            .await
            .map_err(|e| NexarError::transport_with_source("open bi to seed", e))?;

        let cluster_token = std::env::var("NEXAR_CLUSTER_TOKEN")
            .map(|t| t.into_bytes())
            .unwrap_or_default();
        let hello = NexarMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            capabilities: 0,
            cluster_token,
        };
        let buf = encode_message(&hello, Priority::Critical)?;
        send.write_all(&buf)
            .await
            .map_err(|e| NexarError::transport_with_source("send hello", e))?;

        // Read Welcome response.
        let mut header_buf = [0u8; HEADER_SIZE];
        recv.read_exact(&mut header_buf)
            .await
            .map_err(|e| NexarError::transport_with_source("read welcome header", e))?;
        let payload_len =
            u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]])
                as usize;
        let mut payload = vec![0u8; payload_len];
        recv.read_exact(&mut payload)
            .await
            .map_err(|e| NexarError::transport_with_source("read welcome payload", e))?;

        let mut full_buf = Vec::with_capacity(HEADER_SIZE + payload_len);
        full_buf.extend_from_slice(&header_buf);
        full_buf.extend_from_slice(&payload);
        let (_, msg) = decode_message(&full_buf)?;

        match msg {
            NexarMessage::Welcome {
                rank,
                world_size,
                peers,
                ca_cert,
                node_cert,
                node_key,
            } => Ok(WorkerNode {
                rank,
                world_size,
                peers,
                seed_conn: conn,
                ca_cert,
                node_cert,
                node_key,
            }),
            other => Err(NexarError::DecodeFailed(format!(
                "expected Welcome, got {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::SeedNode;

    #[tokio::test]
    async fn test_single_worker_join() {
        let seed_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let seed = SeedNode::bind_local(seed_addr, 1).unwrap();
        let seed_addr = seed.local_addr();

        let (seed_result, worker_result) =
            tokio::join!(seed.form_cluster(), WorkerNode::connect(seed_addr),);

        let (_map, conns) = seed_result.unwrap();
        assert_eq!(conns.len(), 1);

        let worker = worker_result.unwrap();
        assert_eq!(worker.rank, 0);
        assert_eq!(worker.world_size, 1);
        assert_eq!(worker.peers.len(), 1);
        // Verify mTLS credentials were received.
        assert!(!worker.ca_cert.is_empty());
        assert!(!worker.node_cert.is_empty());
        assert!(!worker.node_key.is_empty());
    }

    #[tokio::test]
    async fn test_two_workers_join() {
        let seed_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let seed = SeedNode::bind_local(seed_addr, 2).unwrap();
        let seed_addr = seed.local_addr();

        let seed_handle = tokio::spawn(async move { seed.form_cluster().await });

        let w1 = tokio::spawn(WorkerNode::connect(seed_addr));
        let w2 = tokio::spawn(WorkerNode::connect(seed_addr));

        let (seed_result, w1_result, w2_result) = tokio::join!(seed_handle, w1, w2);

        let (_map, conns) = seed_result.unwrap().unwrap();
        assert_eq!(conns.len(), 2);

        let w1 = w1_result.unwrap().unwrap();
        let w2 = w2_result.unwrap().unwrap();

        // Ranks should be unique.
        assert_ne!(w1.rank, w2.rank);
        assert_eq!(w1.world_size, 2);
        assert_eq!(w2.world_size, 2);
        assert_eq!(w1.peers.len(), 2);
        assert_eq!(w2.peers.len(), 2);
        // Both workers should have the same CA cert.
        assert_eq!(w1.ca_cert, w2.ca_cert);
        // But different node certs (unique per node).
        assert_ne!(w1.node_cert, w2.node_cert);
    }

    #[tokio::test]
    async fn test_four_workers_join() {
        let seed_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let seed = SeedNode::bind_local(seed_addr, 4).unwrap();
        let seed_addr = seed.local_addr();

        let seed_handle = tokio::spawn(async move { seed.form_cluster().await });

        let mut handles = Vec::new();
        for _ in 0..4 {
            handles.push(tokio::spawn(WorkerNode::connect(seed_addr)));
        }

        let (_map, conns) = seed_handle.await.unwrap().unwrap();
        assert_eq!(conns.len(), 4);

        let mut ranks = Vec::new();
        for h in handles {
            let w = h.await.unwrap().unwrap();
            assert_eq!(w.world_size, 4);
            assert_eq!(w.peers.len(), 4);
            ranks.push(w.rank);
        }
        ranks.sort();
        assert_eq!(ranks, vec![0, 1, 2, 3]);
    }
}
