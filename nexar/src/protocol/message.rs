use crate::types::Rank;

/// Control messages exchanged between nexar nodes.
///
/// Tensor data does NOT flow through this enum. Bulk tensor transfers use
/// dedicated QUIC unidirectional streams with a minimal binary header
/// followed by raw bytes — avoiding rkyv overhead on large payloads.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, Debug, Clone, PartialEq)]
pub enum NexarMessage {
    /// Initial handshake from worker to seed.
    Hello {
        protocol_version: u16,
        capabilities: u64,
        /// Pre-shared cluster token for bootstrap authentication.
        /// Empty = no authentication required.
        cluster_token: Vec<u8>,
    },

    /// Seed's response with rank assignment, peer list, and mTLS credentials.
    Welcome {
        rank: Rank,
        world_size: u32,
        /// `(rank, socket_addr_string)` for each peer.
        peers: Vec<(Rank, String)>,
        /// DER-encoded cluster CA certificate (trust anchor for mesh mTLS).
        ca_cert: Vec<u8>,
        /// DER-encoded leaf certificate for this node, signed by the cluster CA.
        node_cert: Vec<u8>,
        /// DER-encoded private key for this node's leaf certificate.
        node_key: Vec<u8>,
    },

    /// Barrier request: all ranks must reach this epoch before proceeding.
    Barrier { epoch: u64, comm_id: u64 },

    /// Barrier acknowledgement from coordinator.
    BarrierAck { epoch: u64, comm_id: u64 },

    /// Periodic heartbeat for failure detection.
    Heartbeat { timestamp_ns: u64 },

    /// Notification that a new node has joined the cluster.
    NodeJoined { rank: Rank, addr: String },

    /// Notification that a node has left (or been detected as failed).
    NodeLeft { rank: Rank },

    /// Remote procedure call request.
    Rpc {
        req_id: u64,
        fn_id: u16,
        payload: Vec<u8>,
    },

    /// Response to an RPC request.
    RpcResponse { req_id: u64, payload: Vec<u8> },

    /// P2P data envelope for tagged point-to-point messaging.
    Data {
        tag: u32,
        src_rank: Rank,
        payload: Vec<u8>,
    },

    /// RDMA endpoint exchange for mesh formation (feature = "rdma").
    /// Carries the IB QP endpoint info needed for RDMA handshake.
    RdmaEndpoint {
        /// Local ID (LID) of the HCA port.
        lid: u16,
        /// Queue Pair Number.
        qpn: u32,
        /// Packet Sequence Number.
        psn: u32,
        /// Global ID (GID) as 16 bytes (IPv6 format).
        gid: Vec<u8>,
    },

    /// Communicator split request: carries (color, key) for group formation.
    SplitRequest { color: u32, key: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_roundtrip() {
        let msg = NexarMessage::Hello {
            protocol_version: 1,
            capabilities: 0xFF,
            cluster_token: vec![],
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&msg).unwrap();
        let deserialized: NexarMessage =
            rkyv::from_bytes::<NexarMessage, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_welcome_roundtrip() {
        let msg = NexarMessage::Welcome {
            rank: 3,
            world_size: 8,
            peers: vec![(0, "127.0.0.1:5000".into()), (1, "127.0.0.1:5001".into())],
            ca_cert: vec![1, 2, 3],
            node_cert: vec![4, 5, 6],
            node_key: vec![7, 8, 9],
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&msg).unwrap();
        let deserialized: NexarMessage =
            rkyv::from_bytes::<NexarMessage, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_all_variants_roundtrip() {
        let messages = vec![
            NexarMessage::Hello {
                protocol_version: 1,
                capabilities: 0,
                cluster_token: vec![],
            },
            NexarMessage::Welcome {
                rank: 0,
                world_size: 1,
                peers: vec![],
                ca_cert: vec![10],
                node_cert: vec![20],
                node_key: vec![30],
            },
            NexarMessage::Barrier {
                epoch: 42,
                comm_id: 0,
            },
            NexarMessage::BarrierAck {
                epoch: 42,
                comm_id: 0,
            },
            NexarMessage::Heartbeat {
                timestamp_ns: 123456789,
            },
            NexarMessage::NodeJoined {
                rank: 5,
                addr: "10.0.0.5:9000".into(),
            },
            NexarMessage::NodeLeft { rank: 2 },
            NexarMessage::Rpc {
                req_id: 1,
                fn_id: 100,
                payload: vec![1, 2, 3],
            },
            NexarMessage::RpcResponse {
                req_id: 1,
                payload: vec![4, 5, 6],
            },
            NexarMessage::Data {
                tag: 7,
                src_rank: 0,
                payload: vec![0xFF; 64],
            },
            NexarMessage::RdmaEndpoint {
                lid: 1,
                qpn: 42,
                psn: 100,
                gid: vec![0; 16],
            },
            NexarMessage::SplitRequest { color: 0, key: 1 },
        ];

        for msg in messages {
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&msg).unwrap();
            let back: NexarMessage =
                rkyv::from_bytes::<NexarMessage, rkyv::rancor::Error>(&bytes).unwrap();
            assert_eq!(msg, back, "roundtrip failed for {msg:?}");
        }
    }
}
