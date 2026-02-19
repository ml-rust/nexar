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
    },

    /// Seed's response with rank assignment and peer list.
    Welcome {
        rank: Rank,
        world_size: u32,
        /// `(rank, socket_addr_string)` for each peer.
        peers: Vec<(Rank, String)>,
    },

    /// Barrier request: all ranks must reach this epoch before proceeding.
    Barrier { epoch: u64 },

    /// Barrier acknowledgement from coordinator.
    BarrierAck { epoch: u64 },

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_roundtrip() {
        let msg = NexarMessage::Hello {
            protocol_version: 1,
            capabilities: 0xFF,
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
            },
            NexarMessage::Welcome {
                rank: 0,
                world_size: 1,
                peers: vec![],
            },
            NexarMessage::Barrier { epoch: 42 },
            NexarMessage::BarrierAck { epoch: 42 },
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
        ];

        for msg in messages {
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&msg).unwrap();
            let back: NexarMessage =
                rkyv::from_bytes::<NexarMessage, rkyv::rancor::Error>(&bytes).unwrap();
            assert_eq!(msg, back, "roundtrip failed for {msg:?}");
        }
    }
}
