use crate::types::Rank;

pub type Result<T> = std::result::Result<T, NexarError>;

#[derive(Debug, thiserror::Error)]
pub enum NexarError {
    #[error("connection to rank {rank} failed: {reason}")]
    ConnectionFailed { rank: Rank, reason: String },

    #[error("peer {rank} disconnected unexpectedly")]
    PeerDisconnected { rank: Rank },

    #[error("rank {rank} not found in cluster")]
    UnknownPeer { rank: Rank },

    #[error("protocol version mismatch: local={local}, remote={remote}")]
    ProtocolMismatch { local: u16, remote: u16 },

    #[error("message decode failed: {0}")]
    DecodeFailed(String),

    #[error("message encode failed: {0}")]
    EncodeFailed(String),

    #[error("barrier timed out after {timeout_ms}ms (epoch {epoch})")]
    BarrierTimeout { epoch: u64, timeout_ms: u64 },

    #[error("cluster formation timed out: {joined}/{expected} nodes joined")]
    ClusterFormationTimeout { joined: u32, expected: u32 },

    #[error("unsupported data type: {dtype:?} for operation {op}")]
    UnsupportedDType {
        dtype: crate::types::DataType,
        op: &'static str,
    },

    #[error("buffer size mismatch: expected {expected} bytes, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    #[error("invalid rank {rank}: world size is {world_size}")]
    InvalidRank { rank: Rank, world_size: u32 },

    #[error("QUIC transport error: {message}")]
    Transport {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("TLS configuration error: {0}")]
    Tls(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("RPC handler not registered for fn_id={fn_id}")]
    RpcNotRegistered { fn_id: u16 },

    #[error("RPC call to rank {rank} failed: {reason}")]
    RpcFailed { rank: Rank, reason: String },

    #[error("device adapter error: {message}")]
    DeviceError {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("{operation} failed at rank {rank}: {reason}")]
    CollectiveFailed {
        operation: &'static str,
        rank: Rank,
        reason: String,
    },

    #[error("operation cancelled")]
    Cancelled,

    #[error("cluster token mismatch: bootstrap authentication failed")]
    ClusterTokenMismatch,

    #[error(
        "count {count} is not evenly divisible by world size {world_size} (required by {operation})"
    )]
    IndivisibleCount {
        count: usize,
        world_size: usize,
        operation: &'static str,
    },

    #[error("internal lock poisoned: {0}")]
    LockPoisoned(&'static str),

    #[error("recovery failed (dead ranks: {dead_ranks:?}): {message}")]
    Recovery {
        dead_ranks: Vec<Rank>,
        message: String,
    },
}

impl NexarError {
    /// Create a `Transport` error with just a message.
    pub fn transport(msg: impl Into<String>) -> Self {
        Self::Transport {
            message: msg.into(),
            source: None,
        }
    }

    /// Create a `Transport` error with a message and a source error.
    pub fn transport_with_source(
        msg: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Transport {
            message: msg.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create a `DeviceError` with just a message.
    pub fn device(msg: impl Into<String>) -> Self {
        Self::DeviceError {
            message: msg.into(),
            source: None,
        }
    }

    /// Create a `DeviceError` with a message and a source error.
    pub fn device_with_source(
        msg: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::DeviceError {
            message: msg.into(),
            source: Some(Box::new(source)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = NexarError::ConnectionFailed {
            rank: 3,
            reason: "timeout".into(),
        };
        assert_eq!(e.to_string(), "connection to rank 3 failed: timeout");
    }

    #[test]
    fn test_barrier_timeout_display() {
        let e = NexarError::BarrierTimeout {
            epoch: 42,
            timeout_ms: 5000,
        };
        assert_eq!(e.to_string(), "barrier timed out after 5000ms (epoch 42)");
    }

    #[test]
    fn test_collective_failed_display() {
        let e = NexarError::CollectiveFailed {
            operation: "allreduce",
            rank: 3,
            reason: "connection reset".into(),
        };
        assert_eq!(
            e.to_string(),
            "allreduce failed at rank 3: connection reset"
        );
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::AddrInUse, "port busy");
        let nexar_err: NexarError = io_err.into();
        assert!(nexar_err.to_string().contains("port busy"));
    }

    #[test]
    fn test_all_variants_display() {
        // Ensure all variants produce non-empty display strings
        let errors: Vec<NexarError> = vec![
            NexarError::ConnectionFailed {
                rank: 0,
                reason: "x".into(),
            },
            NexarError::PeerDisconnected { rank: 1 },
            NexarError::UnknownPeer { rank: 2 },
            NexarError::ProtocolMismatch {
                local: 1,
                remote: 2,
            },
            NexarError::DecodeFailed("bad".into()),
            NexarError::EncodeFailed("bad".into()),
            NexarError::BarrierTimeout {
                epoch: 0,
                timeout_ms: 100,
            },
            NexarError::ClusterFormationTimeout {
                joined: 2,
                expected: 4,
            },
            NexarError::UnsupportedDType {
                dtype: crate::types::DataType::F32,
                op: "reduce",
            },
            NexarError::BufferSizeMismatch {
                expected: 100,
                actual: 50,
            },
            NexarError::InvalidRank {
                rank: 5,
                world_size: 4,
            },
            NexarError::transport("conn reset"),
            NexarError::Tls("bad cert".into()),
            NexarError::RpcNotRegistered { fn_id: 42 },
            NexarError::RpcFailed {
                rank: 1,
                reason: "timeout".into(),
            },
            NexarError::device("oom"),
            NexarError::CollectiveFailed {
                operation: "allreduce",
                rank: 2,
                reason: "peer disconnected".into(),
            },
            NexarError::Cancelled,
            NexarError::IndivisibleCount {
                count: 7,
                world_size: 4,
                operation: "rs_ag_allreduce",
            },
            NexarError::LockPoisoned("extensions"),
            NexarError::Recovery {
                dead_ranks: vec![2],
                message: "test".into(),
            },
        ];
        for e in &errors {
            assert!(!e.to_string().is_empty(), "empty display for {e:?}");
        }
    }
}
