use crate::error::NexarError;
use crate::types::{PROTOCOL_VERSION, Rank};

/// Why a bootstrap participant stopped cluster formation.
///
/// Carried by [`NexarMessage::Abort`](crate::protocol::NexarMessage::Abort).
/// The receiver turns it into a typed [`NexarError`] with [`AbortReason::into_error`].
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AbortReason {
    /// The joiner's cluster token does not match the seed's token.
    TokenMismatch,
    /// The joiner speaks another protocol version. `seed` is the seed's version.
    ProtocolMismatch { seed: u16 },
    /// The requested rank is out of range or another worker holds it.
    RankConflict { rank: Rank, detail: String },
    /// Formation did not finish before the seed's join timeout.
    FormationTimeout { joined: u32, expected: u32 },
    /// A rank failed while it built its part of the mesh.
    PeerFailed { rank: Rank, reason: String },
    /// The seed could not decode the joiner's handshake.
    Malformed { detail: String },
}

impl AbortReason {
    /// Convert the reason into the error the receiving side returns.
    pub fn into_error(self) -> NexarError {
        match self {
            AbortReason::TokenMismatch => NexarError::ClusterTokenMismatch,
            AbortReason::ProtocolMismatch { seed } => NexarError::ProtocolMismatch {
                local: PROTOCOL_VERSION,
                remote: seed,
            },
            AbortReason::RankConflict { rank, detail } => NexarError::RankConflict { rank, detail },
            AbortReason::FormationTimeout { joined, expected } => {
                NexarError::ClusterFormationTimeout { joined, expected }
            }
            AbortReason::PeerFailed { rank, reason } => {
                NexarError::ConnectionFailed { rank, reason }
            }
            AbortReason::Malformed { detail } => NexarError::BootstrapAborted { reason: detail },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abort_reason_maps_to_typed_errors() {
        assert!(matches!(
            AbortReason::TokenMismatch.into_error(),
            NexarError::ClusterTokenMismatch
        ));
        assert!(matches!(
            AbortReason::ProtocolMismatch { seed: 9 }.into_error(),
            NexarError::ProtocolMismatch { remote: 9, .. }
        ));
        assert!(matches!(
            AbortReason::RankConflict {
                rank: 2,
                detail: "taken".into()
            }
            .into_error(),
            NexarError::RankConflict { rank: 2, .. }
        ));
        assert!(matches!(
            AbortReason::FormationTimeout {
                joined: 2,
                expected: 3
            }
            .into_error(),
            NexarError::ClusterFormationTimeout {
                joined: 2,
                expected: 3
            }
        ));
        assert!(matches!(
            AbortReason::PeerFailed {
                rank: 1,
                reason: "x".into()
            }
            .into_error(),
            NexarError::ConnectionFailed { rank: 1, .. }
        ));
        assert!(matches!(
            AbortReason::Malformed { detail: "x".into() }.into_error(),
            NexarError::BootstrapAborted { .. }
        ));
    }
}
