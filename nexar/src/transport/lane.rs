//! Receive keys: which receiver a message from one peer belongs to.
//!
//! A receiver waits on one key and takes only messages with that key.
//! Messages with other keys stay queued for their own receivers.

use crate::protocol::NexarMessage;

/// Stream type byte of a framed message (`NexarMessage`, not sequenced).
pub(crate) const STREAM_TAG_FRAMED: u8 = 0x01;
/// Stream type byte of a sequenced byte message: [`SeqHeader`] then the payload.
pub(crate) const STREAM_TAG_SEQ: u8 = 0x05;
/// Stream type byte of a skip notice: `[count: u32 LE][seq: u64 LE; count]`.
pub(crate) const STREAM_TAG_SKIP: u8 = 0x06;

/// Most sequence numbers one skip notice carries.
pub(crate) const MAX_SKIPS: u32 = 1024;

/// The byte lane of a sequenced message, scoped to one communicator.
///
/// The sending [`PeerConnection`](crate::PeerConnection) numbers every
/// sequenced message to one peer, so each lane is FIFO per sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaneKey {
    /// Untagged bytes: `send_bytes` / `recv_bytes`.
    Raw { comm_id: u64 },
    /// Tagged bytes: `send_bytes_tagged` / `recv_bytes_tagged` and collectives.
    Tagged { comm_id: u64, tag: u64 },
    /// Point-to-point data: `send` / `recv` and `send_iov` / `recv_iov`.
    Data { comm_id: u64, tag: u32 },
}

const KIND_RAW: u8 = 1;
const KIND_TAGGED: u8 = 2;
const KIND_DATA: u8 = 3;

impl LaneKey {
    /// The tag a TCP bulk sidecar frame of tagged lane `(comm_id, tag)` carries.
    ///
    /// The sidecar frame has one `u64` tag. The root communicator uses the
    /// tag itself. Other communicators mix their `comm_id` in, so the tags of
    /// two communicators on one sidecar stay apart. The result for a nonzero
    /// `comm_id` is never 0, the sidecar's untagged lane.
    pub fn sidecar_tag(comm_id: u64, tag: u64) -> u64 {
        if comm_id == 0 {
            return tag;
        }
        // FNV-1a over comm_id then tag.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in comm_id.to_le_bytes().into_iter().chain(tag.to_le_bytes()) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h.max(1)
    }

    fn parts(self) -> (u8, u64, u64) {
        match self {
            LaneKey::Raw { comm_id } => (KIND_RAW, comm_id, 0),
            LaneKey::Tagged { comm_id, tag } => (KIND_TAGGED, comm_id, tag),
            LaneKey::Data { comm_id, tag } => (KIND_DATA, comm_id, u64::from(tag)),
        }
    }

    fn from_parts(kind: u8, comm_id: u64, tag: u64) -> Option<Self> {
        match kind {
            KIND_RAW if tag == 0 => Some(LaneKey::Raw { comm_id }),
            KIND_TAGGED => Some(LaneKey::Tagged { comm_id, tag }),
            KIND_DATA => u32::try_from(tag)
                .ok()
                .map(|tag| LaneKey::Data { comm_id, tag }),
            _ => None,
        }
    }
}

/// Header of a sequenced stream, after the [`STREAM_TAG_SEQ`] byte.
///
/// Wire format: `[kind: u8][comm_id: u64 LE][tag: u64 LE][seq: u64 LE][len: u64 LE]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SeqHeader {
    /// `None`: the kind byte names no lane this version knows.
    pub lane: Option<LaneKey>,
    pub seq: u64,
    pub len: u64,
}

impl SeqHeader {
    /// Encoded size, without the stream type byte.
    pub(crate) const SIZE: usize = 1 + 8 + 8 + 8 + 8;

    /// The stream type byte followed by the header.
    pub(crate) fn encode(lane: LaneKey, seq: u64, len: u64) -> [u8; 1 + Self::SIZE] {
        let (kind, comm_id, tag) = lane.parts();
        let mut out = [0u8; 1 + Self::SIZE];
        out[0] = STREAM_TAG_SEQ;
        out[1] = kind;
        out[2..10].copy_from_slice(&comm_id.to_le_bytes());
        out[10..18].copy_from_slice(&tag.to_le_bytes());
        out[18..26].copy_from_slice(&seq.to_le_bytes());
        out[26..34].copy_from_slice(&len.to_le_bytes());
        out
    }

    pub(crate) fn decode(buf: &[u8; Self::SIZE]) -> Self {
        let word = |at: usize| {
            let mut w = [0u8; 8];
            w.copy_from_slice(&buf[at..at + 8]);
            u64::from_le_bytes(w)
        };
        Self {
            lane: LaneKey::from_parts(buf[0], word(1), word(9)),
            seq: word(17),
            len: word(25),
        }
    }
}

/// The receiver key of a control message on the mesh.
///
/// A receiver of a key never takes, or fails on, a message of another key.
/// Keys with an epoch keep the rounds of one protocol apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlKey {
    Barrier {
        comm_id: u64,
        epoch: u64,
    },
    BarrierAck {
        comm_id: u64,
        epoch: u64,
    },
    /// Recovery epochs differ between ranks, so the key carries none.
    RecoveryVote,
    RecoveryAgreement,
    ElasticCheckpoint {
        epoch: u64,
    },
    ElasticCheckpointAck {
        epoch: u64,
    },
}

impl ControlKey {
    /// The key of `msg`. `None`: no mesh receiver takes this message kind.
    pub fn of(msg: &NexarMessage) -> Option<Self> {
        match *msg {
            NexarMessage::Barrier { epoch, comm_id } => Some(Self::Barrier { comm_id, epoch }),
            NexarMessage::BarrierAck { epoch, comm_id } => {
                Some(Self::BarrierAck { comm_id, epoch })
            }
            NexarMessage::RecoveryVote { .. } => Some(Self::RecoveryVote),
            NexarMessage::RecoveryAgreement { .. } => Some(Self::RecoveryAgreement),
            NexarMessage::ElasticCheckpoint { epoch } => Some(Self::ElasticCheckpoint { epoch }),
            NexarMessage::ElasticCheckpointAck { epoch, .. } => {
                Some(Self::ElasticCheckpointAck { epoch })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip_every_lane() {
        let lanes = [
            LaneKey::Raw { comm_id: 9 },
            LaneKey::Tagged {
                comm_id: 0,
                tag: u64::MAX,
            },
            LaneKey::Data {
                comm_id: 3,
                tag: u32::MAX,
            },
        ];
        for lane in lanes {
            let wire = SeqHeader::encode(lane, 42, 7);
            assert_eq!(wire[0], STREAM_TAG_SEQ);
            let mut body = [0u8; SeqHeader::SIZE];
            body.copy_from_slice(&wire[1..]);
            let back = SeqHeader::decode(&body);
            assert_eq!(back.lane, Some(lane));
            assert_eq!((back.seq, back.len), (42, 7));
        }
    }

    #[test]
    fn test_unknown_kind_decodes_to_no_lane() {
        let mut body = [0u8; SeqHeader::SIZE];
        body[0] = 0xEE;
        assert_eq!(SeqHeader::decode(&body).lane, None);
    }

    #[test]
    fn test_sidecar_tag_keeps_communicators_apart() {
        assert_eq!(LaneKey::sidecar_tag(0, 5), 5);
        let split = LaneKey::sidecar_tag(77, 5);
        assert_ne!(split, 5);
        assert_ne!(split, 0);
        assert_ne!(split, LaneKey::sidecar_tag(78, 5));
    }

    #[test]
    fn test_control_key_of_barrier_carries_epoch() {
        let msg = NexarMessage::Barrier {
            epoch: 4,
            comm_id: 1,
        };
        assert_eq!(
            ControlKey::of(&msg),
            Some(ControlKey::Barrier {
                comm_id: 1,
                epoch: 4
            })
        );
        assert_eq!(ControlKey::of(&NexarMessage::NodeLeft { rank: 1 }), None);
    }
}
