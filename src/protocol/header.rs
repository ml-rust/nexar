/// Size of the wire header in bytes.
pub const HEADER_SIZE: usize = 8;

/// Type tag for the message that follows the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Serialized `NexarMessage` (rkyv-encoded control message).
    Control = 0,
    /// Tensor metadata header (dtype, count, tensor_id).
    TensorMeta = 1,
    /// Raw data bytes (tensor payload).
    RawData = 2,
}

impl MessageType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(MessageType::Control),
            1 => Some(MessageType::TensorMeta),
            2 => Some(MessageType::RawData),
            _ => None,
        }
    }
}

/// 8-byte wire header prepended to every framed message.
///
/// ```text
/// [0..4] payload_length: u32 LE
/// [4]    priority: u8
/// [5]    message_type: u8
/// [6..8] reserved: u16 (must be 0)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Length of the payload following this header.
    pub payload_length: u32,
    /// Priority lane (maps to `Priority` enum).
    pub priority: u8,
    /// Type of the payload.
    pub message_type: MessageType,
}

impl Header {
    /// Encode header to 8 bytes (little-endian).
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.payload_length.to_le_bytes());
        buf[4] = self.priority;
        buf[5] = self.message_type as u8;
        // buf[6..8] reserved = 0
        buf
    }

    /// Decode header from 8 bytes.
    ///
    /// Returns `None` if the message type byte is invalid.
    pub fn decode(buf: &[u8; HEADER_SIZE]) -> Option<Self> {
        let payload_length = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let priority = buf[4];
        let message_type = MessageType::from_u8(buf[5])?;
        // buf[6..8] reserved — ignored on decode
        Some(Header {
            payload_length,
            priority,
            message_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let h = Header {
            payload_length: 12345,
            priority: 2,
            message_type: MessageType::Control,
        };
        let encoded = h.encode();
        let decoded = Header::decode(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn test_header_all_types() {
        for (ty, val) in [
            (MessageType::Control, 0u8),
            (MessageType::TensorMeta, 1),
            (MessageType::RawData, 2),
        ] {
            let h = Header {
                payload_length: 100,
                priority: 0,
                message_type: ty,
            };
            let enc = h.encode();
            assert_eq!(enc[5], val);
            let dec = Header::decode(&enc).unwrap();
            assert_eq!(dec.message_type, ty);
        }
    }

    #[test]
    fn test_header_max_payload() {
        let h = Header {
            payload_length: u32::MAX,
            priority: 0,
            message_type: MessageType::RawData,
        };
        let enc = h.encode();
        let dec = Header::decode(&enc).unwrap();
        assert_eq!(dec.payload_length, u32::MAX);
    }

    #[test]
    fn test_header_invalid_message_type() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[5] = 255; // invalid
        assert!(Header::decode(&buf).is_none());
    }

    #[test]
    fn test_header_reserved_bytes_zeroed() {
        let h = Header {
            payload_length: 42,
            priority: 1,
            message_type: MessageType::TensorMeta,
        };
        let enc = h.encode();
        assert_eq!(enc[6], 0);
        assert_eq!(enc[7], 0);
    }

    #[test]
    fn test_message_type_from_u8() {
        assert_eq!(MessageType::from_u8(0), Some(MessageType::Control));
        assert_eq!(MessageType::from_u8(1), Some(MessageType::TensorMeta));
        assert_eq!(MessageType::from_u8(2), Some(MessageType::RawData));
        assert_eq!(MessageType::from_u8(3), None);
    }
}
