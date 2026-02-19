use crate::error::{NexarError, Result};
use crate::protocol::header::{HEADER_SIZE, Header, MessageType};
use crate::protocol::message::NexarMessage;
use crate::types::Priority;

/// Encode a `NexarMessage` into a framed byte buffer: `[header][rkyv payload]`.
pub fn encode_message(msg: &NexarMessage, priority: Priority) -> Result<Vec<u8>> {
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(msg)
        .map_err(|e| NexarError::EncodeFailed(e.to_string()))?;

    if payload.len() > u32::MAX as usize {
        return Err(NexarError::EncodeFailed(format!(
            "payload too large for framed header: {} bytes exceeds u32::MAX",
            payload.len()
        )));
    }

    let header = Header {
        payload_length: payload.len() as u32,
        priority: priority as u8,
        message_type: MessageType::Control,
    };

    let mut buf = Vec::with_capacity(HEADER_SIZE + payload.len());
    buf.extend_from_slice(&header.encode());
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Decode a framed byte buffer back into a `(Header, NexarMessage)`.
///
/// The input must contain at least `HEADER_SIZE` bytes, followed by
/// `header.payload_length` bytes of rkyv-encoded payload.
pub fn decode_message(buf: &[u8]) -> Result<(Header, NexarMessage)> {
    if buf.len() < HEADER_SIZE {
        return Err(NexarError::DecodeFailed(format!(
            "buffer too short: {} < {HEADER_SIZE}",
            buf.len()
        )));
    }

    let header_bytes: &[u8; HEADER_SIZE] = buf[..HEADER_SIZE]
        .try_into()
        .map_err(|_| NexarError::DecodeFailed("header slice length mismatch".into()))?;

    let header = Header::decode(header_bytes)
        .ok_or_else(|| NexarError::DecodeFailed("invalid header: unknown message type".into()))?;

    let payload_end = HEADER_SIZE + header.payload_length as usize;
    if buf.len() < payload_end {
        return Err(NexarError::DecodeFailed(format!(
            "buffer too short for payload: {} < {payload_end}",
            buf.len()
        )));
    }

    let payload = &buf[HEADER_SIZE..payload_end];
    let msg = rkyv::from_bytes::<NexarMessage, rkyv::rancor::Error>(payload)
        .map_err(|e| NexarError::DecodeFailed(e.to_string()))?;

    Ok((header, msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let msg = NexarMessage::Hello {
            protocol_version: 1,
            capabilities: 0xABCD,
        };
        let buf = encode_message(&msg, Priority::Critical).unwrap();
        let (header, decoded) = decode_message(&buf).unwrap();
        assert_eq!(header.priority, Priority::Critical as u8);
        assert_eq!(header.message_type, MessageType::Control);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_all_priorities() {
        let msg = NexarMessage::Heartbeat { timestamp_ns: 42 };
        for priority in [Priority::Critical, Priority::Realtime, Priority::Bulk] {
            let buf = encode_message(&msg, priority).unwrap();
            let (header, _) = decode_message(&buf).unwrap();
            assert_eq!(header.priority, priority as u8);
        }
    }

    #[test]
    fn test_complex_message_roundtrip() {
        let msg = NexarMessage::Welcome {
            rank: 7,
            world_size: 128,
            peers: (0..128)
                .map(|i| (i, format!("10.0.{}.{}:9000", i / 256, i % 256)))
                .collect(),
            ca_cert: vec![1, 2, 3],
            node_cert: vec![4, 5, 6],
            node_key: vec![7, 8, 9],
        };
        let buf = encode_message(&msg, Priority::Realtime).unwrap();
        let (_, decoded) = decode_message(&buf).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_decode_buffer_too_short() {
        let result = decode_message(&[0u8; 4]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("too short"), "got: {err}");
    }

    #[test]
    fn test_decode_invalid_message_type() {
        let mut buf = [0u8; 8];
        buf[5] = 255; // invalid MessageType
        let result = decode_message(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_truncated_payload() {
        let msg = NexarMessage::Barrier { epoch: 1 };
        let mut buf = encode_message(&msg, Priority::Critical).unwrap();
        buf.truncate(HEADER_SIZE + 2); // truncate payload
        let result = decode_message(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_message_roundtrip() {
        let msg = NexarMessage::Data {
            tag: 42,
            src_rank: 3,
            payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        let buf = encode_message(&msg, Priority::Bulk).unwrap();
        let (header, decoded) = decode_message(&buf).unwrap();
        assert_eq!(header.priority, Priority::Bulk as u8);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_rpc_roundtrip() {
        let msg = NexarMessage::Rpc {
            req_id: u64::MAX,
            fn_id: 1000,
            payload: vec![1; 1024],
        };
        let buf = encode_message(&msg, Priority::Realtime).unwrap();
        let (_, decoded) = decode_message(&buf).unwrap();
        assert_eq!(decoded, msg);
    }
}
