//! Framed control messages on bootstrap QUIC streams.

use crate::error::{NexarError, Result};
use crate::protocol::codec::{decode_message, encode_message};
use crate::protocol::header::HEADER_SIZE;
use crate::protocol::{AbortReason, NexarMessage};
use crate::types::Priority;
use quinn::{RecvStream, SendStream};
use std::time::Duration;
use tokio::sync::mpsc;

/// Largest control payload a bootstrap stream accepts.
pub(crate) const MAX_CONTROL_BYTES: usize = 16 * 1024 * 1024;

/// How long a closing stream waits for the peer to acknowledge its last bytes.
pub(crate) const ABORT_LINGER: Duration = Duration::from_secs(5);

/// Messages a reader queues before it waits for the consumer.
pub(crate) const CONTROL_QUEUE: usize = 64;

/// Messages from a reader task: the stream's id and what it read.
pub(crate) type ControlEvent<K> = (K, Result<NexarMessage>);

/// A bounded channel for [`spawn_reader`] events.
pub(crate) fn control_channel<K>() -> (
    mpsc::Sender<ControlEvent<K>>,
    mpsc::Receiver<ControlEvent<K>>,
) {
    mpsc::channel(CONTROL_QUEUE)
}

/// Read one framed control message.
///
/// A header that announces more than [`MAX_CONTROL_BYTES`] is an error, so a
/// stray connection cannot force a large allocation.
pub(crate) async fn read_control(recv: &mut RecvStream) -> Result<NexarMessage> {
    let mut header = [0u8; HEADER_SIZE];
    recv.read_exact(&mut header)
        .await
        .map_err(|e| NexarError::transport_with_source("read control header", e))?;
    let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    if len > MAX_CONTROL_BYTES {
        return Err(NexarError::DecodeFailed(format!(
            "control message of {len} bytes exceeds the {MAX_CONTROL_BYTES}-byte limit"
        )));
    }
    let mut buf = vec![0u8; HEADER_SIZE + len];
    buf[..HEADER_SIZE].copy_from_slice(&header);
    recv.read_exact(&mut buf[HEADER_SIZE..])
        .await
        .map_err(|e| NexarError::transport_with_source("read control payload", e))?;
    let (_, msg) = decode_message(&buf)?;
    Ok(msg)
}

/// Write one framed control message.
pub(crate) async fn write_control(send: &mut SendStream, msg: &NexarMessage) -> Result<()> {
    let buf = encode_message(msg, Priority::Critical)?;
    send.write_all(&buf)
        .await
        .map_err(|e| NexarError::transport_with_source("write control message", e))
}

/// Forward every message on `recv` to `tx`, tagged with `id`.
///
/// The task stops after the first read error, which it forwards too. When `tx`
/// is full the task waits, so a peer that floods the stream cannot grow the queue.
/// The consumer must keep draining `tx`, or drop it to stop the task.
pub(crate) fn spawn_reader<K: Copy + Send + 'static>(
    id: K,
    mut recv: RecvStream,
    tx: mpsc::Sender<ControlEvent<K>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let msg = read_control(&mut recv).await;
            let stop = msg.is_err();
            if tx.send((id, msg)).await.is_err() || stop {
                return;
            }
        }
    })
}

/// Write `msg` as the last message on `send` and wait until the peer acknowledges it.
///
/// Closing a connection discards unacknowledged stream data, so callers run
/// this before they drop the connection. The wait ends after [`ABORT_LINGER`].
pub(crate) async fn write_final(send: &mut SendStream, msg: &NexarMessage) -> Result<()> {
    write_control(send, msg).await?;
    let _ = send.finish();
    let _ = tokio::time::timeout(ABORT_LINGER, send.stopped()).await;
    Ok(())
}

/// Send `Abort` as the last message on `send`. Errors are ignored: the peer may be gone.
pub(crate) async fn send_abort(send: &mut SendStream, reason: AbortReason) {
    let _ = write_final(send, &NexarMessage::Abort { reason }).await;
}

/// Compare two tokens in time that depends only on their lengths.
pub(crate) fn tokens_match(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reader_waits_when_the_queue_is_full() {
        use crate::transport::TransportListener;
        use crate::transport::tls::make_bootstrap_client_config;

        let listener = TransportListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr();
        let mut client = quinn::Endpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
        client.set_default_client_config(make_bootstrap_client_config().unwrap());
        let accept = async {
            let conn = listener.endpoint().accept().await.unwrap().await.unwrap();
            conn.accept_bi().await.unwrap()
        };
        let dial = async {
            let conn = client.connect(addr, "localhost").unwrap().await.unwrap();
            let (mut send, recv) = conn.open_bi().await.unwrap();
            // QUIC announces a stream when it carries data.
            write_control(&mut send, &NexarMessage::NodeLeft { rank: 0 })
                .await
                .unwrap();
            (conn, send, recv)
        };
        let ((_, server_recv), (_conn, mut send, _recv)) = tokio::join!(accept, dial);

        let (tx, mut rx) = mpsc::channel(2);
        let reader = spawn_reader(7u8, server_recv, tx);
        for rank in 1..10 {
            write_control(&mut send, &NexarMessage::NodeLeft { rank })
                .await
                .unwrap();
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            rx.len(),
            2,
            "the reader queues at most the channel capacity"
        );
        assert!(!reader.is_finished());

        for want in 0..10 {
            let (id, msg) = rx.recv().await.unwrap();
            assert_eq!(id, 7);
            assert!(matches!(msg.unwrap(), NexarMessage::NodeLeft { rank } if rank == want));
        }
        reader.abort();
    }

    #[test]
    fn test_tokens_match() {
        assert!(tokens_match(b"secret", b"secret"));
        assert!(!tokens_match(b"secret", b"secreT"));
        assert!(!tokens_match(b"secret", b"secret2"));
        assert!(tokens_match(b"", b""));
    }
}
