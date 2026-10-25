//! Send side of sequenced messages.
//!
//! Every sequenced message to one peer takes the next number of one counter.
//! A send that does not complete (error or cancelled future) leaves a gap the
//! receiver would wait on forever, so the sender reports the number in a skip
//! notice.

use super::lane::{MAX_SKIPS, STREAM_TAG_SKIP};
use crate::error::{NexarError, Result};
use quinn::VarInt;

/// Stream reset code of an abandoned sequenced send.
const ABANDONED_CODE: u32 = 0x4e58_0010;

/// One sequenced send in progress. Dropping it before
/// [`finish`](Self::finish) resets the stream and sends a skip notice.
pub(crate) struct SeqSend {
    stream: Option<quinn::SendStream>,
    seq: u64,
    conn: quinn::Connection,
}

impl SeqSend {
    pub(crate) fn new(stream: quinn::SendStream, seq: u64, conn: quinn::Connection) -> Self {
        Self {
            stream: Some(stream),
            seq,
            conn,
        }
    }

    pub(crate) async fn write(&mut self, bytes: &[u8], what: &'static str) -> Result<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| NexarError::transport("sequenced stream already finished"))?;
        stream
            .write_all(bytes)
            .await
            .map_err(|e| NexarError::transport_with_source(what, e))
    }

    /// Finish the stream. The message is then complete on the wire.
    pub(crate) fn finish(mut self) -> Result<()> {
        let Some(mut stream) = self.stream.take() else {
            return Err(NexarError::transport("sequenced stream already finished"));
        };
        match stream.finish() {
            Ok(()) => Ok(()),
            Err(e) => {
                abandon(&self.conn, self.seq);
                Err(NexarError::transport_with_source("finish stream", e))
            }
        }
    }
}

impl Drop for SeqSend {
    fn drop(&mut self) {
        if let Some(mut stream) = self.stream.take() {
            // Reset, so the peer never reads a partial payload as complete.
            let _ = stream.reset(VarInt::from_u32(ABANDONED_CODE));
            abandon(&self.conn, self.seq);
        }
    }
}

/// Tell the peer that `seq` never completes, on a task of its own.
fn abandon(conn: &quinn::Connection, seq: u64) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        // No runtime: the process is exiting, and the connection with it.
        return;
    };
    let conn = conn.clone();
    runtime.spawn(async move {
        if let Err(e) = send_skip(&conn, &[seq]).await {
            tracing::debug!(seq, "skip notice not sent: {e}");
        }
    });
}

/// Send one skip notice: `[0x06][count: u32 LE][seq: u64 LE; count]`.
async fn send_skip(conn: &quinn::Connection, seqs: &[u64]) -> Result<()> {
    let count = u32::try_from(seqs.len())
        .ok()
        .filter(|&n| n <= MAX_SKIPS)
        .ok_or_else(|| NexarError::transport("too many sequence numbers in one skip notice"))?;
    let mut buf = Vec::with_capacity(5 + seqs.len() * 8);
    buf.push(STREAM_TAG_SKIP);
    buf.extend_from_slice(&count.to_le_bytes());
    for seq in seqs {
        buf.extend_from_slice(&seq.to_le_bytes());
    }
    let mut stream = conn
        .open_uni()
        .await
        .map_err(|e| NexarError::transport_with_source("open skip stream", e))?;
    stream
        .write_all(&buf)
        .await
        .map_err(|e| NexarError::transport_with_source("write skip notice", e))?;
    stream
        .finish()
        .map_err(|e| NexarError::transport_with_source("finish skip notice", e))
}
