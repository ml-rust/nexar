//! Graceful shutdown of a rank's peer connections.
//!
//! Closing a QUIC connection discards stream data the peer has not
//! acknowledged. A rank therefore closes a connection only after the peer
//! signals that it is done too:
//!
//! 1. Each side shuts down its TCP sidecar writer (flush, TLS `close_notify`, FIN).
//! 2. Each side opens a bidirectional stream and writes one `DONE` byte.
//! 3. Each side waits for the peer's `DONE` byte, or for the peer to close the
//!    connection on purpose. Either means the peer read everything it needs.
//! 4. Each side closes the connection and waits until the endpoint sends the close.

use crate::client::NexarClient;
use crate::error::{NexarError, Result};
use crate::transport::TaggedBulkTransport;
use crate::types::Rank;
use quinn::{ConnectionError, VarInt};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::Instant;

/// Close code of a connection after a completed shutdown.
pub(crate) const SHUTDOWN_CODE: u32 = 0x4e58_0001;
/// Close code of a connection whose last client dropped without `shutdown`.
pub(crate) const DROP_CODE: u32 = 0x4e58_0002;
/// The byte that says "this side is done".
const DONE: u8 = 0xD0;
/// Longest wait for close frames to leave the endpoint.
const CLOSE_FLUSH: Duration = Duration::from_secs(1);

/// A peer's original rank, its connection, and its sidecar, if any.
type ShutdownPeer = (
    Rank,
    quinn::Connection,
    Option<Arc<dyn TaggedBulkTransport>>,
);

/// The connections and endpoints that a root client and every client derived
/// from it (split, rebuilt) share. The last owner to drop closes them.
#[derive(Default)]
pub(crate) struct MeshOwner {
    conns: Mutex<Vec<quinn::Connection>>,
    endpoints: Mutex<Vec<quinn::Endpoint>>,
}

impl MeshOwner {
    /// Register a connection to close with this mesh.
    pub(crate) fn add_conn(&self, conn: quinn::Connection) {
        self.conns
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(conn);
    }

    /// Keep `endpoint` alive as long as the mesh.
    pub(crate) fn add_endpoint(&self, endpoint: quinn::Endpoint) {
        self.endpoints
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(endpoint);
    }

    fn endpoints(&self) -> Vec<quinn::Endpoint> {
        self.endpoints
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

impl Drop for MeshOwner {
    /// Best effort and non-blocking: close every connection at once. Unlike
    /// [`NexarClient::shutdown`], this can discard data the peers did not acknowledge.
    fn drop(&mut self) {
        let conns = self.conns.get_mut().unwrap_or_else(|p| p.into_inner());
        for conn in conns.iter() {
            conn.close(VarInt::from_u32(DROP_CODE), b"client dropped");
        }
    }
}

impl NexarClient {
    /// Shut down with the barrier timeout from this client's config.
    ///
    /// See [`NexarClient::shutdown_with_timeout`].
    pub async fn shutdown(self) -> Result<()> {
        let timeout = self.config.barrier_timeout;
        self.shutdown_with_timeout(timeout).await
    }

    /// Close every peer connection without losing data in flight.
    ///
    /// Every rank calls it after its last send and recv. It returns once each
    /// direct peer also called it, or closed its side on purpose. Data this
    /// rank sent before the call reaches every peer that is still running.
    ///
    /// Call it once per process, on the root client or the client that
    /// replaced it. Communicators split from this client stop working.
    ///
    /// On timeout every connection is closed anyway, and the call returns
    /// [`NexarError::ShutdownIncomplete`] naming the peers that did not finish.
    pub async fn shutdown_with_timeout(mut self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        if let Some(monitor) = self._monitor_handle.take() {
            // A peer that closes during shutdown is not a failure.
            monitor.abort();
        }

        let peers: Vec<ShutdownPeer> = self
            .peers
            .iter()
            .map(|(&rank, peer)| {
                let sidecar = peer
                    .extension::<Arc<dyn TaggedBulkTransport>>()
                    .ok()
                    .flatten()
                    .map(|t| Arc::clone(&*t));
                (self.resolve_rank(rank), peer.conn.clone(), sidecar)
            })
            .collect();

        let results =
            futures::future::join_all(peers.into_iter().map(|(rank, conn, sidecar)| async move {
                if let Some(sidecar) = sidecar
                    && let Err(e) = sidecar.shutdown().await
                {
                    tracing::debug!(peer = rank, "sidecar shutdown: {e}");
                }
                let done = tokio::time::timeout_at(deadline, exchange_done(&conn)).await;
                conn.close(VarInt::from_u32(SHUTDOWN_CODE), b"shutdown");
                (rank, done)
            }))
            .await;

        let flush_until = deadline.min(Instant::now() + CLOSE_FLUSH);
        let endpoints = self.mesh.endpoints();
        let idle = futures::future::join_all(endpoints.iter().map(|e| e.wait_idle()));
        let _ = tokio::time::timeout_at(flush_until, idle).await;

        let mut timed_out = Vec::new();
        let mut disconnected = Vec::new();
        for (rank, done) in results {
            match done {
                Err(_) => timed_out.push(rank),
                Ok(Err(e)) => {
                    tracing::warn!(peer = rank, "shutdown: peer disconnected first: {e}");
                    disconnected.push(rank);
                }
                Ok(Ok(())) => {}
            }
        }
        if timed_out.is_empty() && disconnected.is_empty() {
            return Ok(());
        }
        timed_out.sort();
        disconnected.sort();
        Err(NexarError::ShutdownIncomplete {
            timed_out,
            disconnected,
            timeout_ms: timeout.as_millis() as u64,
        })
    }
}

/// Signal `DONE` to the peer and wait for the peer's `DONE`.
///
/// A peer that closes the connection with an application close code also counts
/// as done: it closes only after its own `DONE` exchange or a deliberate drop.
async fn exchange_done(conn: &quinn::Connection) -> std::result::Result<(), ConnectionError> {
    let signal = async {
        let (mut send, _recv) = conn.open_bi().await?;
        let _ = send.write_all(&[DONE]).await;
        let _ = send.finish();
        // Keep `send` alive until the peer's answer arrives.
        Ok::<_, ConnectionError>(send)
    };
    let answer = async {
        let (_send, mut recv) = conn.accept_bi().await?;
        let mut byte = [0u8; 1];
        match recv.read_exact(&mut byte).await {
            Ok(()) => Ok(()),
            Err(quinn::ReadExactError::ReadError(quinn::ReadError::ConnectionLost(e))) => Err(e),
            Err(_) => Ok(()),
        }
    };
    let (signaled, answered) = tokio::join!(signal, answer);
    let result = answered.and(signaled.map(|_| ()));
    match result {
        Err(ConnectionError::ApplicationClosed(_)) => Ok(()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuAdapter;

    async fn pair() -> (NexarClient, NexarClient) {
        let config = crate::config::NexarConfig::default();
        let mut clients =
            NexarClient::bootstrap_local_with_config(2, Arc::new(CpuAdapter::new()), config)
                .await
                .unwrap();
        let c1 = clients.pop().unwrap();
        (clients.pop().unwrap(), c1)
    }

    #[tokio::test]
    async fn test_send_then_shutdown_delivers() {
        let (c0, c1) = pair().await;
        let payload = vec![7u8; 1 << 20];
        let receiver = tokio::spawn(async move {
            let mut buf = vec![0u8; 1 << 20];
            unsafe { c1.recv(buf.as_mut_ptr() as u64, buf.len(), 0, 9).await }?;
            c1.shutdown().await?;
            Ok::<_, NexarError>(buf)
        });
        unsafe { c0.send(payload.as_ptr() as u64, payload.len(), 1, 9).await }.unwrap();
        c0.shutdown().await.unwrap();
        assert_eq!(receiver.await.unwrap().unwrap(), payload);
    }

    #[tokio::test]
    async fn test_shutdown_times_out_when_peer_stays() {
        let (c0, c1) = pair().await;
        let err = c0
            .shutdown_with_timeout(Duration::from_millis(300))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, NexarError::ShutdownIncomplete { timed_out, .. } if timed_out == &[1]),
            "{err}"
        );
        drop(c1);
    }

    #[tokio::test]
    async fn test_dropped_peer_counts_as_done() {
        let (c0, c1) = pair().await;
        drop(c1);
        c0.shutdown_with_timeout(Duration::from_secs(5))
            .await
            .unwrap();
    }
}
