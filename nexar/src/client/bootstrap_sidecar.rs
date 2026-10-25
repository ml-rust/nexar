//! TCP bulk sidecars for one rank: the lower rank of a pair listens, the higher rank dials.

use crate::config::NexarConfig;
use crate::error::{NexarError, Result};
use crate::transport::tcp_bulk::{
    tcp_bulk_accept_stream, tcp_bulk_connect, tcp_bulk_connect_tls, tcp_bulk_listen,
};
use crate::transport::tls::{make_bulk_tls_client_config, make_bulk_tls_server_config};
use crate::transport::{TaggedBulkTransport, TcpBulkTransport};
use crate::types::Rank;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;

/// Tag of the frame in which a dialing rank names itself. Collective tags count up from 1.
const SIDECAR_HELLO_TAG: u64 = u64::MAX;

/// Time an accepted sidecar connection has to finish TLS and name its rank.
const SIDECAR_HELLO_TIMEOUT: Duration = Duration::from_secs(10);

/// This rank's sidecar settings and, when it has higher-ranked peers, its listener.
pub(super) struct Sidecar {
    enabled: bool,
    tls: bool,
    max_frame_bytes: usize,
    server: Option<Arc<rustls::ServerConfig>>,
    client: Option<Arc<rustls::ClientConfig>>,
    listener: Option<TcpListener>,
    /// Address higher ranks dial. Empty when this rank runs no listener.
    offer: String,
}

impl Sidecar {
    /// Read the sidecar flags from `config` and bind the listener if needed.
    ///
    /// The listener binds `bind`'s IP. A fixed `bind` port is reused for TCP, so
    /// one firewall rule per port covers QUIC (UDP) and the sidecar (TCP).
    pub(super) async fn prepare(
        config: &NexarConfig,
        bind: SocketAddr,
        advertised: SocketAddr,
        has_higher: bool,
        creds: (
            &CertificateDer<'static>,
            &PrivateKeyDer<'static>,
            &CertificateDer<'static>,
        ),
    ) -> Result<Self> {
        let (cert, key, ca) = creds;
        if !config.enable_tcp_bulk_sidecar {
            tracing::info!("TCP bulk sidecar disabled by config");
            return Ok(Self {
                enabled: false,
                tls: config.encrypt_bulk_transport,
                max_frame_bytes: config.max_frame_bytes,
                server: None,
                client: None,
                listener: None,
                offer: String::new(),
            });
        }
        let tls = config.encrypt_bulk_transport;
        let (server, client) = if tls {
            (
                Some(make_bulk_tls_server_config(
                    cert.clone(),
                    key.clone_key(),
                    ca,
                )?),
                Some(make_bulk_tls_client_config(
                    cert.clone(),
                    key.clone_key(),
                    ca,
                )?),
            )
        } else {
            tracing::warn!(
                "TCP bulk sidecar is UNENCRYPTED. Tensor data will be sent in plaintext. \
                 Set NEXAR_ENCRYPT_BULK_TRANSPORT=true or config.encrypt_bulk_transport=true \
                 to enable TLS on the bulk transport."
            );
            (None, None)
        };
        let (listener, offer) = if has_higher {
            let (listener, local) = tcp_bulk_listen(bind).await.map_err(|e| {
                NexarError::transport(format!(
                    "bind TCP bulk sidecar on {bind}: {e}; free the TCP port or set enable_tcp_bulk_sidecar=false on every rank"
                ))
            })?;
            let port = if bind.port() != 0 {
                advertised.port()
            } else {
                local.port()
            };
            (
                Some(listener),
                SocketAddr::new(advertised.ip(), port).to_string(),
            )
        } else {
            (None, String::new())
        };
        Ok(Self {
            enabled: true,
            tls,
            max_frame_bytes: config.max_frame_bytes,
            server,
            client,
            listener,
            offer,
        })
    }

    /// Whether this rank runs sidecars.
    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    /// The `(address, tls)` this rank offers in its `PeerHello`.
    pub(super) fn offer(&self) -> (String, bool) {
        (self.offer.clone(), self.tls)
    }

    /// Check a lower rank's offer against this rank's config.
    ///
    /// Returns the address to dial, or `None` when both ranks run no sidecar.
    /// A mismatch is an error, so sidecars are never off on one side only.
    pub(super) fn check_offer(
        &self,
        me: Rank,
        peer: Rank,
        addr: &str,
        tls: bool,
    ) -> Result<Option<SocketAddr>> {
        let fail = |reason: String| NexarError::ConnectionFailed { rank: peer, reason };
        match (self.enabled, addr.is_empty()) {
            (false, true) => Ok(None),
            (true, true) | (false, false) => Err(fail(format!(
                "TCP bulk sidecar is {} on rank {peer} and {} on rank {me}; set the same enable_tcp_bulk_sidecar on every rank",
                on_off(!addr.is_empty()),
                on_off(self.enabled)
            ))),
            (true, false) if tls != self.tls => Err(fail(format!(
                "TCP bulk sidecar encryption is {} on rank {peer} and {} on rank {me}; set the same encrypt_bulk_transport on every rank",
                on_off(tls),
                on_off(self.tls)
            ))),
            (true, false) => addr
                .parse()
                .map(Some)
                .map_err(|e| fail(format!("invalid sidecar address '{addr}': {e}"))),
        }
    }

    /// Accept one sidecar from each rank in `higher`.
    ///
    /// Each accepted connection names its rank in a hello frame. A connection
    /// that fails, stalls, or names an unexpected rank is dropped with a warning.
    pub(super) async fn accept_from(
        &self,
        higher: &[Rank],
    ) -> Result<HashMap<Rank, TcpBulkTransport>> {
        let Some(listener) = &self.listener else {
            return Ok(HashMap::new());
        };
        let expected: HashSet<Rank> = higher.iter().copied().collect();
        let mut got = HashMap::new();
        let mut pending = FuturesUnordered::new();
        while got.len() < expected.len() {
            tokio::select! {
                accepted = listener.accept() => {
                    let (stream, from) = accepted
                        .map_err(|e| NexarError::transport_with_source("tcp bulk accept", e))?;
                    let tls = self.server.clone();
                    let max_frame = self.max_frame_bytes;
                    pending.push(async move {
                        let named = tokio::time::timeout(SIDECAR_HELLO_TIMEOUT, async {
                            let transport = tcp_bulk_accept_stream(stream, tls, max_frame).await?;
                            let hello = transport.recv_bulk_tagged(SIDECAR_HELLO_TAG, 4).await?;
                            let bytes: [u8; 4] = hello.as_slice().try_into().map_err(|_| {
                                NexarError::DecodeFailed(format!("sidecar hello of {} bytes", hello.len()))
                            })?;
                            Ok::<_, NexarError>((Rank::from_le_bytes(bytes), transport))
                        })
                        .await;
                        (from, named)
                    });
                }
                Some((from, named)) = pending.next() => match named {
                    Ok(Ok((rank, transport))) if expected.contains(&rank) && !got.contains_key(&rank) => {
                        got.insert(rank, transport);
                    }
                    Ok(Ok((rank, _))) => tracing::warn!(%from, rank, "sidecar: dropped a connection from an unexpected rank"),
                    Ok(Err(e)) => tracing::warn!(%from, "sidecar: dropped a connection: {e}"),
                    Err(_) => tracing::warn!(%from, "sidecar: dropped a connection that sent no hello in time"),
                }
            }
        }
        Ok(got)
    }

    /// Dial the sidecar of the lower rank `peer` at `addr` and name this rank.
    pub(super) async fn connect_to(
        &self,
        me: Rank,
        peer: Rank,
        addr: SocketAddr,
    ) -> Result<TcpBulkTransport> {
        let fail = |e: NexarError| NexarError::ConnectionFailed {
            rank: peer,
            reason: format!("TCP bulk sidecar to {addr}: {e}"),
        };
        let transport = match &self.client {
            Some(cfg) => tcp_bulk_connect_tls(addr, Arc::clone(cfg), self.max_frame_bytes).await,
            None => tcp_bulk_connect(addr, self.max_frame_bytes).await,
        }
        .map_err(fail)?;
        transport
            .send_bulk_tagged(SIDECAR_HELLO_TAG, &me.to_le_bytes())
            .await
            .map_err(fail)?;
        Ok(transport)
    }
}

fn on_off(on: bool) -> &'static str {
    if on { "enabled" } else { "disabled" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sidecar(enabled: bool, tls: bool) -> Sidecar {
        Sidecar {
            enabled,
            tls,
            max_frame_bytes: 1024,
            server: None,
            client: None,
            listener: None,
            offer: String::new(),
        }
    }

    #[test]
    fn test_check_offer_agreement() {
        assert_eq!(
            sidecar(false, true).check_offer(1, 0, "", true).unwrap(),
            None
        );
        let addr = sidecar(true, true)
            .check_offer(1, 0, "127.0.0.1:9", true)
            .unwrap();
        assert_eq!(addr, Some("127.0.0.1:9".parse().unwrap()));
    }

    #[test]
    fn test_check_offer_mismatch_is_an_error() {
        for (s, addr, tls) in [
            (sidecar(true, true), "", true),
            (sidecar(false, true), "127.0.0.1:9", true),
            (sidecar(true, true), "127.0.0.1:9", false),
            (sidecar(true, true), "bogus", true),
        ] {
            let err = s.check_offer(1, 0, addr, tls).unwrap_err();
            assert!(
                matches!(err, NexarError::ConnectionFailed { rank: 0, .. }),
                "{err}"
            );
        }
    }
}
