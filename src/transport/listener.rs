use crate::error::{NexarError, Result};
use crate::transport::tls::{generate_self_signed_cert, make_server_config};
use std::net::SocketAddr;

/// Listens for incoming QUIC connections on a bound address.
pub struct TransportListener {
    endpoint: quinn::Endpoint,
    local_addr: SocketAddr,
}

impl TransportListener {
    /// Bind a QUIC listener on the given address with a self-signed certificate.
    pub fn bind(addr: SocketAddr) -> Result<Self> {
        let (cert, key) = generate_self_signed_cert()?;
        let server_config = make_server_config(cert, key)?;

        let endpoint = quinn::Endpoint::server(server_config, addr)
            .map_err(|e| NexarError::Transport(format!("bind {addr}: {e}")))?;

        let local_addr = endpoint
            .local_addr()
            .map_err(|e| NexarError::Transport(format!("local_addr: {e}")))?;

        Ok(Self {
            endpoint,
            local_addr,
        })
    }

    /// Bind with an existing server config (for sharing certs across tests).
    pub fn bind_with_config(addr: SocketAddr, config: quinn::ServerConfig) -> Result<Self> {
        let endpoint = quinn::Endpoint::server(config, addr)
            .map_err(|e| NexarError::Transport(format!("bind {addr}: {e}")))?;

        let local_addr = endpoint
            .local_addr()
            .map_err(|e| NexarError::Transport(format!("local_addr: {e}")))?;

        Ok(Self {
            endpoint,
            local_addr,
        })
    }

    /// Accept the next incoming QUIC connection.
    pub async fn accept(&self) -> Result<quinn::Connection> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| NexarError::Transport("endpoint closed".into()))?;

        incoming
            .await
            .map_err(|e| NexarError::Transport(format!("accept: {e}")))
    }

    /// The local address this listener is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Get a reference to the underlying quinn Endpoint (for creating client connections too).
    pub fn endpoint(&self) -> &quinn::Endpoint {
        &self.endpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bind_listener() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let listener = TransportListener::bind(addr).unwrap();
        assert_ne!(listener.local_addr().port(), 0);
    }
}
