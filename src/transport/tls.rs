use crate::error::{NexarError, Result};
use std::sync::Arc;

/// Generate a self-signed certificate and private key for cluster-internal TLS.
pub fn generate_self_signed_cert() -> Result<(
    rustls::pki_types::CertificateDer<'static>,
    rustls::pki_types::PrivateKeyDer<'static>,
)> {
    let cert_params = rcgen::CertificateParams::new(vec!["localhost".into()])
        .map_err(|e| NexarError::Tls(e.to_string()))?;
    let key_pair = rcgen::KeyPair::generate().map_err(|e| NexarError::Tls(e.to_string()))?;
    let cert = cert_params
        .self_signed(&key_pair)
        .map_err(|e| NexarError::Tls(e.to_string()))?;

    let cert_der = rustls::pki_types::CertificateDer::from(cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::try_from(key_pair.serialize_der())
        .map_err(|e| NexarError::Tls(e.to_string()))?;

    Ok((cert_der, key_der))
}

/// Build a rustls `ServerConfig` from a cert + key for the QUIC server side.
pub fn make_server_config(
    cert: rustls::pki_types::CertificateDer<'static>,
    key: rustls::pki_types::PrivateKeyDer<'static>,
) -> Result<quinn::ServerConfig> {
    let mut tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert], key)
        .map_err(|e| NexarError::Tls(e.to_string()))?;

    tls_config.alpn_protocols = vec![b"nexar/1".to_vec()];

    let quic_config = quinn::crypto::rustls::QuicServerConfig::try_from(Arc::new(tls_config))
        .map_err(|e| NexarError::Tls(e.to_string()))?;

    Ok(quinn::ServerConfig::with_crypto(Arc::new(quic_config)))
}

/// Build a rustls `ClientConfig` that skips certificate verification
/// (suitable for cluster-internal communication with self-signed certs).
pub fn make_client_config() -> Result<quinn::ClientConfig> {
    let mut tls_config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    tls_config.alpn_protocols = vec![b"nexar/1".to_vec()];

    let quic_config = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(tls_config))
        .map_err(|e| NexarError::Tls(e.to_string()))?;

    Ok(quinn::ClientConfig::new(Arc::new(quic_config)))
}

/// Certificate verifier that accepts any certificate.
/// Used for cluster-internal communication where all nodes are trusted.
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_self_signed_cert() {
        let (cert, key) = generate_self_signed_cert().unwrap();
        assert!(!cert.is_empty());
        // Key should be parseable
        let _ = key;
    }

    #[test]
    fn test_make_server_config() {
        let (cert, key) = generate_self_signed_cert().unwrap();
        let config = make_server_config(cert, key).unwrap();
        let _ = config;
    }

    #[test]
    fn test_make_client_config() {
        let config = make_client_config().unwrap();
        let _ = config;
    }
}
