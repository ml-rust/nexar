//! Runtime-configurable tuning parameters for nexar.
//!
//! All values have sensible defaults. Override via environment variables
//! (prefixed `NEXAR_`) or by constructing a custom `NexarConfig`.

use std::time::Duration;

/// Tuning parameters for collective operations and transport.
#[derive(Debug, Clone)]
pub struct NexarConfig {
    /// Timeout for individual send/recv operations within collectives.
    pub collective_timeout: Duration,

    /// Timeout for barrier operations.
    pub barrier_timeout: Duration,

    /// Timeout for RPC calls.
    pub rpc_timeout: Duration,

    /// Messages larger than this threshold use pipelined ring allreduce
    /// instead of ring or halving-doubling.
    pub large_msg_bytes: usize,

    /// Segment size for pipelined allreduce stages.
    pub pipeline_segment_bytes: usize,

    /// Maximum world size for preferring ring allreduce over
    /// halving-doubling in the medium-message range.
    pub ring_max_world: usize,

    /// Enable TCP bulk sidecar connections for large tensor transfers.
    ///
    /// When enabled, a raw TCP connection is established alongside each QUIC
    /// peer connection. Collectives automatically prefer the TCP path for
    /// large payloads to bypass QUIC's AES-256-GCM overhead.
    ///
    /// **Security warning:** Unless `encrypt_bulk_transport` is also set,
    /// data sent via the TCP sidecar is **unencrypted**. Do not enable in
    /// zero-trust environments (public clouds) without encryption.
    pub enable_tcp_bulk_sidecar: bool,

    /// Require TLS encryption on TCP bulk sidecar connections.
    ///
    /// Defaults to `true` (encrypted). When `true`, the TCP sidecar uses
    /// TLS with the cluster CA for encryption. Set to `false` to disable
    /// encryption for maximum throughput on trusted networks (e.g. isolated
    /// InfiniBand fabrics).
    ///
    /// Only meaningful when `enable_tcp_bulk_sidecar` is `true`.
    pub encrypt_bulk_transport: bool,

    /// Maximum memory (in bytes) that compressed allreduce may allocate for
    /// buffering compressed chunks from all peers.
    ///
    /// The allgather-then-reduce algorithm requires `O(N × chunk_size)` memory
    /// where N is the world size. This limit prevents OOM crashes on large
    /// clusters. Set to `0` to disable the check.
    pub compressed_allreduce_max_bytes: usize,

    /// Interval between heartbeat probes sent to each peer.
    pub heartbeat_interval: Duration,

    /// Duration after which a peer with no heartbeat response is considered dead.
    pub heartbeat_timeout: Duration,
}

impl Default for NexarConfig {
    fn default() -> Self {
        Self {
            collective_timeout: Duration::from_secs(30),
            barrier_timeout: Duration::from_secs(30),
            rpc_timeout: Duration::from_secs(30),
            large_msg_bytes: 8 * 1024 * 1024,        // 8 MiB
            pipeline_segment_bytes: 2 * 1024 * 1024, // 2 MiB
            ring_max_world: 8,
            enable_tcp_bulk_sidecar: true,
            encrypt_bulk_transport: true,
            compressed_allreduce_max_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB
            heartbeat_interval: Duration::from_secs(1),
            heartbeat_timeout: Duration::from_secs(5),
        }
    }
}

impl NexarConfig {
    /// Load config from environment variables, falling back to defaults.
    ///
    /// Recognized variables:
    /// - `NEXAR_COLLECTIVE_TIMEOUT_SECS`
    /// - `NEXAR_BARRIER_TIMEOUT_SECS`
    /// - `NEXAR_RPC_TIMEOUT_SECS`
    /// - `NEXAR_LARGE_MSG_BYTES`
    /// - `NEXAR_PIPELINE_SEGMENT_BYTES`
    /// - `NEXAR_RING_MAX_WORLD`
    /// - `NEXAR_ENABLE_TCP_BULK_SIDECAR` (default: true, set to "0" or "false" to disable)
    /// - `NEXAR_ENCRYPT_BULK_TRANSPORT` (default: true, set to "0" or "false" to disable)
    /// - `NEXAR_COMPRESSED_ALLREDUCE_MAX_BYTES` (default: 4 GiB, set to "0" to disable)
    /// - `NEXAR_HEARTBEAT_INTERVAL_SECS` (default: 1)
    /// - `NEXAR_HEARTBEAT_TIMEOUT_SECS` (default: 5)
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("NEXAR_COLLECTIVE_TIMEOUT_SECS")
            && let Ok(s) = v.parse::<u64>()
        {
            cfg.collective_timeout = Duration::from_secs(s);
        }
        if let Ok(v) = std::env::var("NEXAR_BARRIER_TIMEOUT_SECS")
            && let Ok(s) = v.parse::<u64>()
        {
            cfg.barrier_timeout = Duration::from_secs(s);
        }
        if let Ok(v) = std::env::var("NEXAR_RPC_TIMEOUT_SECS")
            && let Ok(s) = v.parse::<u64>()
        {
            cfg.rpc_timeout = Duration::from_secs(s);
        }
        if let Ok(v) = std::env::var("NEXAR_LARGE_MSG_BYTES")
            && let Ok(n) = v.parse::<usize>()
        {
            cfg.large_msg_bytes = n;
        }
        if let Ok(v) = std::env::var("NEXAR_PIPELINE_SEGMENT_BYTES")
            && let Ok(n) = v.parse::<usize>()
        {
            cfg.pipeline_segment_bytes = n;
        }
        if let Ok(v) = std::env::var("NEXAR_RING_MAX_WORLD")
            && let Ok(n) = v.parse::<usize>()
        {
            cfg.ring_max_world = n;
        }
        if let Ok(v) = std::env::var("NEXAR_ENABLE_TCP_BULK_SIDECAR") {
            cfg.enable_tcp_bulk_sidecar = v != "0" && v.to_lowercase() != "false";
        }
        if let Ok(v) = std::env::var("NEXAR_ENCRYPT_BULK_TRANSPORT") {
            cfg.encrypt_bulk_transport = v != "0" && v.to_lowercase() != "false";
        }
        if let Ok(v) = std::env::var("NEXAR_COMPRESSED_ALLREDUCE_MAX_BYTES")
            && let Ok(n) = v.parse::<usize>()
        {
            cfg.compressed_allreduce_max_bytes = n;
        }

        if let Ok(v) = std::env::var("NEXAR_HEARTBEAT_INTERVAL_SECS")
            && let Ok(s) = v.parse::<u64>()
        {
            cfg.heartbeat_interval = Duration::from_secs(s);
        }
        if let Ok(v) = std::env::var("NEXAR_HEARTBEAT_TIMEOUT_SECS")
            && let Ok(s) = v.parse::<u64>()
        {
            cfg.heartbeat_timeout = Duration::from_secs(s);
        }

        cfg
    }
}
