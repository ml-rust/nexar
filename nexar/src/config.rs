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
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("NEXAR_COLLECTIVE_TIMEOUT_SECS") {
            if let Ok(s) = v.parse::<u64>() {
                cfg.collective_timeout = Duration::from_secs(s);
            }
        }
        if let Ok(v) = std::env::var("NEXAR_BARRIER_TIMEOUT_SECS") {
            if let Ok(s) = v.parse::<u64>() {
                cfg.barrier_timeout = Duration::from_secs(s);
            }
        }
        if let Ok(v) = std::env::var("NEXAR_RPC_TIMEOUT_SECS") {
            if let Ok(s) = v.parse::<u64>() {
                cfg.rpc_timeout = Duration::from_secs(s);
            }
        }
        if let Ok(v) = std::env::var("NEXAR_LARGE_MSG_BYTES") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.large_msg_bytes = n;
            }
        }
        if let Ok(v) = std::env::var("NEXAR_PIPELINE_SEGMENT_BYTES") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.pipeline_segment_bytes = n;
            }
        }
        if let Ok(v) = std::env::var("NEXAR_RING_MAX_WORLD") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.ring_max_world = n;
            }
        }

        cfg
    }
}
