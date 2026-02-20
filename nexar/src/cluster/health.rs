use crate::error::Result;
use crate::protocol::NexarMessage;
use crate::transport::PeerConnection;
use crate::types::{Priority, Rank};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;

/// Manages heartbeat sending and failure detection for a node.
pub struct HealthMonitor {
    interval: Duration,
    timeout: Duration,
}

impl HealthMonitor {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a monitor with explicit interval and timeout.
    pub fn with_timeout(interval: Duration, timeout: Duration) -> Self {
        Self { interval, timeout }
    }

    /// Send a single heartbeat to the given peer.
    pub async fn send_heartbeat(&self, peer: &PeerConnection) -> Result<()> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        peer.send_message(
            &NexarMessage::Heartbeat { timestamp_ns },
            Priority::Critical,
        )
        .await
    }

    /// Get the configured interval.
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Spawn a background task that periodically sends heartbeats and detects failures.
    ///
    /// When a peer exceeds the timeout without a successful heartbeat send, its rank
    /// is added to the dead peers list and published on `failure_tx`.
    ///
    /// Returns a `JoinHandle` that can be aborted to stop monitoring.
    pub fn start_monitoring(
        &self,
        peers: Vec<(Rank, Arc<PeerConnection>)>,
        failure_tx: Arc<watch::Sender<Vec<Rank>>>,
    ) -> tokio::task::JoinHandle<()> {
        let interval = self.interval;
        let timeout = self.timeout;

        tokio::spawn(async move {
            let mut last_success: HashMap<Rank, Instant> =
                peers.iter().map(|(r, _)| (*r, Instant::now())).collect();
            let mut dead: Vec<Rank> = Vec::new();

            loop {
                tokio::time::sleep(interval).await;

                for (rank, peer) in &peers {
                    if dead.contains(rank) {
                        continue;
                    }

                    let timestamp_ns = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;

                    let msg = NexarMessage::Heartbeat { timestamp_ns };
                    match peer.send_message(&msg, Priority::Critical).await {
                        Ok(()) => {
                            last_success.insert(*rank, Instant::now());
                        }
                        Err(e) => {
                            tracing::warn!(peer_rank = rank, error = %e, "heartbeat send failed");
                        }
                    }

                    if let Some(last) = last_success.get(rank)
                        && last.elapsed() > timeout
                    {
                        tracing::error!(peer_rank = rank, "peer exceeded heartbeat timeout");
                        dead.push(*rank);
                        dead.sort();
                        dead.dedup();
                        let _ = failure_tx.send(dead.clone());
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_interval() {
        let hm = HealthMonitor::new(Duration::from_secs(1));
        assert_eq!(hm.interval(), Duration::from_secs(1));
    }

    #[test]
    fn test_health_monitor_with_timeout() {
        let hm = HealthMonitor::with_timeout(Duration::from_millis(500), Duration::from_secs(3));
        assert_eq!(hm.interval(), Duration::from_millis(500));
        assert_eq!(hm.timeout, Duration::from_secs(3));
    }
}
