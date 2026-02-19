use crate::error::Result;
use crate::protocol::NexarMessage;
use crate::transport::PeerConnection;
use crate::types::Priority;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Manages heartbeat sending for a node.
pub struct HealthMonitor {
    interval: Duration,
}

impl HealthMonitor {
    pub fn new(interval: Duration) -> Self {
        Self { interval }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_interval() {
        let hm = HealthMonitor::new(Duration::from_secs(1));
        assert_eq!(hm.interval(), Duration::from_secs(1));
    }
}
