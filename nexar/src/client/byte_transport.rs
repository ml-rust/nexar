use crate::error::Result;
use crate::transport::buffer_pool::PooledBuf;
use crate::transport::lane::LaneKey;
use crate::types::Rank;
use std::sync::Arc;

use super::NexarClient;

impl NexarClient {
    /// The untagged lane of this communicator.
    fn raw_lane(&self) -> LaneKey {
        LaneKey::Raw {
            comm_id: self.comm_id,
        }
    }

    /// The lane of `tag` in this communicator.
    fn tagged_lane(&self, tag: u64) -> LaneKey {
        LaneKey::Tagged {
            comm_id: self.comm_id,
            tag,
        }
    }

    /// Send raw bytes to a peer.
    ///
    /// Uses comm-aware send for split communicators. Always uses QUIC transport.
    /// For bulk data in collectives, prefer `send_bytes_best_effort` which
    /// auto-selects RDMA when available.
    pub async fn send_bytes(&self, dest: Rank, data: &[u8]) -> Result<()> {
        self.peer(dest)?.send_lane(self.raw_lane(), data).await
    }

    /// Send raw bytes using the best available transport (RDMA if available, QUIC fallback).
    /// For split communicators, always uses QUIC (comm-id routing required).
    pub(crate) async fn send_bytes_best_effort(&self, dest: Rank, data: &[u8]) -> Result<()> {
        let peer = self.peer(dest)?;
        if self.comm_id == 0 {
            peer.send_raw_best_effort(data).await
        } else {
            // Split communicators need comm_id tagging, QUIC only.
            peer.send_lane(self.raw_lane(), data).await
        }
    }

    /// Send raw bytes to a peer with a u64 tag.
    ///
    /// Tagged sends are always via QUIC. Messages of one `(dest, tag)` arrive
    /// in send order. Each communicator has its own tag space.
    pub async fn send_bytes_tagged(&self, dest: Rank, tag: u64, data: &[u8]) -> Result<()> {
        self.peer(dest)?
            .send_lane(self.tagged_lane(tag), data)
            .await
    }

    /// Receive tagged bytes using the best available transport.
    ///
    /// Tries `TaggedBulkTransport` (TCP sidecar) first, falling back to QUIC.
    pub(crate) async fn recv_bytes_tagged_best_effort(
        &self,
        src: Rank,
        tag: u64,
        expected_size: usize,
    ) -> Result<PooledBuf> {
        let peer = self.peer(src)?;
        let tagged_bulk: Option<Arc<dyn crate::transport::TaggedBulkTransport>> = peer
            .extension::<Arc<dyn crate::transport::TaggedBulkTransport>>()?
            .map(|b| Arc::clone(&*b));
        if let Some(bulk) = tagged_bulk {
            let wire_tag = LaneKey::sidecar_tag(self.comm_id, tag);
            if let Ok(data) = bulk.recv_bulk_tagged(wire_tag, expected_size).await {
                return Ok(PooledBuf::from_vec(data, Arc::clone(&self._pool)));
            }
        }
        self.recv_bytes_tagged(src, tag).await
    }

    /// Send tagged bytes using the best available transport.
    pub(crate) async fn send_bytes_tagged_best_effort(
        &self,
        dest: Rank,
        tag: u64,
        data: &[u8],
    ) -> Result<()> {
        self.peer(dest)?
            .send_lane_best_effort(self.tagged_lane(tag), data)
            .await
    }

    /// Receive tagged raw bytes from a peer.
    ///
    /// Waits for the next message from `src` with `tag`, in send order.
    /// Messages with other tags stay queued for their own receivers, and a
    /// message that arrives before its receiver waits for it. After `src`
    /// disconnects and no message of `tag` is left, returns
    /// [`NexarError::PeerDisconnected`](crate::NexarError::PeerDisconnected).
    pub async fn recv_bytes_tagged(&self, src: Rank, tag: u64) -> Result<PooledBuf> {
        self.router(src)?
            .recv_lane(self.tagged_lane(tag), src)
            .await
    }

    /// Receive raw bytes using the best available transport.
    ///
    /// Tries `BulkTransport::recv_bulk` first (e.g., RDMA), falling back to QUIC.
    /// Only works for the default communicator (comm_id 0) and requires knowing
    /// the expected size.
    pub(crate) async fn recv_bytes_best_effort(
        &self,
        src: Rank,
        expected_size: usize,
    ) -> Result<PooledBuf> {
        if self.comm_id == 0 {
            let peer = self.peer(src)?;
            // Try BulkTransport recv_bulk if available.
            let bulk: Option<Arc<dyn crate::transport::BulkTransport>> = peer
                .extension::<Arc<dyn crate::transport::BulkTransport>>()?
                .map(|b| Arc::clone(&*b));
            if let Some(bulk) = bulk {
                match bulk.recv_bulk(expected_size).await {
                    Ok(data) => {
                        return Ok(PooledBuf::from_vec(data, Arc::clone(&self._pool)));
                    }
                    Err(e) => {
                        tracing::warn!(
                            src,
                            expected_size,
                            error = %e,
                            "bulk transport recv failed, falling back to QUIC"
                        );
                    }
                }
            }
        }
        // Fallback to QUIC recv.
        self.recv_bytes(src).await
    }

    /// Receive raw bytes from a peer, in send order.
    ///
    /// Uses comm-aware recv for split communicators.
    pub async fn recv_bytes(&self, src: Rank) -> Result<PooledBuf> {
        self.router(src)?.recv_lane(self.raw_lane(), src).await
    }
}

#[cfg(test)]
mod tests {
    use crate::NexarConfig;
    use crate::client::bootstrap::{local_pair, local_pair_with, shutdown_pair};
    use crate::error::NexarError;
    use std::sync::Arc;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(20);

    /// Payload `[tag, index LE...]`, so a receiver can check tag and order.
    fn frame(tag: u64, i: u32) -> Vec<u8> {
        let mut v = vec![tag as u8];
        v.extend_from_slice(&i.to_le_bytes());
        v
    }

    fn index(buf: &[u8]) -> u32 {
        u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]])
    }

    #[tokio::test]
    async fn test_receiver_of_one_tag_never_takes_another() {
        let (a, b) = local_pair(16).await;
        a.send_bytes_tagged(1, 10, &frame(10, 0)).await.unwrap();
        a.send_bytes_tagged(1, 20, &frame(20, 0)).await.unwrap();
        // A receive on an unused tag times out and consumes nothing.
        let idle = tokio::time::timeout(Duration::from_millis(200), b.recv_bytes_tagged(0, 30));
        assert!(idle.await.is_err());
        // Receive in the reverse of send order.
        let got20 = b.recv_bytes_tagged(0, 20).await.unwrap();
        let got10 = b.recv_bytes_tagged(0, 10).await.unwrap();
        assert_eq!((got20[0], got10[0]), (20, 10));
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_interleaved_tags_arrive_in_send_order_per_tag() {
        let (a, b) = local_pair(256).await;
        let (a, b) = (Arc::new(a), Arc::new(b));
        let tags = [3u64, 4, 5, 6];
        let per_tag = 100u32;
        let sender = {
            let a = Arc::clone(&a);
            tokio::spawn(async move {
                for i in 0..per_tag {
                    for &t in &tags {
                        a.send_bytes_tagged(1, t, &frame(t, i)).await.unwrap();
                    }
                    a.send_bytes(1, &frame(0, i)).await.unwrap();
                }
            })
        };
        let mut receivers: Vec<_> = tags
            .iter()
            .map(|&t| {
                let b = Arc::clone(&b);
                tokio::spawn(async move {
                    for want in 0..per_tag {
                        let buf = b.recv_bytes_tagged(0, t).await.unwrap();
                        assert_eq!(u64::from(buf[0]), t, "message of another tag");
                        assert_eq!(index(&buf), want, "tag {t} out of order");
                    }
                })
            })
            .collect();
        receivers.push({
            let b = Arc::clone(&b);
            tokio::spawn(async move {
                for want in 0..per_tag {
                    assert_eq!(index(&b.recv_bytes(0).await.unwrap()), want);
                }
            })
        });
        for t in receivers.into_iter().chain([sender]) {
            tokio::time::timeout(WAIT, t).await.unwrap().unwrap();
        }
        let (a, b) = (Arc::into_inner(a).unwrap(), Arc::into_inner(b).unwrap());
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_many_tags_depth_one_never_deadlock() {
        let (a, b) = local_pair(1).await;
        let (a, b) = (Arc::new(a), Arc::new(b));
        let tags: Vec<u64> = (1..=16).collect();
        let per_tag = 20u32;
        let sender = {
            let a = Arc::clone(&a);
            let tags = tags.clone();
            tokio::spawn(async move {
                for i in 0..per_tag {
                    for &t in &tags {
                        a.send_bytes_tagged(1, t, &frame(t, i)).await.unwrap();
                    }
                }
            })
        };
        // Receivers drain in reverse tag order, at different paces.
        let receivers: Vec<_> = tags
            .iter()
            .rev()
            .map(|&t| {
                let b = Arc::clone(&b);
                tokio::spawn(async move {
                    for want in 0..per_tag {
                        if t % 3 == 0 {
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                        assert_eq!(index(&b.recv_bytes_tagged(0, t).await.unwrap()), want);
                    }
                })
            })
            .collect();
        for t in receivers.into_iter().chain([sender]) {
            tokio::time::timeout(WAIT, t).await.unwrap().unwrap();
        }
        let (a, b) = (Arc::into_inner(a).unwrap(), Arc::into_inner(b).unwrap());
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_full_tag_does_not_stall_another_tag() {
        let (a, b) = local_pair(2).await;
        // Tag 1 gets far more messages than its queue holds; nobody reads it yet.
        for i in 0..32 {
            a.send_bytes_tagged(1, 1, &frame(1, i)).await.unwrap();
        }
        a.send_bytes_tagged(1, 2, &frame(2, 0)).await.unwrap();
        let other = tokio::time::timeout(WAIT, b.recv_bytes_tagged(0, 2)).await;
        assert_eq!(other.expect("tag 2 stalled behind tag 1").unwrap()[0], 2);
        for want in 0..32 {
            assert_eq!(index(&b.recv_bytes_tagged(0, 1).await.unwrap()), want);
        }
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_cancelled_send_does_not_block_later_messages() {
        let (a, b) = local_pair(16).await;
        // Larger than one QUIC stream window, so the write cannot finish
        // before the receiver reads: the timeout cancels it mid-payload.
        let big = vec![0xAB; 128 * 1024 * 1024];
        let cut = tokio::time::timeout(Duration::from_millis(5), a.send_bytes_tagged(1, 3, &big));
        assert!(
            cut.await.is_err(),
            "the large send finished before the timeout"
        );
        a.send_bytes_tagged(1, 3, &frame(3, 1)).await.unwrap();
        a.send_bytes_tagged(1, 4, &frame(4, 1)).await.unwrap();
        // Another lane passes the abandoned number.
        let other = tokio::time::timeout(WAIT, b.recv_bytes_tagged(0, 4)).await;
        assert_eq!(index(&other.expect("tag 4 stalled").unwrap()), 1);
        // The same lane delivers the next message.
        let same = tokio::time::timeout(WAIT, b.recv_bytes_tagged(0, 3)).await;
        assert_eq!(index(&same.expect("tag 3 stalled").unwrap()), 1);
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_disconnect_wakes_blocked_receivers_with_typed_error() {
        let (a, b) = local_pair(16).await;
        let b = Arc::new(b);
        let waiters: Vec<_> = [7u64, 8]
            .into_iter()
            .map(|t| {
                let b = Arc::clone(&b);
                tokio::spawn(async move { b.recv_bytes_tagged(0, t).await })
            })
            .collect();
        let raw = {
            let b = Arc::clone(&b);
            tokio::spawn(async move { b.recv_bytes(0).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        a.close();
        for w in waiters.into_iter().chain([raw]) {
            let got = tokio::time::timeout(WAIT, w).await.unwrap().unwrap();
            assert!(
                matches!(got, Err(NexarError::PeerDisconnected { rank: 0 })),
                "expected PeerDisconnected from rank 0"
            );
        }
        // A receive that starts after the disconnect fails at once.
        let late = tokio::time::timeout(WAIT, b.recv_bytes_tagged(0, 9)).await;
        assert!(matches!(
            late.unwrap(),
            Err(NexarError::PeerDisconnected { rank: 0 })
        ));
    }

    #[tokio::test]
    async fn test_over_limit_message_is_refused_and_later_messages_flow() {
        let (a, b) = local_pair_with(NexarConfig {
            max_frame_bytes: 1024,
            ..NexarConfig::default()
        })
        .await;
        // The refused stream is stopped before the router allocates for it, so
        // the send can fail or finish first. Either way nothing is delivered.
        let big = vec![0xCD; 8 * 1024 * 1024];
        let _ = a.send_bytes_tagged(1, 5, &big).await;
        a.send_bytes_tagged(1, 5, &frame(5, 1)).await.unwrap();
        a.send_bytes(1, &frame(0, 2)).await.unwrap();
        let next = tokio::time::timeout(WAIT, b.recv_bytes_tagged(0, 5)).await;
        let next = next
            .expect("lane stalled behind a refused message")
            .unwrap();
        assert_eq!(next.len(), 5, "the refused message was delivered");
        assert_eq!(index(&next), 1);
        let raw = tokio::time::timeout(WAIT, b.recv_bytes(0)).await;
        assert_eq!(index(&raw.expect("raw lane stalled").unwrap()), 2);
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_inflight_budget_slows_readers_without_losing_messages() {
        let (a, b) = local_pair_with(NexarConfig {
            recv_inflight_bytes: 1024 * 1024,
            ..NexarConfig::default()
        })
        .await;
        let (a, b) = (Arc::new(a), Arc::new(b));
        // Eight 600 KiB messages on eight lanes cannot read at once in a
        // 1 MiB budget. All arrive intact.
        let len = 600 * 1024;
        let senders: Vec<_> = (1..=8u64)
            .map(|t| {
                let a = Arc::clone(&a);
                tokio::spawn(async move {
                    a.send_bytes_tagged(1, t, &vec![t as u8; len])
                        .await
                        .unwrap();
                })
            })
            .collect();
        for t in 1..=8u64 {
            let got = tokio::time::timeout(WAIT, b.recv_bytes_tagged(0, t)).await;
            let got = got.expect("budget deadlocked a lane").unwrap();
            assert_eq!(got.len(), len);
            assert!(got.iter().all(|&x| u64::from(x) == t));
        }
        for s in senders {
            s.await.unwrap();
        }
        let (a, b) = (Arc::into_inner(a).unwrap(), Arc::into_inner(b).unwrap());
        shutdown_pair(a, b).await;
    }

    #[tokio::test]
    async fn test_message_larger_than_budget_passes_alone() {
        let (a, b) = local_pair_with(NexarConfig {
            recv_inflight_bytes: 1024 * 1024,
            ..NexarConfig::default()
        })
        .await;
        let len = 3 * 1024 * 1024;
        a.send_bytes_tagged(1, 9, &vec![7u8; len]).await.unwrap();
        let got = tokio::time::timeout(WAIT, b.recv_bytes_tagged(0, 9)).await;
        let got = got.expect("large message stalled").unwrap();
        assert_eq!(got.len(), len);
        assert!(got.iter().all(|&x| x == 7));
        shutdown_pair(a, b).await;
    }
}
