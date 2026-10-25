//! Receive side of sequenced messages from one peer.
//!
//! The sender numbers every sequenced message to this peer with one counter.
//! QUIC carries each message on its own stream, so headers arrive in any
//! order. The inbox admits a message into its lane only when every lower
//! number is known, so each lane delivers in sender order.

use super::lane::LaneKey;
use super::notify_cell::NotifyCell;
use crate::transport::buffer_pool::PooledBuf;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;

/// How far past the watermark a sequence number can lie.
///
/// A sender has at most one QUIC stream limit of messages in flight, so an
/// honest gap stays far below this. A peer that names a number past it
/// cannot grow the inbox: the number is refused.
const MAX_AHEAD: u64 = 1 << 16;

/// Per-lane FIFO queues over the sequenced messages of one peer.
///
/// Each lane holds at most `depth` admitted messages. A message past that
/// waits before its payload is read, so a full lane holds back only its own
/// streams. A lane exists only while it holds or awaits a message.
pub(crate) struct SeqInbox {
    cell: NotifyCell<State>,
    depth: usize,
}

struct State {
    /// Every sequence number below this one is known.
    watermark: u64,
    /// Known sequence numbers at or above `watermark`.
    known_ahead: BTreeSet<u64>,
    lanes: HashMap<LaneKey, Lane>,
    closed: bool,
}

#[derive(Default)]
struct Lane {
    /// Admitted messages in sequence order. `None`: payload still in flight.
    queue: VecDeque<(u64, Option<PooledBuf>)>,
    /// Announced messages that wait for admission.
    waiting: BTreeSet<u64>,
}

impl State {
    /// Record `seq` as known. Returns `false` when it was known already or
    /// lies more than `MAX_AHEAD` past the watermark.
    fn mark_known(&mut self, seq: u64) -> bool {
        if seq < self.watermark
            || seq - self.watermark >= MAX_AHEAD
            || !self.known_ahead.insert(seq)
        {
            return false;
        }
        while self.known_ahead.remove(&self.watermark) {
            self.watermark = self.watermark.saturating_add(1);
        }
        true
    }

    fn drop_lane_if_idle(&mut self, lane: LaneKey) {
        if self
            .lanes
            .get(&lane)
            .is_some_and(|l| l.queue.is_empty() && l.waiting.is_empty())
        {
            self.lanes.remove(&lane);
        }
    }
}

impl SeqInbox {
    pub(crate) fn new(depth: usize) -> Self {
        Self {
            cell: NotifyCell::new(State {
                watermark: 0,
                known_ahead: BTreeSet::new(),
                lanes: HashMap::new(),
                closed: false,
            }),
            depth: depth.max(1),
        }
    }

    /// Record sequence numbers the sender abandoned. Their messages, if they
    /// arrive later, are dropped.
    pub(crate) fn skip(&self, seqs: &[u64]) {
        self.cell.update(|st| {
            for &seq in seqs {
                st.mark_known(seq);
            }
        });
    }

    /// Record the header of message `seq` on `lane`.
    ///
    /// Returns `None` when `seq` was skipped, known already, or too far
    /// ahead of the lower numbers: drop the stream unread.
    pub(crate) fn announce(self: &Arc<Self>, lane: LaneKey, seq: u64) -> Option<Pending> {
        let fresh = self.cell.update(|st| {
            let fresh = st.mark_known(seq);
            if fresh {
                st.lanes.entry(lane).or_default().waiting.insert(seq);
            }
            fresh
        });
        fresh.then(|| Pending {
            inbox: Arc::clone(self),
            lane,
            seq,
            stage: Stage::Waiting,
        })
    }

    /// Take the next message of `lane`, waiting until it is complete.
    ///
    /// Returns `None` once the inbox is closed and `lane` has no complete
    /// message left. Cancel-safe.
    pub(crate) async fn recv(&self, lane: LaneKey) -> Option<PooledBuf> {
        self.cell
            .wait(|st| {
                let front_ready = st
                    .lanes
                    .get(&lane)
                    .and_then(|l| l.queue.front())
                    .is_some_and(|(_, buf)| buf.is_some());
                if !front_ready {
                    return st.closed.then_some(None);
                }
                let buf = st
                    .lanes
                    .get_mut(&lane)
                    .and_then(|l| l.queue.pop_front())
                    .and_then(|(_, buf)| buf);
                st.drop_lane_if_idle(lane);
                Some(buf)
            })
            .await
    }

    /// Wake every receiver. Complete messages stay receivable.
    pub(crate) fn close(&self) {
        self.cell.update(|st| st.closed = true);
    }
}

enum Stage {
    Waiting,
    Admitted,
    Filled,
}

/// One announced message on its way into its lane.
///
/// Dropping it before [`fill`](Self::fill) removes the message: later
/// messages of the lane no longer wait for it.
pub(crate) struct Pending {
    inbox: Arc<SeqInbox>,
    lane: LaneKey,
    seq: u64,
    stage: Stage,
}

impl Pending {
    /// Wait until the message may take a place in its lane.
    ///
    /// That needs every lower sequence number known, every earlier message of
    /// the lane admitted, and room in the lane. Cancel-safe.
    pub(crate) async fn admitted(&mut self) {
        let (lane, seq, depth) = (self.lane, self.seq, self.inbox.depth);
        self.inbox
            .cell
            .wait(|st| {
                let below_watermark = seq < st.watermark;
                let l = st.lanes.get_mut(&lane)?;
                let first = l.waiting.first() == Some(&seq);
                if !(below_watermark && first && l.queue.len() < depth) {
                    return None;
                }
                l.waiting.remove(&seq);
                l.queue.push_back((seq, None));
                Some(())
            })
            .await;
        self.stage = Stage::Admitted;
    }

    /// Hand the complete payload to the lane.
    pub(crate) fn fill(mut self, buf: PooledBuf) {
        let (lane, seq) = (self.lane, self.seq);
        self.inbox.cell.update(|st| {
            if let Some(slot) = st
                .lanes
                .get_mut(&lane)
                .and_then(|l| l.queue.iter_mut().find(|(s, _)| *s == seq))
            {
                slot.1 = Some(buf);
            }
        });
        self.stage = Stage::Filled;
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        let (lane, seq) = (self.lane, self.seq);
        match self.stage {
            Stage::Filled => {}
            Stage::Waiting => self.inbox.cell.update(|st| {
                if let Some(l) = st.lanes.get_mut(&lane) {
                    l.waiting.remove(&seq);
                }
                st.drop_lane_if_idle(lane);
            }),
            Stage::Admitted => self.inbox.cell.update(|st| {
                if let Some(l) = st.lanes.get_mut(&lane) {
                    l.queue.retain(|(s, _)| *s != seq);
                }
                st.drop_lane_if_idle(lane);
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::buffer_pool::BufferPool;
    use std::time::Duration;

    const A: LaneKey = LaneKey::Tagged { comm_id: 0, tag: 1 };
    const B: LaneKey = LaneKey::Tagged { comm_id: 0, tag: 2 };
    const WAIT: Duration = Duration::from_secs(5);

    fn buf(byte: u8) -> PooledBuf {
        PooledBuf::from_vec(vec![byte], BufferPool::new())
    }

    /// Announce, admit, and fill one message, as the router does.
    async fn deliver(inbox: &Arc<SeqInbox>, lane: LaneKey, seq: u64, byte: u8) {
        let mut p = inbox.announce(lane, seq).expect("not skipped");
        tokio::time::timeout(WAIT, p.admitted())
            .await
            .expect("admission stalled");
        p.fill(buf(byte));
    }

    async fn take(inbox: &SeqInbox, lane: LaneKey) -> u8 {
        let got = tokio::time::timeout(WAIT, inbox.recv(lane)).await;
        got.expect("recv stalled").expect("inbox closed")[0]
    }

    #[tokio::test]
    async fn test_out_of_order_headers_reach_the_right_lanes_in_order() {
        let inbox = Arc::new(SeqInbox::new(8));
        // Sender order: A0 B1 A2 B3. Arrival order: B3 A2 B1 A0.
        let tasks: Vec<_> = [(B, 3, 13), (A, 2, 12), (B, 1, 11), (A, 0, 10)]
            .into_iter()
            .map(|(lane, seq, byte)| {
                let inbox = Arc::clone(&inbox);
                tokio::spawn(async move { deliver(&inbox, lane, seq, byte).await })
            })
            .collect();
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(take(&inbox, B).await, 11);
        assert_eq!(take(&inbox, B).await, 13);
        assert_eq!(take(&inbox, A).await, 10);
        assert_eq!(take(&inbox, A).await, 12);
    }

    #[tokio::test]
    async fn test_later_message_waits_for_unknown_earlier_number() {
        let inbox = Arc::new(SeqInbox::new(8));
        let mut later = inbox.announce(A, 1).unwrap();
        let blocked = tokio::time::timeout(Duration::from_millis(50), later.admitted()).await;
        assert!(blocked.is_err(), "seq 1 admitted before seq 0 was known");
        deliver(&inbox, B, 0, 1).await;
        tokio::time::timeout(WAIT, later.admitted()).await.unwrap();
        later.fill(buf(2));
        assert_eq!(take(&inbox, A).await, 2);
    }

    #[tokio::test]
    async fn test_skip_releases_later_messages_and_drops_late_header() {
        let inbox = Arc::new(SeqInbox::new(8));
        inbox.skip(&[0]);
        deliver(&inbox, A, 1, 7).await;
        assert_eq!(take(&inbox, A).await, 7);
        assert!(inbox.announce(A, 0).is_none(), "skipped seq was accepted");
        // A skip for a number already announced changes nothing.
        deliver(&inbox, A, 2, 8).await;
        inbox.skip(&[2]);
        assert_eq!(take(&inbox, A).await, 8);
    }

    #[tokio::test]
    async fn test_far_future_numbers_are_refused() {
        let inbox = Arc::new(SeqInbox::new(8));
        assert!(inbox.announce(A, u64::MAX).is_none());
        assert!(inbox.announce(A, MAX_AHEAD).is_none());
        inbox.skip(&[u64::MAX, MAX_AHEAD]);
        assert!(inbox.announce(A, MAX_AHEAD - 1).is_some());
        deliver(&inbox, A, 0, 1).await;
        assert_eq!(take(&inbox, A).await, 1);
    }

    #[tokio::test]
    async fn test_dropped_pending_does_not_block_its_lane() {
        let inbox = Arc::new(SeqInbox::new(8));
        let mut lost = inbox.announce(A, 0).unwrap();
        lost.admitted().await;
        drop(lost); // payload read failed
        deliver(&inbox, A, 1, 5).await;
        assert_eq!(take(&inbox, A).await, 5);
    }

    #[tokio::test]
    async fn test_full_lane_holds_only_its_own_messages() {
        let inbox = Arc::new(SeqInbox::new(1));
        deliver(&inbox, A, 0, 1).await;
        let mut second_a = inbox.announce(A, 1).unwrap();
        let blocked = tokio::time::timeout(Duration::from_millis(50), second_a.admitted()).await;
        assert!(blocked.is_err(), "full lane admitted a second message");
        // Lane B proceeds although A is full and A1 waits.
        deliver(&inbox, B, 2, 3).await;
        assert_eq!(take(&inbox, B).await, 3);
        assert_eq!(take(&inbox, A).await, 1);
        tokio::time::timeout(WAIT, second_a.admitted())
            .await
            .unwrap();
        second_a.fill(buf(2));
        assert_eq!(take(&inbox, A).await, 2);
    }

    #[tokio::test]
    async fn test_many_lanes_depth_one_never_deadlock() {
        let inbox = Arc::new(SeqInbox::new(1));
        let lanes: Vec<LaneKey> = (0..8)
            .map(|t| LaneKey::Tagged { comm_id: 0, tag: t })
            .collect();
        let per_lane = 20u64;
        let n = lanes.len() as u64;
        let total = per_lane * n;
        // Round-robin over lanes; headers announced in reverse order.
        let senders: Vec<_> = (0..total)
            .rev()
            .map(|seq| {
                let inbox = Arc::clone(&inbox);
                let lane = lanes[(seq % n) as usize];
                tokio::spawn(async move {
                    let mut p = inbox.announce(lane, seq).unwrap();
                    p.admitted().await;
                    p.fill(buf((seq / n) as u8));
                })
            })
            .collect();
        let receivers: Vec<_> = lanes
            .iter()
            .map(|&lane| {
                let inbox = Arc::clone(&inbox);
                tokio::spawn(async move {
                    for want in 0..per_lane {
                        assert_eq!(take(&inbox, lane).await, want as u8);
                    }
                })
            })
            .collect();
        for t in receivers.into_iter().chain(senders) {
            tokio::time::timeout(WAIT, t).await.unwrap().unwrap();
        }
    }

    #[tokio::test]
    async fn test_close_wakes_receivers_after_draining() {
        let inbox = Arc::new(SeqInbox::new(4));
        deliver(&inbox, A, 0, 9).await;
        let waiter = {
            let inbox = Arc::clone(&inbox);
            tokio::spawn(async move { inbox.recv(B).await })
        };
        tokio::task::yield_now().await;
        inbox.close();
        assert!(waiter.await.unwrap().is_none());
        assert_eq!(take(&inbox, A).await, 9);
        assert!(inbox.recv(A).await.is_none());
    }
}
