//! Per-key bounded queues with selective receive.

use super::notify_cell::NotifyCell;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// Queues messages by key. A receiver takes only messages of its key.
///
/// Each key holds at most `depth` messages. A sender to a full key waits, and
/// senders to other keys go on. Messages of one key leave in the order they
/// entered. The mailbox does not order messages that arrive concurrently.
pub(crate) struct KeyedMailbox<K, V> {
    cell: NotifyCell<State<K, V>>,
    depth: usize,
}

struct State<K, V> {
    queues: HashMap<K, VecDeque<V>>,
    closed: bool,
}

impl<K: Eq + Hash + Clone, V> KeyedMailbox<K, V> {
    /// A mailbox whose keys hold `depth` messages each (at least 1).
    pub(crate) fn new(depth: usize) -> Self {
        Self {
            cell: NotifyCell::new(State {
                queues: HashMap::new(),
                closed: false,
            }),
            depth: depth.max(1),
        }
    }

    /// Queue `value` under `key`, waiting while that key is full.
    ///
    /// Returns `Err(value)` once the mailbox is closed. Cancel-safe.
    pub(crate) async fn send(&self, key: K, value: V) -> Result<(), V> {
        let depth = self.depth;
        let mut value = Some(value);
        self.cell
            .wait(|st| {
                if st.closed {
                    return Some(Err(value.take()?));
                }
                let queue = st.queues.entry(key.clone()).or_default();
                if queue.len() >= depth {
                    return None;
                }
                queue.push_back(value.take()?);
                Some(Ok(()))
            })
            .await
    }

    /// Take the oldest message of `key`, waiting until one arrives.
    ///
    /// Returns `None` once the mailbox is closed and `key` has no message
    /// left. Cancel-safe.
    pub(crate) async fn recv(&self, key: &K) -> Option<V> {
        self.cell
            .wait(|st| {
                let Some(queue) = st.queues.get_mut(key) else {
                    return st.closed.then_some(None);
                };
                let value = queue.pop_front();
                if queue.is_empty() {
                    st.queues.remove(key);
                }
                match value {
                    Some(v) => Some(Some(v)),
                    None => st.closed.then_some(None),
                }
            })
            .await
    }

    /// Refuse new messages and wake every waiter. Queued messages stay
    /// receivable.
    pub(crate) fn close(&self) {
        self.cell.update(|st| st.closed = true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_receiver_takes_only_its_key() {
        let mb = KeyedMailbox::new(4);
        mb.send(1u8, "a").await.unwrap();
        mb.send(2u8, "b").await.unwrap();
        assert_eq!(mb.recv(&2).await, Some("b"));
        assert_eq!(mb.recv(&1).await, Some("a"));
    }

    #[tokio::test]
    async fn test_full_key_does_not_block_other_keys() {
        let mb = Arc::new(KeyedMailbox::new(1));
        mb.send(1u8, 10).await.unwrap();
        let blocked = {
            let mb = Arc::clone(&mb);
            tokio::spawn(async move { mb.send(1u8, 11).await })
        };
        tokio::time::timeout(Duration::from_secs(5), mb.send(2u8, 20))
            .await
            .expect("send to another key waited on a full key")
            .unwrap();
        assert_eq!(mb.recv(&2).await, Some(20));
        assert_eq!(mb.recv(&1).await, Some(10));
        blocked.await.unwrap().unwrap();
        assert_eq!(mb.recv(&1).await, Some(11));
    }

    #[tokio::test]
    async fn test_close_drains_then_ends() {
        let mb = Arc::new(KeyedMailbox::new(2));
        mb.send(1u8, 1).await.unwrap();
        let waiter = {
            let mb = Arc::clone(&mb);
            tokio::spawn(async move { mb.recv(&9u8).await })
        };
        tokio::task::yield_now().await;
        mb.close();
        assert_eq!(waiter.await.unwrap(), None);
        assert_eq!(mb.recv(&1).await, Some(1));
        assert_eq!(mb.recv(&1).await, None);
        assert_eq!(mb.send(1u8, 2).await, Err(2));
    }
}
