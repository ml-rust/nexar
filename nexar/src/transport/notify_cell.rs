//! Shared state behind a blocking mutex, with async waits on its changes.

use std::sync::{Mutex, MutexGuard};
use tokio::sync::Notify;

/// State that tasks change under a lock and wait on without holding the lock.
///
/// Every [`update`](Self::update) wakes every waiter, and each waiter checks its
/// own condition again. The lock is never held across an `.await`.
pub(crate) struct NotifyCell<S> {
    state: Mutex<S>,
    changed: Notify,
}

impl<S> NotifyCell<S> {
    pub(crate) fn new(state: S) -> Self {
        Self {
            state: Mutex::new(state),
            changed: Notify::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, S> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Read or change the state, then wake every waiter.
    pub(crate) fn update<R>(&self, f: impl FnOnce(&mut S) -> R) -> R {
        let out = f(&mut self.lock());
        self.changed.notify_waiters();
        out
    }

    /// Wait until `f` returns `Some`, and return that value.
    ///
    /// `f` runs under the lock after each change. A `Some` result counts as a
    /// change and wakes the other waiters. Cancel-safe: a dropped wait changes
    /// nothing.
    pub(crate) async fn wait<R>(&self, mut f: impl FnMut(&mut S) -> Option<R>) -> R {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            // Register before the check, so a change between the check and
            // the await still wakes this task.
            notified.as_mut().enable();
            let found = f(&mut self.lock());
            if let Some(out) = found {
                self.changed.notify_waiters();
                return out;
            }
            notified.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_wait_sees_later_update() {
        let cell = Arc::new(NotifyCell::new(0u32));
        let waiter = {
            let cell = Arc::clone(&cell);
            tokio::spawn(async move { cell.wait(|n| (*n >= 3).then_some(*n)).await })
        };
        for _ in 0..3 {
            tokio::task::yield_now().await;
            cell.update(|n| *n += 1);
        }
        assert_eq!(waiter.await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_wait_returns_at_once_when_ready() {
        let cell = NotifyCell::new(7u32);
        assert_eq!(cell.wait(|n| Some(*n)).await, 7);
    }
}
