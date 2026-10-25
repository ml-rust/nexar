//! Budget of payload bytes a peer's router reads at once.
//!
//! A reader reserves its payload size before it allocates the buffer, so a
//! peer that announces many large messages cannot make the router allocate
//! them all. A reader that cannot reserve waits with its payload unread in the
//! QUIC stream, and QUIC flow control slows the sender.

use tokio::sync::{Semaphore, SemaphorePermit};

/// Bytes one permit stands for. A reservation rounds up to whole units.
const UNIT: u64 = 4096;

pub(crate) struct ByteBudget {
    sem: Semaphore,
    units: u32,
}

impl ByteBudget {
    /// A budget of `bytes`, at least one unit.
    pub(crate) fn new(bytes: usize) -> Self {
        let units = (bytes as u64).div_ceil(UNIT).clamp(1, u64::from(u32::MAX)) as u32;
        Self {
            sem: Semaphore::new(units as usize),
            units,
        }
    }

    /// Reserve `len` bytes, waiting for room. Cancel-safe.
    ///
    /// A message larger than the whole budget reserves all of it and so reads
    /// alone. The semaphore is never closed, so `None` does not occur.
    pub(crate) async fn reserve(&self, len: u64) -> Option<SemaphorePermit<'_>> {
        let want = len.div_ceil(UNIT).clamp(1, u64::from(self.units)) as u32;
        self.sem.acquire_many(want).await.ok()
    }

    /// Bytes not reserved, rounded down to whole units.
    #[cfg(test)]
    pub(crate) fn available(&self) -> u64 {
        self.sem.available_permits() as u64 * UNIT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_reservation_waits_for_room_then_proceeds() {
        let budget = ByteBudget::new(8 * 4096);
        let first = budget.reserve(6 * 4096).await.unwrap();
        assert_eq!(budget.available(), 2 * 4096);
        let blocked = tokio::time::timeout(Duration::from_millis(50), budget.reserve(4 * 4096));
        assert!(blocked.await.is_err(), "reserved past the budget");
        drop(first);
        let second = tokio::time::timeout(Duration::from_secs(5), budget.reserve(4 * 4096)).await;
        assert!(
            second
                .expect("freed room did not wake the waiter")
                .is_some()
        );
    }

    #[tokio::test]
    async fn test_message_over_budget_reads_alone() {
        let budget = ByteBudget::new(4 * 4096);
        let whole = budget.reserve(u64::MAX).await.unwrap();
        assert_eq!(budget.available(), 0);
        drop(whole);
        assert_eq!(budget.available(), 4 * 4096);
    }

    #[tokio::test]
    async fn test_empty_message_still_costs_one_unit() {
        let budget = ByteBudget::new(4096);
        let _held = budget.reserve(0).await.unwrap();
        assert_eq!(budget.available(), 0);
    }
}
