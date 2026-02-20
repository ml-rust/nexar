use crate::error::Result;
use crate::types::Rank;

use super::NexarClient;

impl NexarClient {
    /// Rebuild the communicator excluding dead ranks.
    ///
    /// This is a **collective** operation: all surviving ranks must call it
    /// with the same `dead_ranks` list. Dead ranks are simply absent (they
    /// don't call this method).
    ///
    /// Returns a new `NexarClient` with contiguous ranks `[0, survivors)`
    /// and `world_size = survivors`. The relative order of surviving ranks
    /// is preserved.
    ///
    /// Internally uses `split()` with `color=0` for all survivors, and
    /// `key=self.rank` to preserve ordering.
    pub async fn rebuild_excluding(&self, dead_ranks: &[Rank]) -> Result<NexarClient> {
        // All surviving ranks join color 0, ordered by their current rank.
        // Dead ranks simply never call split(), so they're excluded.
        debug_assert!(
            !dead_ranks.contains(&self.rank),
            "a dead rank should not call rebuild_excluding"
        );

        self.split(0, self.rank).await
    }
}
