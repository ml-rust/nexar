use crate::error::Result;
use std::future::Future;
use tokio::task::JoinHandle;

/// A handle to a non-blocking collective operation.
///
/// The collective runs asynchronously in a spawned task. Call `wait()` to
/// block until it completes, or check `is_finished()` to poll.
///
/// If dropped without calling `wait()`, the background task is aborted to
/// prevent writes to potentially-freed memory.
pub struct CollectiveHandle {
    inner: Option<JoinHandle<Result<()>>>,
}

impl CollectiveHandle {
    /// Spawn a future as a non-blocking collective and return a handle.
    pub(crate) fn spawn(fut: impl Future<Output = Result<()>> + Send + 'static) -> Self {
        Self {
            inner: Some(tokio::spawn(fut)),
        }
    }

    /// Wait for the collective to complete and propagate any error.
    pub async fn wait(mut self) -> Result<()> {
        let handle = self
            .inner
            .take()
            .expect("CollectiveHandle already consumed");
        handle.await.map_err(|e| {
            crate::error::NexarError::transport(format!("collective task panicked: {e}"))
        })?
    }

    /// Check if the collective has finished (non-blocking).
    pub fn is_finished(&self) -> bool {
        self.inner.as_ref().is_none_or(|h| h.is_finished())
    }
}

impl Drop for CollectiveHandle {
    fn drop(&mut self) {
        if let Some(handle) = &self.inner {
            handle.abort();
        }
    }
}

/// A group of non-blocking collectives that can be waited on together.
pub struct CollectiveGroup {
    handles: Vec<CollectiveHandle>,
}

impl CollectiveGroup {
    /// Create an empty group.
    pub fn new() -> Self {
        Self {
            handles: Vec::new(),
        }
    }

    /// Add a handle to the group.
    pub fn push(&mut self, h: CollectiveHandle) {
        self.handles.push(h);
    }

    /// Wait for all collectives in the group to complete.
    ///
    /// Returns the first error encountered, if any. All tasks are awaited
    /// regardless of errors.
    pub async fn wait_all(self) -> Result<()> {
        let mut first_err = None;
        for h in self.handles {
            if let Err(e) = h.wait().await
                && first_err.is_none()
            {
                first_err = Some(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl Default for CollectiveGroup {
    fn default() -> Self {
        Self::new()
    }
}
