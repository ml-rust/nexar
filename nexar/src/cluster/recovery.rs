//! Fault recovery orchestrator for nexar.
//!
//! Connects the `HealthMonitor` failure detection to `rebuild_excluding()`,
//! providing automatic communicator recovery when nodes fail.

use crate::client::NexarClient;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::types::{Priority, Rank};
use std::collections::BTreeSet;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast, watch};

/// Policy for handling detected node failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryPolicy {
    /// Automatically reach agreement and rebuild the communicator.
    Automatic,
    /// Notify the application but do not rebuild. The application must
    /// handle recovery itself.
    Manual,
    /// Abort: return an error when any node fails.
    Abort,
}

/// Notification sent to application code after a recovery event.
#[derive(Debug, Clone)]
pub struct RecoveryEvent {
    /// This node's rank before recovery.
    pub old_rank: Rank,
    /// This node's rank after recovery (contiguous re-numbering).
    pub new_rank: Rank,
    /// World size after recovery.
    pub new_world_size: u32,
    /// Ranks that were excluded during recovery.
    pub dead_ranks: Vec<Rank>,
}

/// Orchestrates fault detection, agreement, and communicator rebuild.
///
/// Wraps a `NexarClient` and watches for failures. On detection, it runs a
/// point-to-point agreement protocol among survivors (no working communicator
/// needed), rebuilds the communicator, and notifies subscribers.
pub struct RecoveryOrchestrator {
    client: Arc<Mutex<NexarClient>>,
    policy: RecoveryPolicy,
    event_tx: broadcast::Sender<RecoveryEvent>,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
}

impl RecoveryOrchestrator {
    /// Create a new orchestrator wrapping the given client.
    ///
    /// Returns the orchestrator and an initial event receiver.
    pub fn new(
        client: NexarClient,
        policy: RecoveryPolicy,
    ) -> (Self, broadcast::Receiver<RecoveryEvent>) {
        let (event_tx, event_rx) = broadcast::channel(16);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let orchestrator = Self {
            client: Arc::new(Mutex::new(client)),
            policy,
            event_tx,
            shutdown_tx,
            shutdown_rx,
        };
        (orchestrator, event_rx)
    }

    /// Subscribe to recovery events.
    pub fn subscribe(&self) -> broadcast::Receiver<RecoveryEvent> {
        self.event_tx.subscribe()
    }

    /// Access the current client. The client may change after recovery.
    pub fn client(&self) -> Arc<Mutex<NexarClient>> {
        Arc::clone(&self.client)
    }

    /// Signal the orchestrator to stop.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Run the orchestrator loop. Blocks until shutdown or an unrecoverable error.
    pub async fn run(&self) -> Result<()> {
        let mut failure_rx = {
            let client = self.client.lock().await;
            client.failure_watch()
        };
        let mut shutdown_rx = self.shutdown_rx.clone();

        loop {
            tokio::select! {
                res = shutdown_rx.changed() => {
                    if res.is_ok() && *shutdown_rx.borrow() {
                        return Ok(());
                    }
                    continue;
                }
                res = failure_rx.changed() => {
                    if res.is_err() {
                        let client = self.client.lock().await;
                        failure_rx = client.failure_watch();
                        continue;
                    }
                }
            }

            let dead_ranks: Vec<Rank> = failure_rx.borrow_and_update().clone();
            if dead_ranks.is_empty() {
                continue;
            }

            match self.policy {
                RecoveryPolicy::Abort => {
                    return Err(NexarError::Recovery {
                        dead_ranks,
                        message: "node failure detected and policy is Abort".into(),
                    });
                }
                RecoveryPolicy::Manual => {
                    let client = self.client.lock().await;
                    let _ = self.event_tx.send(RecoveryEvent {
                        old_rank: client.rank(),
                        new_rank: client.rank(),
                        new_world_size: client.world_size(),
                        dead_ranks,
                    });
                }
                RecoveryPolicy::Automatic => {
                    self.run_automatic_recovery(dead_ranks).await?;
                    let client = self.client.lock().await;
                    failure_rx = client.failure_watch();
                }
            }
        }
    }

    /// Execute the automatic recovery protocol with cascading failure support.
    ///
    /// Each agreement round is raced against `failure_watch.changed()`. If the
    /// health monitor detects a new failure mid-agreement, the round is aborted
    /// instantly (no timeout wait) and restarted with the updated dead set.
    async fn run_automatic_recovery(&self, initial_dead: Vec<Rank>) -> Result<()> {
        let mut dead_set: BTreeSet<Rank> = initial_dead.into_iter().collect();

        loop {
            let client = self.client.lock().await;
            let my_rank = client.rank();
            let world_size = client.world_size();
            let recovery_timeout = client.config.recovery_timeout;
            let mut failure_rx = client.failure_watch();
            // Mark current state as seen so changed() only fires for NEW failures.
            failure_rx.borrow_and_update();

            let all_ranks: BTreeSet<Rank> = (0..world_size).collect();
            let alive: BTreeSet<Rank> = all_ranks.difference(&dead_set).copied().collect();
            let leader = *alive.iter().next().ok_or_else(|| NexarError::Recovery {
                dead_ranks: dead_set.iter().copied().collect(),
                message: "no survivors remaining".into(),
            })?;
            let dead_vec: Vec<Rank> = dead_set.iter().copied().collect();

            drop(client);

            // Race the entire agreement round against failure_watch.
            // If a new failure is detected, abort instantly and retry.
            let agreement_result = {
                let agreement_fut =
                    self.run_agreement(my_rank, leader, &alive, &dead_vec, recovery_timeout);
                tokio::select! {
                    result = agreement_fut => result,
                    _ = failure_rx.changed() => {
                        // New failure detected — merge and retry.
                        let latest: BTreeSet<Rank> =
                            failure_rx.borrow().iter().copied().collect();
                        dead_set.extend(latest);
                        tracing::warn!(
                            "new failure detected during agreement, restarting with {:?}",
                            dead_set
                        );
                        continue;
                    }
                }
            };

            let agreed_dead = match agreement_result {
                Ok(agreed) => agreed,
                Err(NexarError::PeerDisconnected { rank }) => {
                    tracing::warn!(
                        rank,
                        "peer disconnected during recovery agreement, retrying"
                    );
                    dead_set.insert(rank);
                    continue;
                }
                Err(e) => return Err(e),
            };

            let client = self.client.lock().await;
            let old_rank = client.rank();
            match client.rebuild_excluding(&agreed_dead).await {
                Ok(new_client) => {
                    let new_rank = new_client.rank();
                    let new_world_size = new_client.world_size();

                    drop(client);
                    {
                        let mut slot = self.client.lock().await;
                        *slot = new_client;
                    }

                    let _ = self.event_tx.send(RecoveryEvent {
                        old_rank,
                        new_rank,
                        new_world_size,
                        dead_ranks: agreed_dead,
                    });
                    return Ok(());
                }
                Err(NexarError::PeerDisconnected { rank }) => {
                    drop(client);
                    tracing::warn!(rank, "peer disconnected during rebuild, retrying");
                    dead_set.insert(rank);
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Run the point-to-point agreement protocol.
    async fn run_agreement(
        &self,
        my_rank: Rank,
        leader: Rank,
        alive: &BTreeSet<Rank>,
        my_dead_view: &[Rank],
        timeout: std::time::Duration,
    ) -> Result<Vec<Rank>> {
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let client = self.client.lock().await;

        if my_rank == leader {
            run_leader_agreement(&client, epoch, alive, my_dead_view, timeout).await
        } else {
            run_follower_agreement(&client, epoch, leader, my_dead_view, timeout).await
        }
    }
}

/// Leader: collect votes from all other survivors, union dead sets, broadcast agreement.
async fn run_leader_agreement(
    client: &NexarClient,
    epoch: u64,
    alive: &BTreeSet<Rank>,
    my_dead_view: &[Rank],
    timeout: std::time::Duration,
) -> Result<Vec<Rank>> {
    let mut union_dead: BTreeSet<Rank> = my_dead_view.iter().copied().collect();
    let my_rank = client.rank();

    let other_survivors: Vec<Rank> = alive.iter().copied().filter(|&r| r != my_rank).collect();
    for &src in &other_survivors {
        let vote_dead = tokio::time::timeout(timeout, recv_recovery_vote(client, src))
            .await
            .map_err(|_| NexarError::Recovery {
                dead_ranks: union_dead.iter().copied().collect(),
                message: format!("timeout waiting for recovery vote from rank {src}"),
            })??;
        union_dead.extend(vote_dead);
    }

    let agreed: Vec<Rank> = union_dead.into_iter().collect();

    let agreement = NexarMessage::RecoveryAgreement {
        epoch,
        dead_ranks: agreed.clone(),
    };
    for &dst in &other_survivors {
        let peer = client.peer(dst)?;
        peer.send_message(&agreement, Priority::Critical).await?;
    }

    Ok(agreed)
}

/// Follower: send vote to leader, wait for agreement.
async fn run_follower_agreement(
    client: &NexarClient,
    epoch: u64,
    leader: Rank,
    my_dead_view: &[Rank],
    timeout: std::time::Duration,
) -> Result<Vec<Rank>> {
    let vote = NexarMessage::RecoveryVote {
        epoch,
        dead_ranks: my_dead_view.to_vec(),
    };
    let leader_peer = client.peer(leader)?;
    leader_peer.send_message(&vote, Priority::Critical).await?;

    let agreed = tokio::time::timeout(timeout, recv_recovery_agreement(client, leader))
        .await
        .map_err(|_| NexarError::Recovery {
            dead_ranks: my_dead_view.to_vec(),
            message: format!("timeout waiting for recovery agreement from leader {leader}"),
        })??;

    Ok(agreed)
}

/// Receive a `RecoveryVote` from `src`, skipping non-recovery control messages.
async fn recv_recovery_vote(client: &NexarClient, src: Rank) -> Result<Vec<Rank>> {
    loop {
        let msg = client.recv_control(src).await?;
        match msg {
            NexarMessage::RecoveryVote { dead_ranks, .. } => return Ok(dead_ranks),
            NexarMessage::Heartbeat { .. } | NexarMessage::NodeLeft { .. } => continue,
            other => {
                tracing::warn!(src, "expected RecoveryVote, got {other:?}, skipping");
            }
        }
    }
}

/// Receive a `RecoveryAgreement` from `leader`, skipping non-recovery control messages.
async fn recv_recovery_agreement(client: &NexarClient, leader: Rank) -> Result<Vec<Rank>> {
    loop {
        let msg = client.recv_control(leader).await?;
        match msg {
            NexarMessage::RecoveryAgreement { dead_ranks, .. } => return Ok(dead_ranks),
            NexarMessage::Heartbeat { .. } | NexarMessage::NodeLeft { .. } => continue,
            other => {
                tracing::warn!(
                    leader,
                    "expected RecoveryAgreement, got {other:?}, skipping"
                );
            }
        }
    }
}
