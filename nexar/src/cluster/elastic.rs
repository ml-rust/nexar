//! Elastic scaling: dynamic grow/shrink of a running nexar cluster.
//!
//! The `ElasticManager` wraps a `NexarClient` and coordinates resize operations
//! at explicit checkpoint boundaries called by the training loop.

use crate::client::NexarClient;
use crate::cluster::seed::PendingJoin;
use crate::config::NexarConfig;
use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::types::{Priority, Rank};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{Mutex, broadcast};

/// Configuration for elastic scaling.
#[derive(Debug, Clone)]
pub struct ElasticConfig {
    pub enabled: bool,
    pub min_world_size: u32,
    pub max_world_size: u32,
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_world_size: 1,
            max_world_size: 0,
        }
    }
}

/// Event emitted after a resize completes.
#[derive(Debug, Clone)]
pub struct ElasticEvent {
    pub old_world_size: u32,
    pub new_world_size: u32,
    pub new_rank: Rank,
    pub joined: Vec<Rank>,
    pub left: Vec<Rank>,
}

/// Result of [`NexarClient::bootstrap_elastic`].
///
/// Contains one `ElasticManager` per initial rank and the seed address
/// for adding nodes later. The seed's `accept_elastic()` loop runs as
/// a background tokio task tied to the `Arc<SeedNode>` lifetime.
pub struct ElasticBootstrap {
    pub managers: Vec<ElasticManager>,
    pub seed_addr: SocketAddr,
}

/// Manages elastic scaling for a single rank.
///
/// Wraps a `NexarClient` and coordinates cooperative resize barriers.
/// The training loop calls `elastic_checkpoint()` between steps to allow
/// pending joins/leaves to take effect.
pub struct ElasticManager {
    client: Arc<Mutex<NexarClient>>,
    pending_joins: Arc<StdMutex<Vec<PendingJoin>>>,
    pending_leaves: Arc<StdMutex<Vec<Rank>>>,
    checkpoint_epoch: AtomicU64,
    event_tx: broadcast::Sender<ElasticEvent>,
    config: ElasticConfig,
    nexar_config: Arc<NexarConfig>,
    /// TLS credentials for establishing mesh connections to new peers.
    ca_cert: Vec<u8>,
    my_cert: Vec<u8>,
    my_key: Vec<u8>,
    /// Seed address for add_nodes testing helper.
    seed_addr: Option<SocketAddr>,
}

impl Clone for ElasticManager {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            pending_joins: Arc::clone(&self.pending_joins),
            pending_leaves: Arc::clone(&self.pending_leaves),
            checkpoint_epoch: AtomicU64::new(self.checkpoint_epoch.load(Ordering::Relaxed)),
            event_tx: self.event_tx.clone(),
            config: self.config.clone(),
            nexar_config: Arc::clone(&self.nexar_config),
            ca_cert: self.ca_cert.clone(),
            my_cert: self.my_cert.clone(),
            my_key: self.my_key.clone(),
            seed_addr: self.seed_addr,
        }
    }
}

impl ElasticManager {
    /// Create a new elastic manager wrapping the given client.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: NexarClient,
        config: ElasticConfig,
        nexar_config: NexarConfig,
        ca_cert: Vec<u8>,
        my_cert: Vec<u8>,
        my_key: Vec<u8>,
        pending_joins: Arc<StdMutex<Vec<PendingJoin>>>,
        seed_addr: Option<SocketAddr>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(16);
        Self {
            client: Arc::new(Mutex::new(client)),
            pending_joins,
            pending_leaves: Arc::new(StdMutex::new(Vec::new())),
            checkpoint_epoch: AtomicU64::new(0),
            event_tx,
            config,
            nexar_config: Arc::new(nexar_config),
            ca_cert,
            my_cert,
            my_key,
            seed_addr,
        }
    }

    /// Subscribe to elastic events.
    pub fn subscribe(&self) -> broadcast::Receiver<ElasticEvent> {
        self.event_tx.subscribe()
    }

    /// Get a reference to the underlying client.
    pub fn client(&self) -> Arc<Mutex<NexarClient>> {
        Arc::clone(&self.client)
    }

    /// Cooperative elastic checkpoint.
    ///
    /// All ranks must call this between training steps. If there are pending
    /// join/leave requests, the resize is coordinated via rank 0. If there
    /// are no pending changes, this acts as a regular barrier.
    ///
    /// Returns `Ok(Some(event))` if a resize occurred, `Ok(None)` otherwise.
    pub async fn elastic_checkpoint(&self) -> Result<Option<ElasticEvent>> {
        let joining: Vec<PendingJoin> = {
            let mut pj = self.pending_joins.lock().unwrap_or_else(|p| p.into_inner());
            std::mem::take(&mut *pj)
        };
        let leaving: Vec<Rank> = {
            let mut pl = self
                .pending_leaves
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            std::mem::take(&mut *pl)
        };

        if joining.is_empty() && leaving.is_empty() {
            // No pending changes — run a regular barrier.
            let client = self.client.lock().await;
            client.barrier().await?;
            return Ok(None);
        }

        let (epoch, rank, old_world_size, timeout) = {
            let client = self.client.lock().await;
            let epoch = self.checkpoint_epoch.fetch_add(1, Ordering::Relaxed);
            (
                epoch,
                client.rank(),
                client.world_size(),
                self.nexar_config.elastic_checkpoint_timeout,
            )
        };

        let (joining_info, leaving_ranks, new_world_size) = if rank == 0 {
            self.coordinate_as_rank0(epoch, old_world_size, timeout, &joining, &leaving)
                .await?
        } else {
            self.participate_as_follower(epoch, timeout).await?
        };

        // Apply resize: rebuild the client with new topology.
        let event = self
            .apply_resize(
                old_world_size,
                new_world_size,
                rank,
                &joining_info,
                &leaving_ranks,
            )
            .await?;

        let _ = self.event_tx.send(event.clone());
        Ok(Some(event))
    }

    /// Rank 0: collect checkpoints from all peers and broadcast ack.
    async fn coordinate_as_rank0(
        &self,
        epoch: u64,
        old_world_size: u32,
        timeout: std::time::Duration,
        joining: &[PendingJoin],
        leaving: &[Rank],
    ) -> Result<(Vec<(Rank, String)>, Vec<Rank>, u32)> {
        let client = self.client.lock().await;

        // Collect ElasticCheckpoint from all existing ranks 1..world_size.
        for src_rank in 1..old_world_size {
            let msg = tokio::time::timeout(timeout, client.recv_control(src_rank))
                .await
                .map_err(|_| NexarError::ElasticTimeout {
                    epoch,
                    timeout_ms: timeout.as_millis() as u64,
                })??;

            match msg {
                NexarMessage::ElasticCheckpoint { epoch: e } if e == epoch => {}
                other => {
                    return Err(NexarError::Elastic(format!(
                        "expected ElasticCheckpoint(epoch={epoch}), got {other:?}"
                    )));
                }
            }
        }

        let joining_info: Vec<(Rank, String)> = joining
            .iter()
            .map(|pj| (pj.rank, pj.listen_addr.clone()))
            .collect();
        let new_world_size = old_world_size + joining.len() as u32 - leaving.len() as u32;

        let ack = NexarMessage::ElasticCheckpointAck {
            epoch,
            joining: joining_info.clone(),
            leaving: leaving.to_vec(),
            new_world_size,
        };

        // Send ack to all existing ranks (except self and leaving).
        for dest_rank in 1..old_world_size {
            if leaving.contains(&dest_rank) {
                continue;
            }
            let peer = client.peer(dest_rank)?;
            peer.send_message(&ack, Priority::Critical).await?;
        }

        Ok((joining_info, leaving.to_vec(), new_world_size))
    }

    /// Non-rank-0: send checkpoint to rank 0 and wait for ack.
    async fn participate_as_follower(
        &self,
        epoch: u64,
        timeout: std::time::Duration,
    ) -> Result<(Vec<(Rank, String)>, Vec<Rank>, u32)> {
        let client = self.client.lock().await;

        let checkpoint = NexarMessage::ElasticCheckpoint { epoch };
        let peer0 = client.peer(0)?;
        peer0.send_message(&checkpoint, Priority::Critical).await?;

        let ack_msg = tokio::time::timeout(timeout, client.recv_control(0))
            .await
            .map_err(|_| NexarError::ElasticTimeout {
                epoch,
                timeout_ms: timeout.as_millis() as u64,
            })??;

        match ack_msg {
            NexarMessage::ElasticCheckpointAck {
                epoch: e,
                joining,
                leaving,
                new_world_size,
            } if e == epoch => Ok((joining, leaving, new_world_size)),
            other => Err(NexarError::Elastic(format!(
                "expected ElasticCheckpointAck(epoch={epoch}), got {other:?}"
            ))),
        }
    }

    /// Apply resize: rebuild the client with the new topology and swap it in.
    async fn apply_resize(
        &self,
        old_world_size: u32,
        new_world_size: u32,
        rank: Rank,
        joined: &[(Rank, String)],
        left: &[Rank],
    ) -> Result<ElasticEvent> {
        let mut client = self.client.lock().await;

        if !joined.is_empty() {
            let new_peers: Vec<(Rank, Arc<crate::transport::PeerConnection>)> = Vec::new();
            // Note: In a full mesh scenario the joining nodes must first establish
            // P2P connections to all existing peers. For local bootstrap testing,
            // rebuild_adding creates the logical client with an expanded world size.
            // Real peer connections from the joining nodes will be wired in by the
            // bootstrap layer or by the join handshake that precedes the checkpoint.
            let rebuilt = client.rebuild_adding(new_peers).await;
            match rebuilt {
                Ok(new_client) => *client = new_client,
                Err(e) => {
                    tracing::warn!("rebuild_adding failed (no new peer connections): {e}");
                    // Even without new peer connections, update the event.
                    // The caller will see the joined ranks and can wire connections.
                }
            }
        }

        if !left.is_empty() {
            let left_ranks: Vec<Rank> = left.to_vec();
            let rebuilt = client.rebuild_excluding(&left_ranks).await?;
            *client = rebuilt;
        }

        Ok(ElasticEvent {
            old_world_size,
            new_world_size,
            new_rank: rank,
            joined: joined.iter().map(|(r, _)| *r).collect(),
            left: left.to_vec(),
        })
    }

    /// Queue a graceful departure for the given rank.
    pub fn remove_node(&self, rank: Rank) {
        self.pending_leaves
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(rank);
    }

    /// Add new nodes to the cluster (testing helper).
    ///
    /// Creates `count` new worker nodes that connect to the seed,
    /// receive credentials, and are queued as pending joins.
    pub async fn add_nodes(&self, count: u32) -> Result<Vec<PendingJoin>> {
        let seed_addr = self.seed_addr.ok_or_else(|| {
            NexarError::Elastic("no seed address configured for add_nodes".into())
        })?;

        let mut new_joins = Vec::new();

        for _ in 0..count {
            let worker = crate::cluster::WorkerNode::connect(seed_addr).await?;
            let pj = PendingJoin {
                rank: worker.rank,
                listen_addr: String::new(),
            };
            new_joins.push(pj.clone());
            self.pending_joins
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(pj);
        }

        Ok(new_joins)
    }
}
