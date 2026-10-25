//! Post-formation join loop for elastic scaling.

use crate::cluster::control::{read_control, tokens_match, write_control};
use crate::cluster::topology::ClusterMap;
use crate::cluster::{PendingJoin, SeedNode};
use crate::error::Result;
use crate::protocol::NexarMessage;
use crate::transport::tls::ClusterCa;
use crate::types::PROTOCOL_VERSION;
use std::sync::Arc;

impl SeedNode {
    /// Background loop accepting new Hello messages from joining nodes after formation.
    ///
    /// Assigns new ranks, issues certs, sends Welcome, and pushes `PendingJoin`
    /// entries for the `ElasticManager` to process at the next checkpoint.
    pub async fn accept_elastic(
        &self,
        ca: Arc<ClusterCa>,
        next_rank: Arc<std::sync::atomic::AtomicU32>,
        max_world_size: u32,
        pending_joins: Arc<std::sync::Mutex<Vec<PendingJoin>>>,
        cluster_map: Arc<std::sync::Mutex<ClusterMap>>,
    ) -> Result<()> {
        let ca_cert_der = ca.cert_der();

        loop {
            let conn = match self.listener.accept().await {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("elastic accept error: {e}");
                    continue;
                }
            };

            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!("elastic accept_bi error: {e}");
                    continue;
                }
            };

            let msg = match read_control(&mut recv).await {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!("elastic join: failed to read hello: {e}");
                    continue;
                }
            };

            let listen_addr = match msg {
                NexarMessage::Hello {
                    protocol_version,
                    cluster_token: token,
                    listen_addr,
                    ..
                } => {
                    if protocol_version != PROTOCOL_VERSION {
                        tracing::warn!("elastic join: protocol mismatch");
                        continue;
                    }
                    if let Some(expected) = &self.cluster_token
                        && !tokens_match(&token, expected)
                    {
                        tracing::warn!("elastic join: token mismatch");
                        continue;
                    }
                    listen_addr
                }
                _ => continue,
            };

            let rank = next_rank.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            if max_world_size > 0 && rank >= max_world_size {
                next_rank.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!("elastic join rejected: max world size {max_world_size} reached");
                continue;
            }

            // Use the advertised address if the node sent one, else the observed one.
            let resolved_addr = if listen_addr.is_empty() {
                conn.remote_address().to_string()
            } else {
                listen_addr
            };

            let (node_cert, node_key) = match ca.issue_cert("localhost") {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!("elastic join: cert issue failed: {e}");
                    continue;
                }
            };

            let peers = {
                let map = cluster_map.lock().unwrap_or_else(|p| p.into_inner());
                map.alive_peers()
            };

            {
                let mut map = cluster_map.lock().unwrap_or_else(|p| p.into_inner());
                map.add_peer(rank, resolved_addr.clone());
            }

            let welcome = NexarMessage::Welcome {
                rank,
                world_size: rank + 1, // Temporary; real world size set at checkpoint
                peers,
                ca_cert: ca_cert_der.as_ref().to_vec(),
                node_cert: node_cert.as_ref().to_vec(),
                node_key: node_key.secret_der().to_vec(),
                observed_addr: conn.remote_address().ip().to_string(),
            };

            if let Err(e) = write_control(&mut send, &welcome).await {
                tracing::warn!("elastic join: send welcome failed: {e}");
                continue;
            }

            pending_joins
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(PendingJoin {
                    rank,
                    listen_addr: resolved_addr,
                });

            tracing::info!("elastic join: rank {rank} queued for next checkpoint");
        }
    }
}
