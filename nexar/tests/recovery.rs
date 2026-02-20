//! Integration tests for the fault recovery orchestrator.

use nexar::{CpuAdapter, NexarClient, NexarConfig, RecoveryOrchestrator, RecoveryPolicy};
use std::sync::Arc;
use std::time::Duration;

/// Build a cluster config with fast heartbeats and short timeouts.
fn fast_config() -> NexarConfig {
    NexarConfig {
        collective_timeout: Duration::from_secs(5),
        barrier_timeout: Duration::from_secs(5),
        rpc_timeout: Duration::from_secs(5),
        heartbeat_interval: Duration::from_millis(50),
        heartbeat_timeout: Duration::from_millis(200),
        recovery_timeout: Duration::from_secs(2),
        ..NexarConfig::default()
    }
}

/// Bootstrap a local cluster with fast config.
async fn bootstrap_fast(world_size: u32) -> Vec<NexarClient> {
    let adapter = Arc::new(CpuAdapter::new());
    NexarClient::bootstrap_local_with_config(world_size, adapter, fast_config())
        .await
        .unwrap()
}

#[tokio::test]
async fn test_abort_policy() {
    let clients = bootstrap_fast(3).await;
    let mut clients_iter = clients.into_iter();
    let c0 = clients_iter.next().unwrap();
    let c1 = clients_iter.next().unwrap();
    let c2 = clients_iter.next().unwrap();

    // Kill rank 2.
    c2.close();

    // Create orchestrator on rank 0 with Abort policy.
    let (orch, _rx) = RecoveryOrchestrator::new(c0, RecoveryPolicy::Abort);

    // Also create one on rank 1 (not used, but keeps connections alive).
    let (_orch1, _rx1) = RecoveryOrchestrator::new(c1, RecoveryPolicy::Manual);

    // run() should return an error once rank 2 failure is detected.
    let result = orch.run().await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("Abort"),
        "expected Abort error, got: {err}"
    );
}

#[tokio::test]
async fn test_manual_policy() {
    let clients = bootstrap_fast(3).await;
    let mut clients_iter = clients.into_iter();
    let c0 = clients_iter.next().unwrap();
    let c1 = clients_iter.next().unwrap();
    let c2 = clients_iter.next().unwrap();

    let (orch, mut rx) = RecoveryOrchestrator::new(c0, RecoveryPolicy::Manual);
    let (_orch1, _rx1) = RecoveryOrchestrator::new(c1, RecoveryPolicy::Manual);

    // Kill rank 2.
    c2.close();

    // Run orchestrator in background.
    let orch = Arc::new(orch);
    let orch_clone = Arc::clone(&orch);
    let handle = tokio::spawn(async move { orch_clone.run().await });

    // Should receive a recovery event (manual = no rebuild, just notification).
    let event = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("timeout waiting for recovery event")
        .expect("channel closed");

    assert!(
        event.dead_ranks.contains(&2),
        "expected rank 2 in dead_ranks, got: {:?}",
        event.dead_ranks
    );
    // In manual mode, rank doesn't change.
    assert_eq!(event.old_rank, event.new_rank);

    orch.shutdown();
    let result = handle.await.unwrap();
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_automatic_recovery() {
    let clients = bootstrap_fast(4).await;
    let mut clients_iter = clients.into_iter();
    let c0 = clients_iter.next().unwrap();
    let c1 = clients_iter.next().unwrap();
    let c2 = clients_iter.next().unwrap();
    let c3 = clients_iter.next().unwrap();

    // Set up orchestrators on all survivors (0, 1, 3). Kill rank 2.
    let (orch0, mut rx0) = RecoveryOrchestrator::new(c0, RecoveryPolicy::Automatic);
    let (orch1, mut rx1) = RecoveryOrchestrator::new(c1, RecoveryPolicy::Automatic);
    let (orch3, mut rx3) = RecoveryOrchestrator::new(c3, RecoveryPolicy::Automatic);

    let orch0 = Arc::new(orch0);
    let orch1 = Arc::new(orch1);
    let orch3 = Arc::new(orch3);

    // Kill rank 2.
    c2.close();

    // Run all orchestrators concurrently.
    let h0 = {
        let o = Arc::clone(&orch0);
        tokio::spawn(async move { o.run().await })
    };
    let h1 = {
        let o = Arc::clone(&orch1);
        tokio::spawn(async move { o.run().await })
    };
    let h3 = {
        let o = Arc::clone(&orch3);
        tokio::spawn(async move { o.run().await })
    };

    // Wait for recovery events from all three survivors.
    let ev0 = tokio::time::timeout(Duration::from_secs(10), rx0.recv())
        .await
        .expect("timeout on rx0")
        .expect("rx0 closed");
    let ev1 = tokio::time::timeout(Duration::from_secs(10), rx1.recv())
        .await
        .expect("timeout on rx1")
        .expect("rx1 closed");
    let ev3 = tokio::time::timeout(Duration::from_secs(10), rx3.recv())
        .await
        .expect("timeout on rx3")
        .expect("rx3 closed");

    // All should agree rank 2 is dead.
    assert!(ev0.dead_ranks.contains(&2));
    assert!(ev1.dead_ranks.contains(&2));
    assert!(ev3.dead_ranks.contains(&2));

    // New world size should be 3.
    assert_eq!(ev0.new_world_size, 3);
    assert_eq!(ev1.new_world_size, 3);
    assert_eq!(ev3.new_world_size, 3);

    // New ranks should be contiguous 0..3.
    let mut new_ranks = vec![ev0.new_rank, ev1.new_rank, ev3.new_rank];
    new_ranks.sort();
    assert_eq!(new_ranks, vec![0, 1, 2]);

    // Shutdown all.
    orch0.shutdown();
    orch1.shutdown();
    orch3.shutdown();

    let _ = h0.await;
    let _ = h1.await;
    let _ = h3.await;
}

#[tokio::test]
async fn test_cascading_failure() {
    // 4 nodes: kill rank 2, then kill rank 3 shortly after.
    // The failure_watch-aware agreement should detect rank 3's death instantly
    // instead of waiting for a timeout, then recover with survivors {0, 1}.
    let clients = bootstrap_fast(4).await;
    let mut clients_iter = clients.into_iter();
    let c0 = clients_iter.next().unwrap();
    let c1 = clients_iter.next().unwrap();
    let c2 = clients_iter.next().unwrap();
    let c3 = clients_iter.next().unwrap();

    // Kill rank 2 immediately.
    c2.close();

    let (orch0, mut rx0) = RecoveryOrchestrator::new(c0, RecoveryPolicy::Automatic);
    let (orch1, mut rx1) = RecoveryOrchestrator::new(c1, RecoveryPolicy::Automatic);

    let orch0 = Arc::new(orch0);
    let orch1 = Arc::new(orch1);

    let h0 = {
        let o = Arc::clone(&orch0);
        tokio::spawn(async move { o.run().await })
    };
    let h1 = {
        let o = Arc::clone(&orch1);
        tokio::spawn(async move { o.run().await })
    };

    // Kill rank 3 shortly after — this triggers cascading failure detection
    // via failure_watch during the agreement phase.
    tokio::time::sleep(Duration::from_millis(100)).await;
    c3.close();

    // Both survivors should recover quickly (no full timeout wait).
    let ev0 = tokio::time::timeout(Duration::from_secs(5), rx0.recv())
        .await
        .expect("timeout on rx0")
        .expect("rx0 closed");
    let ev1 = tokio::time::timeout(Duration::from_secs(5), rx1.recv())
        .await
        .expect("timeout on rx1")
        .expect("rx1 closed");

    assert!(
        ev0.dead_ranks.contains(&2),
        "expected 2 in dead: {:?}",
        ev0.dead_ranks
    );
    assert!(
        ev0.dead_ranks.contains(&3),
        "expected 3 in dead: {:?}",
        ev0.dead_ranks
    );
    assert!(ev1.dead_ranks.contains(&2));
    assert!(ev1.dead_ranks.contains(&3));

    assert_eq!(ev0.new_world_size, 2);
    assert_eq!(ev1.new_world_size, 2);

    let mut new_ranks = vec![ev0.new_rank, ev1.new_rank];
    new_ranks.sort();
    assert_eq!(new_ranks, vec![0, 1]);

    orch0.shutdown();
    orch1.shutdown();
    let _ = h0.await;
    let _ = h1.await;
}
