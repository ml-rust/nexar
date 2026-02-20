//! Fault-injection tests: verify that collectives fail gracefully (no deadlocks)
//! when a peer connection is killed mid-operation.

use nexar::{BufferRef, CpuAdapter, DataType, Host, NexarClient, NexarConfig, ReduceOp};
use std::sync::Arc;
use std::time::Duration;

/// Build a cluster with short timeouts so fault tests complete quickly.
fn fast_config() -> NexarConfig {
    NexarConfig {
        collective_timeout: Duration::from_secs(2),
        barrier_timeout: Duration::from_secs(2),
        rpc_timeout: Duration::from_secs(2),
        heartbeat_interval: Duration::from_millis(100),
        heartbeat_timeout: Duration::from_millis(500),
        ..NexarConfig::default()
    }
}

/// Bootstrap a cluster with fast timeouts.
async fn bootstrap_fast(world_size: u32) -> Vec<Arc<NexarClient>> {
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local_with_config(world_size, adapter, fast_config())
        .await
        .unwrap();
    clients.into_iter().map(Arc::new).collect()
}

// ── AllReduce ───────────────────────────────────────────────────────

#[tokio::test]
async fn allreduce_peer_crash_returns_error() {
    let clients = bootstrap_fast(3).await;

    // Kill rank 2 before the collective starts.
    clients[2].close();

    let mut handles = Vec::new();
    for c in &clients[..2] {
        let c = Arc::clone(c);
        handles.push(tokio::spawn(async move {
            let mut data = vec![1.0f32; 4];
            let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 4 * 4) };
            c.all_reduce_host(&mut buf, 4, DataType::F32, ReduceOp::Sum)
                .await
        }));
    }

    // Both surviving ranks should get an error, NOT deadlock.
    for h in handles {
        let result = h.await.unwrap();
        assert!(
            result.is_err(),
            "expected error from allreduce with dead peer"
        );
    }
}

// ── Barrier ─────────────────────────────────────────────────────────

#[tokio::test]
async fn barrier_peer_crash_returns_error() {
    let clients = bootstrap_fast(3).await;

    // Kill rank 1 before the barrier.
    clients[1].close();

    let mut handles = Vec::new();
    for idx in [0, 2] {
        let c = Arc::clone(&clients[idx]);
        handles.push(tokio::spawn(async move { c.barrier().await }));
    }

    for h in handles {
        let result = h.await.unwrap();
        assert!(
            result.is_err(),
            "expected error from barrier with dead peer"
        );
    }
}

// ── Broadcast ───────────────────────────────────────────────────────

#[tokio::test]
async fn broadcast_root_crash_returns_error() {
    let clients = bootstrap_fast(3).await;

    // Kill the root (rank 0) before broadcast.
    clients[0].close();

    let mut handles = Vec::new();
    for c in &clients[1..] {
        let c = Arc::clone(c);
        handles.push(tokio::spawn(async move {
            let mut data = vec![0.0f32; 4];
            let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 4 * 4) };
            c.broadcast_host(&mut buf, 4, DataType::F32, 0).await
        }));
    }

    for h in handles {
        let result = h.await.unwrap();
        assert!(
            result.is_err(),
            "expected error from broadcast with dead root"
        );
    }
}

// ── Mid-collective crash ────────────────────────────────────────────

#[tokio::test]
async fn allreduce_mid_collective_crash_returns_error() {
    let clients = bootstrap_fast(4).await;

    let kill_target = Arc::clone(&clients[3]);

    let mut handles = Vec::new();
    for c in &clients[..3] {
        let c = Arc::clone(c);
        handles.push(tokio::spawn(async move {
            let mut data = vec![1.0f32; 64];
            let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 64 * 4) };
            c.all_reduce_host(&mut buf, 64, DataType::F32, ReduceOp::Sum)
                .await
        }));
    }

    // Let the collective start, then kill rank 3 after a brief delay.
    tokio::time::sleep(Duration::from_millis(50)).await;
    kill_target.close();

    for h in handles {
        let result = h.await.unwrap();
        assert!(
            result.is_err(),
            "expected error from allreduce after mid-collective crash"
        );
    }
}

// ── Health monitor detects failure ──────────────────────────────────

#[tokio::test]
async fn health_monitor_detects_dead_peer() {
    let clients = bootstrap_fast(2).await;

    let mut watch = clients[0].failure_watch();

    // Kill rank 1.
    clients[1].close();

    // The health monitor should detect the failure within heartbeat_timeout.
    let detected = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            watch.changed().await.unwrap();
            let dead = watch.borrow().clone();
            if dead.contains(&1) {
                return;
            }
        }
    })
    .await;

    assert!(
        detected.is_ok(),
        "health monitor did not detect dead peer within timeout"
    );
}
