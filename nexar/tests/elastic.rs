//! Integration tests for elastic scaling (dynamic grow/shrink).

use nexar::device::CpuAdapter;
use nexar::{ElasticConfig, NexarClient};
use std::sync::Arc;

/// Helper: run allreduce on all managers' clients to verify collective works.
async fn verify_allreduce(managers: &[nexar::ElasticManager], expected_world: u32) {
    use nexar::DataType;
    use nexar::ReduceOp;

    let n = managers.len();
    assert_eq!(n as u32, expected_world);

    // Each rank contributes [1.0, 2.0, 3.0, 4.0].
    // After allreduce sum, expect [N, 2N, 3N, 4N].
    let count = 4usize;
    let mut handles = Vec::new();

    for m in managers {
        let client = m.client();
        let world = expected_world;
        handles.push(tokio::spawn(async move {
            let client = client.lock().await;
            let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let ptr = data.as_mut_ptr() as u64;
            unsafe {
                client
                    .all_reduce(ptr, count, DataType::F32, ReduceOp::Sum)
                    .await
                    .unwrap();
            }
            let expected: Vec<f32> = vec![
                world as f32,
                2.0 * world as f32,
                3.0 * world as f32,
                4.0 * world as f32,
            ];
            assert_eq!(data, expected, "allreduce mismatch");
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}

#[tokio::test]
async fn test_elastic_bootstrap_and_barrier() {
    let adapter = Arc::new(CpuAdapter::new());
    let config = nexar::NexarConfig::default();

    let bootstrap = NexarClient::bootstrap_elastic(4, ElasticConfig::default(), config, adapter)
        .await
        .unwrap();

    assert_eq!(bootstrap.managers.len(), 4);

    // Verify all managers have correct rank/world_size.
    for (i, m) in bootstrap.managers.iter().enumerate() {
        let client = m.client();
        let c = client.lock().await;
        assert_eq!(c.rank() as usize, i);
        assert_eq!(c.world_size(), 4);
    }

    // Run allreduce to verify the cluster works.
    verify_allreduce(&bootstrap.managers, 4).await;
}

#[tokio::test]
async fn test_elastic_noop_checkpoint() {
    let adapter = Arc::new(CpuAdapter::new());
    let config = nexar::NexarConfig::default();

    let bootstrap = NexarClient::bootstrap_elastic(2, ElasticConfig::default(), config, adapter)
        .await
        .unwrap();

    // No pending changes → checkpoint should act as barrier and return None.
    let mut handles = Vec::new();
    for m in &bootstrap.managers {
        let m = m.clone();
        handles.push(tokio::spawn(async move {
            m.elastic_checkpoint().await.unwrap()
        }));
    }

    for h in handles {
        let result = h.await.unwrap();
        assert!(result.is_none(), "expected None for no-op checkpoint");
    }

    // Verify cluster still works after no-op checkpoint.
    verify_allreduce(&bootstrap.managers, 2).await;
}
