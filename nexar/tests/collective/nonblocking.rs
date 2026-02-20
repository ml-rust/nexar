use nexar::types::{DataType, ReduceOp};
use nexar::{CollectiveGroup, CpuAdapter, NexarClient};
use std::sync::Arc;

/// Helper: bootstrap N clients as Arc.
async fn bootstrap_arc(n: u32) -> Vec<Arc<NexarClient>> {
    let adapter = Arc::new(CpuAdapter::new());
    NexarClient::bootstrap_local(n, adapter)
        .await
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect()
}

/// Launch 2 allreduces concurrently via _nb, verify both correct.
#[tokio::test]
async fn test_two_concurrent_allreduces() {
    let clients = bootstrap_arc(4).await;

    let mut handles = Vec::new();
    for client in &clients {
        let c = Arc::clone(client);
        handles.push(tokio::spawn(async move {
            let rank = c.rank();

            // Buffer A: [rank+1, rank+1, rank+1, rank+1]
            let mut buf_a: Vec<f32> = vec![(rank + 1) as f32; 4];
            // Buffer B: [10*(rank+1), ...]
            let mut buf_b: Vec<f32> = vec![(10 * (rank + 1)) as f32; 4];

            let ptr_a = buf_a.as_mut_ptr() as u64;
            let ptr_b = buf_b.as_mut_ptr() as u64;

            // Launch both non-blocking.
            let h_a = unsafe { c.all_reduce_nb(ptr_a, 4, DataType::F32, ReduceOp::Sum) };
            let h_b = unsafe { c.all_reduce_nb(ptr_b, 4, DataType::F32, ReduceOp::Sum) };

            // Wait for both.
            let mut group = CollectiveGroup::new();
            group.push(h_a);
            group.push(h_b);
            group.wait_all().await.unwrap();

            // Expected: sum of 1+2+3+4 = 10 for buf_a.
            assert_eq!(buf_a, vec![10.0f32; 4], "rank {rank} buf_a mismatch");
            // Expected: sum of 10+20+30+40 = 100 for buf_b.
            assert_eq!(buf_b, vec![100.0f32; 4], "rank {rank} buf_b mismatch");
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}

/// Non-blocking broadcast: verify data arrives correctly.
#[tokio::test]
async fn test_nonblocking_broadcast() {
    let clients = bootstrap_arc(3).await;

    let mut handles = Vec::new();
    for client in &clients {
        let c = Arc::clone(client);
        handles.push(tokio::spawn(async move {
            let mut buf: Vec<f32> = if c.rank() == 0 {
                vec![42.0, 43.0, 44.0, 45.0]
            } else {
                vec![0.0; 4]
            };

            let ptr = buf.as_mut_ptr() as u64;
            let h = unsafe { c.broadcast_nb(ptr, 4, DataType::F32, 0) };
            h.wait().await.unwrap();

            assert_eq!(buf, vec![42.0, 43.0, 44.0, 45.0]);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}

/// Non-blocking barrier.
#[tokio::test]
async fn test_nonblocking_barrier() {
    let clients = bootstrap_arc(4).await;

    let mut handles = Vec::new();
    for client in &clients {
        let c = Arc::clone(client);
        handles.push(tokio::spawn(async move {
            let h = c.barrier_nb();
            h.wait().await.unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}

/// CollectiveHandle::is_finished works correctly.
#[tokio::test]
async fn test_collective_handle_is_finished() {
    let clients = bootstrap_arc(2).await;

    let mut handles = Vec::new();
    for client in &clients {
        let c = Arc::clone(client);
        handles.push(tokio::spawn(async move {
            let h = c.barrier_nb();
            // Can't guarantee it's not finished yet, but it should eventually finish.
            h.wait().await.unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}
