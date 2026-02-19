use nexar::rpc::RpcHandler;
use nexar::{CpuAdapter, DataType, NexarClient, ReduceOp};
use std::sync::Arc;

/// Helper: run a collective operation across N clients concurrently.
/// Keeps all clients alive until every task completes.
async fn run_collective<F, Fut>(world_size: u32, f: F)
where
    F: Fn(Arc<NexarClient>) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(world_size, adapter)
        .await
        .unwrap();
    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();

    let f = Arc::new(f);
    let mut handles = Vec::new();
    for c in &clients {
        let c = Arc::clone(c);
        let f = Arc::clone(&f);
        handles.push(tokio::spawn(async move { f(c).await }));
    }
    for h in handles {
        h.await.unwrap();
    }
    // `clients` dropped here — all tasks already complete.
}

// ============================================================================
// AllReduce tests
// ============================================================================

#[tokio::test]
async fn test_allreduce_2_nodes_f32() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        let val = (rank + 1) as f32;
        let mut data = vec![val; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce(ptr, 4, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Sum of [1,1,1,1] + [2,2,2,2] = [3,3,3,3]
        assert_eq!(data, vec![3.0f32; 4], "rank {rank} allreduce failed");
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_3_nodes_f32() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let val = (rank + 1) as f32;
        let mut data = vec![val; 6];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce(ptr, 6, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Sum: 1 + 2 + 3 = 6
        assert_eq!(data, vec![6.0f32; 6], "rank {rank} allreduce failed");
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_4_nodes_i32() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let val = (rank + 1) as i32;
        let mut data = vec![val; 8];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce(ptr, 8, DataType::I32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Sum: 1 + 2 + 3 + 4 = 10
        assert_eq!(data, vec![10i32; 8], "rank {rank} allreduce failed");
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_uneven_count() {
    // 3 nodes, 7 elements (not divisible by 3)
    run_collective(3, |client| async move {
        let rank = client.rank();
        let mut data: Vec<f32> = (0..7).map(|i| (i as f32) * ((rank + 1) as f32)).collect();
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce(ptr, 7, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Each element i: sum of i*1 + i*2 + i*3 = i*6
        let expected: Vec<f32> = (0..7).map(|i| (i as f32) * 6.0).collect();
        assert_eq!(data, expected, "rank {rank} uneven allreduce failed");
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_min_3_nodes() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let val = (rank + 1) as f32;
        let mut data = vec![val; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce(ptr, 4, DataType::F32, ReduceOp::Min)
                .await
                .unwrap();
        }

        // Min of 1, 2, 3 = 1
        assert_eq!(data, vec![1.0f32; 4], "rank {rank} allreduce min failed");
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_max_3_nodes() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let val = (rank + 1) as f32;
        let mut data = vec![val; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce(ptr, 4, DataType::F32, ReduceOp::Max)
                .await
                .unwrap();
        }

        // Max of 1, 2, 3 = 3
        assert_eq!(data, vec![3.0f32; 4], "rank {rank} allreduce max failed");
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_prod_2_nodes() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        let val = (rank + 2) as f32; // rank 0 → 2.0, rank 1 → 3.0
        let mut data = vec![val; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce(ptr, 4, DataType::F32, ReduceOp::Prod)
                .await
                .unwrap();
        }

        // Product of 2.0 * 3.0 = 6.0
        assert_eq!(data, vec![6.0f32; 4], "rank {rank} allreduce prod failed");
    })
    .await;
}

// ============================================================================
// Broadcast tests
// ============================================================================

#[tokio::test]
async fn test_broadcast_from_root_0() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let mut data: Vec<f32> = if rank == 0 {
            vec![42.0, 43.0, 44.0, 45.0]
        } else {
            vec![0.0; 4]
        };
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client.broadcast(ptr, 4, DataType::F32, 0).await.unwrap();
        }

        assert_eq!(
            data,
            vec![42.0, 43.0, 44.0, 45.0],
            "rank {rank} broadcast failed"
        );
    })
    .await;
}

#[tokio::test]
async fn test_broadcast_from_nonzero_root() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let root = 2;
        let mut data: Vec<f32> = if rank == root {
            vec![99.0, 100.0, 101.0]
        } else {
            vec![0.0; 3]
        };
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client.broadcast(ptr, 3, DataType::F32, root).await.unwrap();
        }

        assert_eq!(
            data,
            vec![99.0, 100.0, 101.0],
            "rank {rank} broadcast from root {root} failed"
        );
    })
    .await;
}

#[tokio::test]
async fn test_broadcast_5_nodes_tree() {
    // 5 nodes triggers tree broadcast (threshold is 4).
    run_collective(5, |client| async move {
        let rank = client.rank();
        let mut data: Vec<i32> = if rank == 0 { vec![7, 8, 9] } else { vec![0; 3] };
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client.broadcast(ptr, 3, DataType::I32, 0).await.unwrap();
        }

        assert_eq!(data, vec![7, 8, 9], "rank {rank} tree broadcast failed");
    })
    .await;
}

// ============================================================================
// AllGather tests
// ============================================================================

#[tokio::test]
async fn test_allgather_3_nodes() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let send_data: Vec<f32> = vec![(rank + 1) as f32; 2];
        let mut recv_data: Vec<f32> = vec![0.0; 6]; // 2 * 3 = 6

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_gather(send_ptr, recv_ptr, 2, DataType::F32)
                .await
                .unwrap();
        }

        // Expected: [1,1, 2,2, 3,3] (concatenated in rank order)
        assert_eq!(
            recv_data,
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "rank {rank} allgather failed"
        );
    })
    .await;
}

#[tokio::test]
async fn test_allgather_4_nodes() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let send_data: Vec<i32> = vec![(rank as i32) * 10; 3];
        let mut recv_data: Vec<i32> = vec![0; 12]; // 3 * 4 = 12

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_gather(send_ptr, recv_ptr, 3, DataType::I32)
                .await
                .unwrap();
        }

        let expected = vec![0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30];
        assert_eq!(recv_data, expected, "rank {rank} allgather 4-node failed");
    })
    .await;
}

// ============================================================================
// ReduceScatter tests
// ============================================================================

#[tokio::test]
async fn test_reduce_scatter_2_nodes() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        let send_data: Vec<f32> = vec![(rank + 1) as f32; 4];
        let mut recv_data: Vec<f32> = vec![0.0; 2];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .reduce_scatter(send_ptr, recv_ptr, 2, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Each chunk: sum of [1+2, 1+2] = [3, 3]
        assert_eq!(
            recv_data,
            vec![3.0, 3.0],
            "rank {rank} reduce_scatter failed"
        );
    })
    .await;
}

#[tokio::test]
async fn test_reduce_scatter_3_nodes() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        // Each rank contributes 6 elements (2 per chunk × 3 ranks).
        let send_data: Vec<f32> = vec![(rank + 1) as f32; 6];
        let mut recv_data: Vec<f32> = vec![0.0; 2];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .reduce_scatter(send_ptr, recv_ptr, 2, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Each chunk: sum of 1+2+3 = 6
        assert_eq!(
            recv_data,
            vec![6.0, 6.0],
            "rank {rank} reduce_scatter 3-node failed"
        );
    })
    .await;
}

// ============================================================================
// Barrier tests
// ============================================================================

#[tokio::test]
async fn test_barrier_4_nodes() {
    run_collective(4, |client| async move {
        client.barrier().await.unwrap();
    })
    .await;
}

#[tokio::test]
async fn test_barrier_5_nodes_dissemination() {
    // 5 nodes triggers dissemination barrier (threshold is 5).
    run_collective(5, |client| async move {
        client.barrier().await.unwrap();
    })
    .await;
}

#[tokio::test]
async fn test_barrier_2_nodes_double() {
    run_collective(2, |client| async move {
        client.barrier().await.unwrap();
        // Second barrier should also work.
        client.barrier().await.unwrap();
    })
    .await;
}

// ============================================================================
// RPC tests
// ============================================================================

#[tokio::test]
async fn test_rpc_call_and_response() {
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(2, adapter).await.unwrap();
    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();

    // Register handler on rank 1: reverses the input bytes and appends 0xFF.
    let handler: RpcHandler = Arc::new(|args: &[u8]| {
        let mut result: Vec<u8> = args.iter().rev().copied().collect();
        result.push(0xFF);
        result
    });
    clients[1].register_rpc(42, handler).await;

    let c0 = Arc::clone(&clients[0]);
    let c1 = Arc::clone(&clients[1]);

    // Rank 0 calls RPC on rank 1, rank 1 handles the incoming request.
    let caller = tokio::spawn(async move { c0.rpc(1, 42, &[1, 2, 3]).await.unwrap() });

    let responder = tokio::spawn(async move {
        // Rank 1 needs to receive the RPC request and dispatch it.
        let dispatcher = c1.rpc_dispatcher();
        let peer = Arc::clone(c1.peer(0).unwrap());
        let msg = c1.recv_rpc_request(0).await.unwrap();
        match msg {
            nexar::NexarMessage::Rpc {
                req_id,
                fn_id,
                payload,
            } => {
                dispatcher
                    .handle_request(&peer, req_id, fn_id, &payload)
                    .await
                    .unwrap();
            }
            other => panic!("expected Rpc, got {other:?}"),
        }
    });

    let result = caller.await.unwrap();
    responder.await.unwrap();

    assert_eq!(result, vec![3, 2, 1, 0xFF]);
}
