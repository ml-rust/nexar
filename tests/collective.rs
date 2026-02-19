use nexar::rpc::RpcHandler;
use nexar::{CpuAdapter, DataType, NexarClient, ReduceOp};
use std::sync::Arc;
use tokio::sync::Barrier;

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

// ============================================================================
// Reduce tests
// ============================================================================

#[tokio::test]
async fn test_reduce_2_nodes_sum() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        let mut data = vec![(rank + 1) as f32; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .reduce(ptr, 4, DataType::F32, ReduceOp::Sum, 0)
                .await
                .unwrap();
        }

        if rank == 0 {
            // Root gets the sum: 1 + 2 = 3
            assert_eq!(data, vec![3.0f32; 4], "root reduce failed");
        }
        // Non-root buffers are unspecified.
    })
    .await;
}

#[tokio::test]
async fn test_reduce_4_nodes_sum() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let mut data = vec![(rank + 1) as f32; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .reduce(ptr, 4, DataType::F32, ReduceOp::Sum, 0)
                .await
                .unwrap();
        }

        if rank == 0 {
            // 1 + 2 + 3 + 4 = 10
            assert_eq!(data, vec![10.0f32; 4], "root reduce 4-node failed");
        }
    })
    .await;
}

#[tokio::test]
async fn test_reduce_3_nodes_nonzero_root() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let root = 2;
        let mut data = vec![(rank + 1) as f32; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .reduce(ptr, 4, DataType::F32, ReduceOp::Sum, root)
                .await
                .unwrap();
        }

        if rank == root {
            // 1 + 2 + 3 = 6
            assert_eq!(data, vec![6.0f32; 4], "root reduce nonzero-root failed");
        }
    })
    .await;
}

// ============================================================================
// All-to-all tests
// ============================================================================

#[tokio::test]
async fn test_alltoall_2_nodes() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        // Each rank sends 2 elements to each peer (2 * 2 = 4 total).
        let send_data: Vec<f32> = vec![
            (rank * 10) as f32,
            (rank * 10 + 1) as f32,
            (rank * 10 + 2) as f32,
            (rank * 10 + 3) as f32,
        ];
        let mut recv_data: Vec<f32> = vec![0.0; 4];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_to_all(send_ptr, recv_ptr, 2, DataType::F32)
                .await
                .unwrap();
        }

        // Rank 0 sends [0,1] to rank 0, [2,3] to rank 1.
        // Rank 1 sends [10,11] to rank 0, [12,13] to rank 1.
        // Rank 0 receives: [0,1] from self, [10,11] from rank 1.
        // Rank 1 receives: [2,3] from rank 0, [12,13] from self.
        let expected = if rank == 0 {
            vec![0.0, 1.0, 10.0, 11.0]
        } else {
            vec![2.0, 3.0, 12.0, 13.0]
        };
        assert_eq!(recv_data, expected, "rank {rank} alltoall failed");
    })
    .await;
}

#[tokio::test]
async fn test_alltoall_4_nodes() {
    run_collective(4, |client| async move {
        let rank = client.rank() as usize;
        // Each rank sends 1 element to each peer (1 * 4 = 4 total).
        let send_data: Vec<i32> = (0..4).map(|dest| (rank * 100 + dest) as i32).collect();
        let mut recv_data: Vec<i32> = vec![0; 4];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_to_all(send_ptr, recv_ptr, 1, DataType::I32)
                .await
                .unwrap();
        }

        // Rank r receives: from rank 0: 0*100+r, from rank 1: 1*100+r, etc.
        let expected: Vec<i32> = (0..4).map(|src| (src * 100 + rank) as i32).collect();
        assert_eq!(recv_data, expected, "rank {rank} alltoall 4-node failed");
    })
    .await;
}

// ============================================================================
// Scan tests
// ============================================================================

#[tokio::test]
async fn test_scan_3_nodes_sum() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let mut data = vec![(rank + 1) as f32; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .scan(ptr, 4, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Inclusive scan with Sum:
        // Rank 0: 1, Rank 1: 1+2=3, Rank 2: 1+2+3=6
        let expected = match rank {
            0 => 1.0,
            1 => 3.0,
            2 => 6.0,
            _ => unreachable!(),
        };
        assert_eq!(data, vec![expected; 4], "rank {rank} scan failed");
    })
    .await;
}

#[tokio::test]
async fn test_scan_4_nodes_sum() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let mut data = vec![1.0f32; 2];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .scan(ptr, 2, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Each rank starts with 1.0. Inclusive scan:
        // Rank 0: 1, Rank 1: 2, Rank 2: 3, Rank 3: 4
        let expected = (rank + 1) as f32;
        assert_eq!(data, vec![expected; 2], "rank {rank} scan 4-node failed");
    })
    .await;
}

#[tokio::test]
async fn test_scan_2_nodes_max() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        let mut data = vec![(rank + 1) as f32; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .scan(ptr, 4, DataType::F32, ReduceOp::Max)
                .await
                .unwrap();
        }

        // Inclusive max scan: Rank 0: 1, Rank 1: max(1,2)=2
        let expected = match rank {
            0 => 1.0,
            1 => 2.0,
            _ => unreachable!(),
        };
        assert_eq!(data, vec![expected; 4], "rank {rank} scan max failed");
    })
    .await;
}

// ============================================================================
// Split tests
// ============================================================================

#[tokio::test]
async fn test_split_two_groups() {
    // 4 ranks split into 2 groups of 2.
    // Ranks 0,1 -> color 0, Ranks 2,3 -> color 1.
    // Each group does allreduce within its group.
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(4, adapter).await.unwrap();
    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();

    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();

    for c in &clients {
        let c = Arc::clone(c);
        let barrier = Arc::clone(&barrier);
        handles.push(tokio::spawn(async move {
            let rank = c.rank();
            let color = if rank < 2 { 0u32 } else { 1u32 };
            let key = rank;

            let sub = c.split(color, key).await.unwrap();

            assert_eq!(sub.world_size(), 2, "rank {rank}: sub world size wrong");

            // Within each group, do allreduce.
            let mut data = vec![(rank + 1) as f32; 4];
            let ptr = data.as_mut_ptr() as u64;

            unsafe {
                sub.all_reduce(ptr, 4, DataType::F32, ReduceOp::Sum)
                    .await
                    .unwrap();
            }

            // Group 0 (ranks 0,1): 1 + 2 = 3
            // Group 1 (ranks 2,3): 3 + 4 = 7
            let expected = if rank < 2 { 3.0f32 } else { 7.0f32 };
            assert_eq!(
                data,
                vec![expected; 4],
                "rank {rank} split allreduce failed"
            );

            barrier.wait().await;
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}
