use nexar::client::NexarClient;
use nexar::cluster::sparse::{
    TopologyStrategy, build_neighbors, build_routing_table, build_spanning_tree,
};
use nexar::config::NexarConfig;
use nexar::device::CpuAdapter;
use nexar::types::{DataType, ReduceOp};
use std::sync::Arc;

fn kregular_config(degree: usize) -> NexarConfig {
    NexarConfig {
        topology: TopologyStrategy::KRegular { degree },
        enable_tcp_bulk_sidecar: false,
        ..NexarConfig::default()
    }
}

fn hypercube_config() -> NexarConfig {
    NexarConfig {
        topology: TopologyStrategy::Hypercube,
        enable_tcp_bulk_sidecar: false,
        ..NexarConfig::default()
    }
}

// --- Unit tests for topology ---

#[test]
fn test_kregular_neighbors() {
    // KRegular{6} with 16 nodes: each rank has exactly 6 neighbors (±1, ±2, ±3).
    for rank in 0..16u32 {
        let neighbors = build_neighbors(&TopologyStrategy::KRegular { degree: 6 }, rank, 16);
        assert_eq!(
            neighbors.len(),
            6,
            "rank {rank} has {} neighbors, expected 6",
            neighbors.len()
        );
        // Verify ±1, ±2, ±3
        for d in 1..=3u32 {
            assert!(neighbors.contains(&((rank + d) % 16)));
            assert!(neighbors.contains(&((rank + 16 - d) % 16)));
        }
    }
}

#[test]
fn test_hypercube_neighbors() {
    // Hypercube with 16 nodes: each rank has 4 neighbors (log2(16) = 4).
    for rank in 0..16u32 {
        let neighbors = build_neighbors(&TopologyStrategy::Hypercube, rank, 16);
        assert_eq!(
            neighbors.len(),
            4,
            "rank {rank} has {} neighbors, expected 4",
            neighbors.len()
        );
        // Each neighbor differs in exactly one bit.
        for &n in &neighbors {
            let diff = rank ^ n;
            assert_eq!(
                diff.count_ones(),
                1,
                "rank {rank} neighbor {n} differs in {} bits",
                diff.count_ones()
            );
        }
    }
}

#[test]
fn test_routing_table_complete() {
    // Every rank should have a route to every other rank.
    let strategy = TopologyStrategy::KRegular { degree: 6 };
    for rank in 0..16u32 {
        let rt = build_routing_table(&strategy, rank, 16);
        for dest in 0..16u32 {
            if dest == rank {
                continue;
            }
            assert!(
                rt.route(dest).is_some(),
                "rank {rank} has no route to {dest}"
            );
        }
    }
}

#[test]
fn test_spanning_tree_covers_all() {
    let tree = build_spanning_tree(&TopologyStrategy::KRegular { degree: 6 }, 0, 16);
    assert_eq!(tree.children.len(), 16, "tree should contain all 16 ranks");
    // Root has no parent.
    assert!(!tree.parent.contains_key(&0));
    // All other ranks have a parent.
    for r in 1..16u32 {
        assert!(tree.parent.contains_key(&r), "rank {r} has no parent");
    }
    // Every parent->child edge should be between direct neighbors.
    let strategy = TopologyStrategy::KRegular { degree: 6 };
    for (&parent, children) in &tree.children {
        let parent_neighbors = build_neighbors(&strategy, parent, 16);
        for &child in children {
            assert!(
                parent_neighbors.contains(&child),
                "tree edge {parent}->{child} is not a direct connection"
            );
        }
    }
}

// --- Integration tests ---

#[tokio::test]
async fn test_connection_count() {
    // Verify each node has exactly K connections (not N-1).
    let adapter = Arc::new(CpuAdapter::new());
    let config = kregular_config(6);
    let clients = NexarClient::bootstrap_local_with_config(16, adapter, config)
        .await
        .unwrap();

    for client in &clients {
        let mut direct_count = 0;
        for r in 0..16u32 {
            if r != client.rank() && client.has_direct_peer(r) {
                direct_count += 1;
            }
        }
        assert_eq!(
            direct_count,
            6,
            "rank {} has {} direct peers, expected 6",
            client.rank(),
            direct_count
        );
    }
}

#[tokio::test]
async fn test_kregular_relay_send_recv() {
    // 16 nodes, KRegular{6}: point-to-point between non-adjacent ranks.
    let adapter = Arc::new(CpuAdapter::new());
    let config = kregular_config(6);
    let clients = NexarClient::bootstrap_local_with_config(16, adapter, config)
        .await
        .unwrap();

    // Rank 0 and rank 8 are NOT direct neighbors in KRegular{6} (only ±1..±3).
    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();
    let c0 = Arc::clone(&clients[0]);
    let c8 = Arc::clone(&clients[8]);

    let send_data: Vec<f32> = vec![3.125, 2.71, 1.41, 1.73];
    let mut recv_buf: Vec<f32> = vec![0.0; 4];
    let size = send_data.len() * std::mem::size_of::<f32>();

    let send_ptr = send_data.as_ptr() as u64;
    let recv_ptr = recv_buf.as_mut_ptr() as u64;

    let send_task = tokio::spawn(async move { unsafe { c0.send(send_ptr, size, 8, 99).await } });
    let recv_task = tokio::spawn(async move { unsafe { c8.recv(recv_ptr, size, 0, 99).await } });

    send_task.await.unwrap().unwrap();
    recv_task.await.unwrap().unwrap();
    assert_eq!(recv_buf, vec![3.125, 2.71, 1.41, 1.73]);
}

#[tokio::test]
async fn test_kregular_allreduce() {
    // 16 nodes, KRegular{6}: ring allreduce completes correctly.
    run_allreduce_test(16, Some(kregular_config(6))).await;
}

#[tokio::test]
async fn test_kregular_broadcast() {
    // 16 nodes, KRegular{6}: broadcast from root reaches all ranks.
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local_with_config(16, adapter, kregular_config(6))
        .await
        .unwrap();

    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();

    let mut handles = Vec::new();
    for client in &clients {
        let client = Arc::clone(client);
        handles.push(tokio::spawn(async move {
            let count = 8usize;
            let mut data: Vec<f32> = if client.rank() == 0 {
                vec![42.0; count]
            } else {
                vec![0.0; count]
            };
            let ptr = data.as_mut_ptr() as u64;
            unsafe {
                client
                    .broadcast(ptr, count, DataType::F32, 0)
                    .await
                    .unwrap();
            }
            data
        }));
    }

    for handle in handles {
        let result = handle.await.unwrap();
        for &v in &result {
            assert!((v - 42.0).abs() < 1e-6, "expected 42.0, got {v}");
        }
    }
}

#[tokio::test]
async fn test_kregular_barrier() {
    // 16 nodes, KRegular{6}: barrier completes.
    let adapter = Arc::new(CpuAdapter::new());
    let config = kregular_config(6);
    let clients = NexarClient::bootstrap_local_with_config(16, adapter, config)
        .await
        .unwrap();

    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();

    let mut handles = Vec::new();
    for client in &clients {
        let client = Arc::clone(client);
        handles.push(tokio::spawn(async move {
            client.barrier().await.unwrap();
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_hypercube_allreduce() {
    // 16 nodes, Hypercube: allreduce completes correctly.
    let config = hypercube_config();
    run_allreduce_test(16, Some(config)).await;
}

// --- Allreduce regression tests (various sizes and algorithms) ---

/// Helper: run allreduce with `world_size` nodes and verify all elements equal the expected sum.
async fn run_allreduce_test(world_size: u32, config: Option<NexarConfig>) {
    let adapter = Arc::new(CpuAdapter::new());
    let clients = if let Some(cfg) = config {
        NexarClient::bootstrap_local_with_config(world_size, adapter, cfg)
            .await
            .unwrap()
    } else {
        NexarClient::bootstrap_local(world_size, adapter)
            .await
            .unwrap()
    };

    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();
    let count = world_size.max(8) as usize;
    let expected_sum: f32 = (0..world_size).map(|r| r as f32 + 1.0).sum();

    let mut handles = Vec::new();
    for (i, client) in clients.iter().enumerate() {
        let client = Arc::clone(client);
        handles.push(tokio::spawn(async move {
            let val = (i as f32) + 1.0;
            let mut data = vec![val; count];
            let ptr = data.as_mut_ptr() as u64;
            unsafe {
                client
                    .all_reduce(ptr, count, DataType::F32, ReduceOp::Sum)
                    .await
                    .unwrap();
            }
            data
        }));
    }

    for (rank, handle) in handles.into_iter().enumerate() {
        let result = handle.await.unwrap();
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected_sum).abs() < 1e-3,
                "world={world_size} rank {rank} elem {i}: expected {expected_sum}, got {v}"
            );
        }
    }
}

fn no_tcp_config(ring_max_world: usize) -> NexarConfig {
    NexarConfig {
        enable_tcp_bulk_sidecar: false,
        ring_max_world,
        ..NexarConfig::default()
    }
}

#[tokio::test]
async fn test_halving_doubling_5nodes() {
    // Non-power-of-2: exercises excess rank handling.
    run_allreduce_test(5, Some(no_tcp_config(0))).await;
}

#[tokio::test]
async fn test_halving_doubling_8nodes() {
    // Power-of-2: straightforward halving-doubling.
    run_allreduce_test(8, Some(no_tcp_config(0))).await;
}

#[tokio::test]
async fn test_ring_7nodes_no_tcp() {
    // Odd count, QUIC-only (no TCP sidecar): regression for connection swapping bug.
    run_allreduce_test(7, Some(no_tcp_config(8))).await;
}
