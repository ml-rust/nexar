//! Integration tests for nexar-rdma mesh establishment.
//!
//! These tests bootstrap a local cluster via nexar, then call
//! `establish_rdma_mesh` to attach RDMA state. On machines without
//! InfiniBand hardware, the RDMA setup gracefully falls back to
//! QUIC-only transport (logged as a warning).

use nexar::{CpuAdapter, DataType, NexarClient, ReduceOp};
use nexar_rdma::bootstrap::establish_rdma_mesh_local;
use std::sync::Arc;

/// Helper: bootstrap + RDMA mesh, then run a collective across N clients.
async fn run_with_rdma<F, Fut>(world_size: u32, f: F)
where
    F: Fn(Arc<NexarClient>) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(world_size, adapter)
        .await
        .unwrap();

    // Attempt to establish RDMA mesh (falls back to QUIC if no IB hardware).
    establish_rdma_mesh_local(&clients).await;

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
}

#[tokio::test]
async fn test_rdma_mesh_bootstrap_2_nodes() {
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(2, adapter).await.unwrap();
    establish_rdma_mesh_local(&clients).await;
    assert_eq!(clients.len(), 2);
    assert_eq!(clients[0].rank(), 0);
    assert_eq!(clients[1].rank(), 1);
}

#[tokio::test]
async fn test_rdma_mesh_bootstrap_4_nodes() {
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(4, adapter).await.unwrap();
    establish_rdma_mesh_local(&clients).await;
    for (i, c) in clients.iter().enumerate() {
        assert_eq!(c.rank() as usize, i);
        assert_eq!(c.world_size(), 4);
    }
}

#[tokio::test]
async fn test_send_recv_with_rdma_mesh() {
    // Tests that point-to-point send/recv works after RDMA mesh setup.
    run_with_rdma(2, |client| async move {
        let rank = client.rank();
        let send_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut recv_buf: Vec<f32> = vec![0.0; 4];
        let size = send_data.len() * std::mem::size_of::<f32>();

        if rank == 0 {
            unsafe {
                client
                    .send(send_data.as_ptr() as u64, size, 1, 99)
                    .await
                    .unwrap();
            }
        } else {
            unsafe {
                client
                    .recv(recv_buf.as_mut_ptr() as u64, size, 0, 99)
                    .await
                    .unwrap();
            }
            assert_eq!(recv_buf, vec![1.0, 2.0, 3.0, 4.0]);
        }
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_with_rdma_mesh() {
    run_with_rdma(4, |client| async move {
        let rank = client.rank();
        let mut data: Vec<f32> = vec![rank as f32 + 1.0; 8];
        let ptr = data.as_mut_ptr() as u64;
        let count = data.len();

        unsafe {
            client
                .all_reduce(ptr, count, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Sum of 1+2+3+4 = 10
        let expected = 10.0f32;
        for &v in &data {
            assert!(
                (v - expected).abs() < 1e-6,
                "rank {rank}: expected {expected}, got {v}"
            );
        }
    })
    .await;
}

#[tokio::test]
async fn test_broadcast_with_rdma_mesh() {
    run_with_rdma(3, |client| async move {
        let rank = client.rank();
        let mut data: Vec<f32> = if rank == 0 {
            vec![42.0; 4]
        } else {
            vec![0.0; 4]
        };
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .broadcast(ptr, data.len(), DataType::F32, 0)
                .await
                .unwrap();
        }

        for &v in &data {
            assert_eq!(v, 42.0, "rank {rank}: expected 42.0, got {v}");
        }
    })
    .await;
}

#[tokio::test]
async fn test_gather_with_rdma_mesh() {
    run_with_rdma(3, |client| async move {
        let rank = client.rank();
        let send_data: Vec<f32> = vec![(rank + 1) as f32; 2];
        let mut recv_data: Vec<f32> = vec![0.0; 6];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .gather(send_ptr, recv_ptr, 2, DataType::F32, 0)
                .await
                .unwrap();
        }

        if rank == 0 {
            assert_eq!(recv_data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        }
    })
    .await;
}

#[tokio::test]
async fn test_scatter_with_rdma_mesh() {
    run_with_rdma(3, |client| async move {
        let rank = client.rank();
        let send_data: Vec<f32> = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let mut recv_data: Vec<f32> = vec![0.0; 2];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .scatter(send_ptr, recv_ptr, 2, DataType::F32, 0)
                .await
                .unwrap();
        }

        let expected = match rank {
            0 => vec![10.0, 11.0],
            1 => vec![20.0, 21.0],
            2 => vec![30.0, 31.0],
            _ => unreachable!(),
        };
        assert_eq!(recv_data, expected, "rank {rank} scatter failed");
    })
    .await;
}

#[tokio::test]
async fn test_barrier_with_rdma_mesh() {
    run_with_rdma(4, |client| async move {
        client.barrier().await.unwrap();
    })
    .await;
}
