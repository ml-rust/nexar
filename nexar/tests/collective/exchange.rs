use nexar::DataType;

use super::helpers::run_collective;

// ============================================================================
// AllGather tests
// ============================================================================

#[tokio::test]
async fn test_allgather_3_nodes() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let send_data: Vec<f32> = vec![(rank + 1) as f32; 2];
        let mut recv_data: Vec<f32> = vec![0.0; 6];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_gather(send_ptr, recv_ptr, 2, DataType::F32)
                .await
                .unwrap();
        }

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
        let mut recv_data: Vec<i32> = vec![0; 12];

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
// All-to-all tests
// ============================================================================

#[tokio::test]
async fn test_alltoall_2_nodes() {
    run_collective(2, |client| async move {
        let rank = client.rank();
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

        let expected: Vec<i32> = (0..4).map(|src| (src * 100 + rank) as i32).collect();
        assert_eq!(recv_data, expected, "rank {rank} alltoall 4-node failed");
    })
    .await;
}

// ============================================================================
// Gather tests
// ============================================================================

#[tokio::test]
async fn test_gather_3_nodes() {
    run_collective(3, |client| async move {
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
async fn test_gather_4_nodes_nonzero_root() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let send_data: Vec<i32> = vec![(rank as i32) * 10; 3];
        let mut recv_data: Vec<i32> = vec![0; 12];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .gather(send_ptr, recv_ptr, 3, DataType::I32, 2)
                .await
                .unwrap();
        }

        if rank == 2 {
            let expected = vec![0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30];
            assert_eq!(recv_data, expected);
        }
    })
    .await;
}

// ============================================================================
// Scatter tests
// ============================================================================

#[tokio::test]
async fn test_scatter_3_nodes() {
    run_collective(3, |client| async move {
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
async fn test_scatter_4_nodes_nonzero_root() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let send_data: Vec<i32> = vec![100, 200, 300, 400];
        let mut recv_data: Vec<i32> = vec![0; 1];

        let send_ptr = send_data.as_ptr() as u64;
        let recv_ptr = recv_data.as_mut_ptr() as u64;

        unsafe {
            client
                .scatter(send_ptr, recv_ptr, 1, DataType::I32, 2)
                .await
                .unwrap();
        }

        let expected = match rank {
            0 => vec![100],
            1 => vec![200],
            2 => vec![300],
            3 => vec![400],
            _ => unreachable!(),
        };
        assert_eq!(
            recv_data, expected,
            "rank {rank} scatter nonzero-root failed"
        );
    })
    .await;
}
