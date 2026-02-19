use nexar::{DataType, ReduceOp};

use super::helpers::run_collective;

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
            assert_eq!(data, vec![3.0f32; 4], "root reduce failed");
        }
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
            assert_eq!(data, vec![6.0f32; 4], "root reduce nonzero-root failed");
        }
    })
    .await;
}

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

        assert_eq!(
            recv_data,
            vec![6.0, 6.0],
            "rank {rank} reduce_scatter 3-node failed"
        );
    })
    .await;
}
