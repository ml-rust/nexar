use nexar::{DataType, ReduceOp};

use super::helpers::run_collective;

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

        assert_eq!(data, vec![10i32; 8], "rank {rank} allreduce failed");
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_uneven_count() {
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

        assert_eq!(data, vec![3.0f32; 4], "rank {rank} allreduce max failed");
    })
    .await;
}

#[tokio::test]
async fn test_allreduce_prod_2_nodes() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        let val = (rank + 2) as f32;
        let mut data = vec![val; 4];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce(ptr, 4, DataType::F32, ReduceOp::Prod)
                .await
                .unwrap();
        }

        assert_eq!(data, vec![6.0f32; 4], "rank {rank} allreduce prod failed");
    })
    .await;
}
