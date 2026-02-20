use nexar::{DataType, ReduceOp};

use super::helpers::run_collective;

/// RS-AG allreduce with count evenly divisible by world size.
#[tokio::test]
async fn test_rs_ag_allreduce_divisible() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let val = (rank + 1) as f32;
        let mut data = vec![val; 6]; // 6 / 3 = 2 per rank
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce_rs_ag(ptr, 6, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // 1 + 2 + 3 = 6
        assert_eq!(data, vec![6.0f32; 6], "rank {rank} rs_ag divisible failed");
    })
    .await;
}

/// RS-AG allreduce with count NOT divisible by world size (7 elements, 3 ranks).
#[tokio::test]
async fn test_rs_ag_allreduce_indivisible() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let mut data: Vec<f32> = (0..7).map(|i| (i as f32) * ((rank + 1) as f32)).collect();
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce_rs_ag(ptr, 7, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // Sum of rank multipliers: 1 + 2 + 3 = 6
        let expected: Vec<f32> = (0..7).map(|i| (i as f32) * 6.0).collect();
        assert_eq!(data, expected, "rank {rank} rs_ag indivisible failed");
    })
    .await;
}

/// RS-AG allreduce with 4 ranks and 5 elements (remainder = 1).
#[tokio::test]
async fn test_rs_ag_allreduce_4_ranks_5_elements() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let val = (rank + 1) as f32;
        let mut data = vec![val; 5];
        let ptr = data.as_mut_ptr() as u64;

        unsafe {
            client
                .all_reduce_rs_ag(ptr, 5, DataType::F32, ReduceOp::Sum)
                .await
                .unwrap();
        }

        // 1 + 2 + 3 + 4 = 10
        assert_eq!(data, vec![10.0f32; 5], "rank {rank} rs_ag 4x5 failed");
    })
    .await;
}
