use nexar::{BufferRef, DataType, Host, ReduceOp};

use super::helpers::run_collective;

#[tokio::test]
async fn test_scan_3_nodes_sum() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let mut data = vec![(rank + 1) as f32; 4];
        let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 4 * 4) };

        client
            .scan_host(&mut buf, 4, DataType::F32, ReduceOp::Sum)
            .await
            .unwrap();

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
        let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 2 * 4) };

        client
            .scan_host(&mut buf, 2, DataType::F32, ReduceOp::Sum)
            .await
            .unwrap();

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
        let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 4 * 4) };

        client
            .scan_host(&mut buf, 4, DataType::F32, ReduceOp::Max)
            .await
            .unwrap();

        let expected = match rank {
            0 => 1.0,
            1 => 2.0,
            _ => unreachable!(),
        };
        assert_eq!(data, vec![expected; 4], "rank {rank} scan max failed");
    })
    .await;
}

#[tokio::test]
async fn test_exclusive_scan_3_nodes_sum() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let mut data = vec![(rank + 1) as f32; 4];
        let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 4 * 4) };

        client
            .exclusive_scan_host(&mut buf, 4, DataType::F32, ReduceOp::Sum)
            .await
            .unwrap();

        let expected = match rank {
            0 => 0.0,
            1 => 1.0,
            2 => 3.0,
            _ => unreachable!(),
        };
        assert_eq!(data, vec![expected; 4], "rank {rank} exclusive_scan failed");
    })
    .await;
}

#[tokio::test]
async fn test_exclusive_scan_4_nodes_prod() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let mut data = vec![(rank + 1) as f32; 2];
        let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 2 * 4) };

        client
            .exclusive_scan_host(&mut buf, 2, DataType::F32, ReduceOp::Prod)
            .await
            .unwrap();

        let expected = match rank {
            0 => 1.0,
            1 => 1.0,
            2 => 2.0,
            3 => 6.0,
            _ => unreachable!(),
        };
        assert_eq!(
            data,
            vec![expected; 2],
            "rank {rank} exclusive_scan prod failed"
        );
    })
    .await;
}
