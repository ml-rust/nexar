use nexar::{BufferRef, DataType, Host};

use super::helpers::run_collective;

#[tokio::test]
async fn test_broadcast_from_root_0() {
    run_collective(3, |client| async move {
        let rank = client.rank();
        let mut data: Vec<f32> = if rank == 0 {
            vec![42.0, 43.0, 44.0, 45.0]
        } else {
            vec![0.0; 4]
        };
        let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 4 * 4) };

        client
            .broadcast_host(&mut buf, 4, DataType::F32, 0)
            .await
            .unwrap();

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
        let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 3 * 4) };

        client
            .broadcast_host(&mut buf, 3, DataType::F32, root)
            .await
            .unwrap();

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
    run_collective(5, |client| async move {
        let rank = client.rank();
        let mut data: Vec<i32> = if rank == 0 { vec![7, 8, 9] } else { vec![0; 3] };
        let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, 3 * 4) };

        client
            .broadcast_host(&mut buf, 3, DataType::I32, 0)
            .await
            .unwrap();

        assert_eq!(data, vec![7, 8, 9], "rank {rank} tree broadcast failed");
    })
    .await;
}
