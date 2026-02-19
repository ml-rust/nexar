use nexar::{CpuAdapter, DataType, NexarClient, ReduceOp};
use std::sync::Arc;
use tokio::sync::Barrier;

#[tokio::test]
async fn test_split_two_groups() {
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
