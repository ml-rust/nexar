use std::sync::Arc;

use nexar::compression::{Compressor, NoCompression, RandomKCompressor, TopKCompressor};
use nexar::{BufferRef, DataType, Host, ReduceOp};

use super::helpers::run_collective;

/// Helper to run compressed allreduce across N nodes and verify sum correctness.
async fn run_compressed_allreduce(world_size: u32, count: usize, compressor: Arc<dyn Compressor>) {
    run_collective(world_size, move |client| {
        let compressor = Arc::clone(&compressor);
        async move {
            let rank = client.rank();
            let val = (rank + 1) as f32;
            let mut data = vec![val; count];
            let mut buf = unsafe { BufferRef::<Host>::new(data.as_mut_ptr() as u64, count * 4) };
            let mut residual = vec![0u8; count * 4];

            client
                .all_reduce_compressed_host(
                    &mut buf,
                    count,
                    DataType::F32,
                    ReduceOp::Sum,
                    compressor.as_ref(),
                    &mut residual,
                )
                .await
                .unwrap();

            // Sum of 1..=world_size
            let expected = (world_size * (world_size + 1) / 2) as f32;
            assert_eq!(
                data,
                vec![expected; count],
                "rank {rank} compressed allreduce failed"
            );
        }
    })
    .await;
}

#[tokio::test]
async fn test_compressed_allreduce_no_compression_2_nodes() {
    run_compressed_allreduce(2, 8, Arc::new(NoCompression)).await;
}

#[tokio::test]
async fn test_compressed_allreduce_no_compression_3_nodes() {
    run_compressed_allreduce(3, 6, Arc::new(NoCompression)).await;
}

#[tokio::test]
async fn test_compressed_allreduce_topk_lossless_2_nodes() {
    run_compressed_allreduce(2, 4, Arc::new(TopKCompressor::new(1.0))).await;
}

#[tokio::test]
async fn test_compressed_allreduce_randomk_lossless_2_nodes() {
    run_compressed_allreduce(2, 4, Arc::new(RandomKCompressor::new(1.0))).await;
}
