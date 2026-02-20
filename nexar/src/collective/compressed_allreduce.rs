//! Compressed allreduce via allgather-then-reduce.
//!
//! Each rank compresses its data (with error feedback into `residual`),
//! broadcasts the compressed representation to all peers, then locally
//! decompresses and reduces all contributions. This avoids the
//! accumulation-of-sums problem that a naive ring approach would have
//! with compression.
//!
//! # Memory complexity
//!
//! Each chunk is decompressed and reduced into a running accumulator
//! as it arrives, so only one compressed chunk plus one dense buffer
//! are live at any time — `O(compressed_chunk_size + tensor_size)`
//! memory per rank regardless of world size.

use crate::client::NexarClient;
use crate::collective::helpers::{collective_recv, collective_send};
use crate::compression::{CompressedTensor, Compressor};
use crate::error::Result;
use crate::reduce::reduce_slice;
use crate::types::{DataType, ReduceOp};

/// Compressed allreduce via allgather-then-reduce.
///
/// # Safety
/// - `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
/// - `residual` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn ring_allreduce_compressed(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    compressor: &dyn Compressor,
    residual: &mut [u8],
) -> Result<()> {
    let world = client.world_size() as usize;

    if world <= 1 {
        return Ok(());
    }

    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    // Memory guard: worst-case the algorithm holds world × total_bytes
    // for compressed chunks plus world × total_bytes for decompression.
    let max_bytes = client.config().compressed_allreduce_max_bytes;
    if max_bytes > 0 {
        let estimated = world.saturating_mul(total_bytes).saturating_mul(2);
        if estimated > max_bytes {
            return Err(crate::error::NexarError::CollectiveFailed {
                operation: "allreduce_compressed",
                rank: client.rank(),
                reason: format!(
                    "estimated memory {estimated} bytes ({world} ranks × {total_bytes} bytes × 2) \
                     exceeds compressed_allreduce_max_bytes limit ({max_bytes} bytes). \
                     Use uncompressed ring allreduce or nexar-nccl's hierarchical allreduce instead, \
                     or raise the limit via NEXAR_COMPRESSED_ALLREDUCE_MAX_BYTES"
                ),
            });
        }
    }

    let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };

    // Compress local data with error feedback.
    let compressed = compressor.compress(&data, count, dtype, residual);
    let my_compressed = compressed.data;

    // Decompress our own contribution into the running accumulator.
    let ct_local = CompressedTensor {
        data: my_compressed.clone(),
        original_count: count,
        dtype,
    };
    let mut result = vec![0u8; total_bytes];
    compressor.decompress(&ct_local, &mut result);

    // Forward compressed data around the ring: N-1 steps.
    // Each received chunk is decompressed and reduced immediately,
    // so only one compressed chunk is live at a time.
    let my_rank = client.rank();
    let next = (my_rank + 1) % client.world_size();
    let prev = (my_rank + client.world_size() - 1) % client.world_size();

    let mut to_forward = my_compressed;
    let mut dense_tmp = vec![0u8; total_bytes];
    for _step in 0..(world - 1) {
        let (_, received) = tokio::try_join!(
            collective_send(client, next, &to_forward, "allreduce_compressed"),
            collective_recv(client, prev, "allreduce_compressed"),
        )?;
        to_forward = received.to_vec();

        // Decompress and reduce into accumulator immediately.
        let ct = CompressedTensor {
            data: to_forward.clone(),
            original_count: count,
            dtype,
        };
        compressor.decompress(&ct, &mut dense_tmp);
        reduce_slice(&mut result, &dense_tmp, count, dtype, op)?;
    }

    // Write the final reduced result back to device.
    unsafe { client.adapter().receive_to_device(&result, ptr)? };

    Ok(())
}
