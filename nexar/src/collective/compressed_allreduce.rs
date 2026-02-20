//! Compressed allreduce via allgather-then-reduce.
//!
//! Each rank compresses its data (with error feedback into `residual`),
//! broadcasts the compressed representation to all peers, then locally
//! decompresses and reduces all contributions. This avoids the
//! accumulation-of-sums problem that a naive ring approach would have
//! with compression.

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

    let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };

    // Compress local data with error feedback.
    let compressed = compressor.compress(&data, count, dtype, residual);
    let my_compressed = compressed.data;

    // Exchange compressed data with all peers using a ring pattern.
    // Each rank sends its compressed data to the next rank and receives
    // from the previous, repeated N-1 times so all ranks get all data.
    let my_rank = client.rank();
    let next = (my_rank + 1) % client.world_size();
    let prev = (my_rank + client.world_size() - 1) % client.world_size();

    let mut all_compressed = Vec::with_capacity(world);
    all_compressed.push(my_compressed.clone());

    // Forward compressed data around the ring: N-1 steps.
    let mut to_forward = my_compressed;
    for _step in 0..(world - 1) {
        let (send_result, recv_result) = tokio::join!(
            collective_send(client, next, &to_forward, "allreduce_compressed"),
            collective_recv(client, prev, "allreduce_compressed"),
        );
        send_result?;
        let received = recv_result?;
        to_forward = received.to_vec();
        all_compressed.push(to_forward.clone());
    }

    // Decompress all contributions and reduce locally.
    let mut result = vec![0u8; total_bytes];
    for (i, compressed_data) in all_compressed.into_iter().enumerate() {
        let ct = CompressedTensor {
            data: compressed_data,
            original_count: count,
            dtype,
        };
        let mut dense = vec![0u8; total_bytes];
        compressor.decompress(&ct, &mut dense);
        if i == 0 {
            result.copy_from_slice(&dense);
        } else {
            reduce_slice(&mut result, &dense, count, dtype, op)?;
        }
    }

    // Write the final reduced result back to device.
    unsafe { client.adapter().receive_to_device(&result, ptr)? };

    Ok(())
}
