use crate::client::NexarClient;
use crate::collective::{collective_recv, collective_send};
use crate::error::{NexarError, Result};
use crate::types::{DataType, ReduceOp};

/// Ring-allreduce: in-place reduce across all ranks.
///
/// Algorithm:
/// 1. Scatter-reduce: N-1 rounds. Each rank sends one chunk to the next rank
///    and receives one chunk from the previous rank, reducing in-place.
/// 2. Allgather: N-1 rounds. Each rank sends its fully-reduced chunk to the
///    next rank and receives from the previous rank.
///
/// After completion, `ptr` on every rank contains the sum (or other op) of
/// all ranks' original data.
///
/// # Safety
/// `ptr` must be valid for at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn ring_allreduce(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;

    if world <= 1 {
        return Ok(());
    }

    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    // Read the entire buffer into a local working copy.
    let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };
    let mut buf = data;

    // Compute chunk boundaries. Handle uneven division by giving the last
    // chunk any remainder.
    let base_chunk = count / world;
    let remainder = count % world;

    let chunk_count = |i: usize| -> usize {
        if i < remainder {
            base_chunk + 1
        } else {
            base_chunk
        }
    };

    let chunk_offset = |i: usize| -> usize {
        let mut off = 0;
        for j in 0..i {
            off += chunk_count(j);
        }
        off
    };

    let next = (rank + 1) % world;
    let prev = (rank + world - 1) % world;

    // Phase 1: Scatter-reduce (N-1 rounds).
    for step in 0..(world - 1) {
        // Send chunk index: rank - step (mod world).
        let send_idx = (rank + world - step) % world;
        let send_off = chunk_offset(send_idx) * elem_size;
        let send_len = chunk_count(send_idx) * elem_size;

        // Recv chunk index: rank - step - 1 (mod world).
        let recv_idx = (rank + world - step - 1) % world;
        let recv_off = chunk_offset(recv_idx) * elem_size;
        let recv_count = chunk_count(recv_idx);
        let recv_len = recv_count * elem_size;

        let send_data = buf[send_off..send_off + send_len].to_vec();

        // Concurrent send + recv.
        let (send_result, recv_result) = tokio::join!(
            collective_send(client, next as u32, &send_data, "allreduce"),
            collective_recv(client, prev as u32, "allreduce"),
        );
        send_result?;
        let received = recv_result?;

        // Reduce received data into our buffer at recv_idx position.
        if received.len() != recv_len {
            return Err(NexarError::BufferSizeMismatch {
                expected: recv_len,
                actual: received.len(),
            });
        }
        let dst_slice = &mut buf[recv_off..recv_off + recv_len];

        // Reduce in-place on the local buffer.
        reduce_slice(dst_slice, &received, recv_count, dtype, op)?;
    }

    // Phase 2: Allgather (N-1 rounds).
    // After scatter-reduce, rank r's fully reduced chunk is at index (r+1)%N.
    // Each round, forward the last received chunk around the ring.
    for step in 0..(world - 1) {
        let send_idx = (rank + world + 1 - step) % world;
        let send_off = chunk_offset(send_idx) * elem_size;
        let send_len = chunk_count(send_idx) * elem_size;

        let recv_idx = (rank + world - step) % world;
        let recv_off = chunk_offset(recv_idx) * elem_size;
        let recv_len = chunk_count(recv_idx) * elem_size;

        let send_data = buf[send_off..send_off + send_len].to_vec();

        let (send_result, recv_result) = tokio::join!(
            collective_send(client, next as u32, &send_data, "allreduce"),
            collective_recv(client, prev as u32, "allreduce"),
        );
        send_result?;
        let received = recv_result?;

        if received.len() != recv_len {
            return Err(NexarError::BufferSizeMismatch {
                expected: recv_len,
                actual: received.len(),
            });
        }
        buf[recv_off..recv_off + recv_len].copy_from_slice(&received);
    }

    // Write result back to device memory.
    unsafe { client.adapter().receive_to_device(&buf, ptr)? };

    Ok(())
}

/// Element-wise reduce on byte slices.
fn reduce_slice(
    dst: &mut [u8],
    src: &[u8],
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    match dtype {
        DataType::F32 => reduce_typed::<f32>(dst, src, count, op),
        DataType::F64 => reduce_typed::<f64>(dst, src, count, op),
        DataType::I32 => reduce_typed::<i32>(dst, src, count, op),
        DataType::I64 => reduce_typed::<i64>(dst, src, count, op),
        DataType::U32 => reduce_typed::<u32>(dst, src, count, op),
        DataType::U64 => reduce_typed::<u64>(dst, src, count, op),
        DataType::I8 => reduce_typed::<i8>(dst, src, count, op),
        DataType::U8 => reduce_typed::<u8>(dst, src, count, op),
        _ => {
            return Err(NexarError::UnsupportedDType {
                dtype,
                op: "reduce",
            });
        }
    }
    Ok(())
}

trait Reducible: Copy + 'static {
    fn reduce(a: Self, b: Self, op: ReduceOp) -> Self;
    /// Read a value from a little-endian byte slice (alignment-safe).
    fn read_le(bytes: &[u8]) -> Self;
    /// Write a value to a little-endian byte slice (alignment-safe).
    fn write_le(self, bytes: &mut [u8]);
}

macro_rules! impl_reducible {
    (int: $($ty:ty),*) => {
        $(
            impl Reducible for $ty {
                #[inline]
                fn reduce(a: Self, b: Self, op: ReduceOp) -> Self {
                    match op {
                        ReduceOp::Sum => a.wrapping_add(b),
                        ReduceOp::Prod => a.wrapping_mul(b),
                        ReduceOp::Min => a.min(b),
                        ReduceOp::Max => a.max(b),
                    }
                }
                #[inline]
                fn read_le(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(bytes.try_into().expect("slice length matches type size"))
                }
                #[inline]
                fn write_le(self, bytes: &mut [u8]) {
                    bytes.copy_from_slice(&self.to_le_bytes());
                }
            }
        )*
    };
    (float: $($ty:ty),*) => {
        $(
            impl Reducible for $ty {
                #[inline]
                fn reduce(a: Self, b: Self, op: ReduceOp) -> Self {
                    match op {
                        ReduceOp::Sum => a + b,
                        ReduceOp::Prod => a * b,
                        ReduceOp::Min => a.min(b),
                        ReduceOp::Max => a.max(b),
                    }
                }
                #[inline]
                fn read_le(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(bytes.try_into().expect("slice length matches type size"))
                }
                #[inline]
                fn write_le(self, bytes: &mut [u8]) {
                    bytes.copy_from_slice(&self.to_le_bytes());
                }
            }
        )*
    };
}

impl_reducible!(int: i8, i32, i64, u8, u32, u64);
impl_reducible!(float: f32, f64);

fn reduce_typed<T: Reducible>(dst: &mut [u8], src: &[u8], count: usize, op: ReduceOp) {
    let t_size = std::mem::size_of::<T>();
    for i in 0..count {
        let off = i * t_size;
        let a = T::read_le(&dst[off..off + t_size]);
        let b = T::read_le(&src[off..off + t_size]);
        let r = T::reduce(a, b, op);
        r.write_le(&mut dst[off..off + t_size]);
    }
}
