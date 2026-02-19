use crate::client::NexarClient;
use crate::error::{NexarError, Result};
use crate::types::{DataType, ReduceOp};

/// Ring reduce-scatter: reduce across all ranks, each rank gets a different
/// slice of the result.
///
/// Input: each rank has `count * world_size` elements at `send_ptr`.
/// Output: rank `i` gets the reduced chunk `i` (of size `count`) at `recv_ptr`.
///
/// # Safety
/// - `send_ptr` must point to at least `count * world_size * dtype.size_in_bytes()` bytes.
/// - `recv_ptr` must point to at least `count * dtype.size_in_bytes()` bytes.
pub async unsafe fn ring_reduce_scatter(
    client: &NexarClient,
    send_ptr: u64,
    recv_ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;

    let elem_size = dtype.size_in_bytes();
    let chunk_bytes = count * elem_size;
    let total_bytes = chunk_bytes * world;

    if world <= 1 {
        // Single node: copy the rank-0 slice.
        let data = unsafe { client.adapter().stage_for_send(send_ptr, chunk_bytes)? };
        unsafe { client.adapter().receive_to_device(&data, recv_ptr)? };
        return Ok(());
    }

    // Read entire send buffer.
    let data = unsafe { client.adapter().stage_for_send(send_ptr, total_bytes)? };
    let mut buf = data;

    let next = (rank + 1) % world;
    let prev = (rank + world - 1) % world;

    // N-1 rounds of scatter-reduce (same as allreduce phase 1).
    for step in 0..(world - 1) {
        let send_idx = (rank + world - step) % world;
        let recv_idx = (rank + world - step - 1) % world;

        let send_off = send_idx * chunk_bytes;
        let recv_off = recv_idx * chunk_bytes;

        let send_data = buf[send_off..send_off + chunk_bytes].to_vec();

        let (send_result, recv_result) = tokio::join!(
            client.send_bytes(next as u32, &send_data),
            client.recv_bytes(prev as u32),
        );
        send_result?;
        let received = recv_result?;

        // Validate received length before reducing.
        if received.len() != chunk_bytes {
            return Err(NexarError::BufferSizeMismatch {
                expected: chunk_bytes,
                actual: received.len(),
            });
        }
        reduce_slice(
            &mut buf[recv_off..recv_off + chunk_bytes],
            &received,
            count,
            dtype,
            op,
        )?;
    }

    // Our result is the chunk at position (rank + 1) % world after scatter-reduce.
    let result_idx = (rank + 1) % world;
    let result_off = result_idx * chunk_bytes;
    let result = &buf[result_off..result_off + chunk_bytes];

    unsafe { client.adapter().receive_to_device(result, recv_ptr)? };

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
    fn read_le(bytes: &[u8]) -> Self;
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
                    Self::from_le_bytes(bytes.try_into().unwrap())
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
                    Self::from_le_bytes(bytes.try_into().unwrap())
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
