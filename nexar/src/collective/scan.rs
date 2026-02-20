use crate::client::NexarClient;
use crate::collective::helpers::{
    CollectiveTag, ceil_log2, collective_recv_with_tag, collective_send_with_tag,
};
use crate::error::{NexarError, Result};
use crate::reduce::{identity_slice, reduce_slice};
use crate::types::{DataType, ReduceOp};

/// Tagged variant for non-blocking collectives.
pub(crate) async unsafe fn inclusive_scan_with_tag(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;

    if world <= 1 {
        return Ok(());
    }

    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    let data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };
    let mut buf = data;

    let num_rounds = ceil_log2(world as u32) as usize;

    for round in 0..num_rounds {
        let distance = 1 << round;

        let should_send = rank + distance < world;
        let should_recv = rank >= distance;
        let source = rank.wrapping_sub(distance);

        match (should_send, should_recv) {
            (true, true) => {
                let dest = rank + distance;
                let send_data = buf.clone();
                let (_, received) = tokio::try_join!(
                    collective_send_with_tag(client, dest as u32, &send_data, "scan", tag),
                    collective_recv_with_tag(client, source as u32, "scan", tag),
                )?;
                if received.len() != total_bytes {
                    return Err(NexarError::BufferSizeMismatch {
                        expected: total_bytes,
                        actual: received.len(),
                    });
                }
                reduce_slice(&mut buf, &received, count, dtype, op)?;
            }
            (true, false) => {
                let dest = rank + distance;
                collective_send_with_tag(client, dest as u32, &buf, "scan", tag).await?;
            }
            (false, true) => {
                let received = collective_recv_with_tag(client, source as u32, "scan", tag).await?;
                if received.len() != total_bytes {
                    return Err(NexarError::BufferSizeMismatch {
                        expected: total_bytes,
                        actual: received.len(),
                    });
                }
                reduce_slice(&mut buf, &received, count, dtype, op)?;
            }
            (false, false) => {}
        }
    }

    unsafe { client.adapter().receive_to_device(&buf, ptr)? };

    Ok(())
}

/// Tagged variant for non-blocking collectives.
pub(crate) async unsafe fn exclusive_scan_with_tag(
    client: &NexarClient,
    ptr: u64,
    count: usize,
    dtype: DataType,
    op: ReduceOp,
    tag: CollectiveTag,
) -> Result<()> {
    let world = client.world_size() as usize;
    let rank = client.rank() as usize;
    let elem_size = dtype.size_in_bytes();
    let total_bytes = count * elem_size;

    if world <= 1 {
        let id = identity_slice(count, dtype, op)?;
        unsafe { client.adapter().receive_to_device(&id, ptr)? };
        return Ok(());
    }

    // Run inclusive scan in-place.
    unsafe { inclusive_scan_with_tag(client, ptr, count, dtype, op, tag).await? };

    // Shift results left by one rank.
    let inclusive_data = unsafe { client.adapter().stage_for_send(ptr, total_bytes)? };

    let should_send = rank + 1 < world;
    let should_recv = rank > 0;

    match (should_send, should_recv) {
        (true, true) => {
            let (_, received) = tokio::try_join!(
                collective_send_with_tag(client, (rank + 1) as u32, &inclusive_data, "exscan", tag),
                collective_recv_with_tag(client, (rank - 1) as u32, "exscan", tag),
            )?;
            if received.len() != total_bytes {
                return Err(NexarError::BufferSizeMismatch {
                    expected: total_bytes,
                    actual: received.len(),
                });
            }
            unsafe { client.adapter().receive_to_device(&received, ptr)? };
        }
        (true, false) => {
            collective_send_with_tag(client, (rank + 1) as u32, &inclusive_data, "exscan", tag)
                .await?;
            let id = identity_slice(count, dtype, op)?;
            unsafe { client.adapter().receive_to_device(&id, ptr)? };
        }
        (false, true) => {
            let received =
                collective_recv_with_tag(client, (rank - 1) as u32, "exscan", tag).await?;
            if received.len() != total_bytes {
                return Err(NexarError::BufferSizeMismatch {
                    expected: total_bytes,
                    actual: received.len(),
                });
            }
            unsafe { client.adapter().receive_to_device(&received, ptr)? };
        }
        (false, false) => unreachable!("world > 1"),
    }

    Ok(())
}
