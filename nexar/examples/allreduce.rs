//! Ring-allreduce across 4 ranks.
//!
//! Each rank starts with its own data. After allreduce(Sum), every rank holds
//! the element-wise sum of all inputs.
//!
//! ```bash
//! cargo run --example allreduce
//! ```

use nexar::{CpuAdapter, DataType, NexarClient, ReduceOp};
use std::sync::Arc;

#[tokio::main]
async fn main() -> nexar::Result<()> {
    let world_size = 4u32;
    let adapter = Arc::new(CpuAdapter::new());
    let clients: Vec<Arc<NexarClient>> = NexarClient::bootstrap_local(world_size, adapter)
        .await?
        .into_iter()
        .map(Arc::new)
        .collect();

    let count = 8usize;

    // Each rank fills its buffer with its own rank value.
    // rank 0: [0.0, 0.0, ...], rank 1: [1.0, 1.0, ...], etc.
    let mut handles = Vec::new();
    for client in &clients {
        let c = Arc::clone(client);
        let rank = c.rank();
        handles.push(tokio::spawn(async move {
            let mut data = vec![rank as f32; count];
            let ptr = data.as_mut_ptr() as u64;

            unsafe {
                c.all_reduce(ptr, count, DataType::F32, ReduceOp::Sum)
                    .await?;
            }

            // After Sum allreduce: each element = 0 + 1 + 2 + 3 = 6.0
            nexar::Result::Ok((rank, data))
        }));
    }

    for h in handles {
        let (rank, data) = h.await.unwrap()?;
        println!("rank {rank}: {data:?}");
    }
    // Output (all ranks identical):
    // rank 0: [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    // rank 1: [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    // ...

    Ok(())
}
