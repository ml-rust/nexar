//! Tree broadcast from root rank to all others.
//!
//! Rank 0 (root) holds the data. After broadcast, every rank has a copy.
//!
//! ```bash
//! cargo run --example broadcast
//! ```

use nexar::{CpuAdapter, DataType, NexarClient};
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

    let count = 4usize;
    let root: u32 = 0;

    let mut handles = Vec::new();
    for client in &clients {
        let c = Arc::clone(client);
        let rank = c.rank();
        handles.push(tokio::spawn(async move {
            // Root has real data, others start with zeros.
            let mut data = if rank == root {
                vec![42.0f32, 13.0, 7.0, 99.0]
            } else {
                vec![0.0f32; count]
            };

            let ptr = data.as_mut_ptr() as u64;
            unsafe {
                c.broadcast(ptr, count, DataType::F32, root).await?;
            }

            nexar::Result::Ok((rank, data))
        }));
    }

    for h in handles {
        let (rank, data) = h.await.unwrap()?;
        println!("rank {rank}: {data:?}");
    }
    // All ranks print: [42.0, 13.0, 7.0, 99.0]

    Ok(())
}
