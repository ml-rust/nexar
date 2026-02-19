//! Barrier synchronization across ranks.
//!
//! Each rank sleeps for a different duration, then hits the barrier. No rank
//! proceeds past the barrier until all ranks have arrived.
//!
//! ```bash
//! cargo run --example barrier
//! ```

use nexar::{CpuAdapter, NexarClient};
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> nexar::Result<()> {
    let world_size = 4u32;
    let adapter = Arc::new(CpuAdapter::new());
    let clients: Vec<Arc<NexarClient>> = NexarClient::bootstrap_local(world_size, adapter)
        .await?
        .into_iter()
        .map(Arc::new)
        .collect();

    let start = Instant::now();

    let mut handles = Vec::new();
    for client in &clients {
        let c = Arc::clone(client);
        let rank = c.rank();
        handles.push(tokio::spawn(async move {
            // Simulate different amounts of work per rank.
            tokio::time::sleep(std::time::Duration::from_millis(rank as u64 * 50)).await;
            println!(
                "rank {rank} arriving at barrier ({}ms elapsed)",
                start.elapsed().as_millis()
            );

            c.barrier().await?;

            println!(
                "rank {rank} passed barrier ({}ms elapsed)",
                start.elapsed().as_millis()
            );
            nexar::Result::Ok(())
        }));
    }

    for h in handles {
        h.await.unwrap()?;
    }

    Ok(())
}
