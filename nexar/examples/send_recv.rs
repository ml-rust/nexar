//! Point-to-point send/recv between two ranks.
//!
//! Spawns a 2-node local cluster. Rank 0 sends a float vector to rank 1,
//! which receives it and prints the result.
//!
//! ```bash
//! cargo run --example send_recv
//! ```

use nexar::{CpuAdapter, NexarClient};
use std::sync::Arc;

#[tokio::main]
async fn main() -> nexar::Result<()> {
    let adapter = Arc::new(CpuAdapter::new());
    let clients: Vec<Arc<NexarClient>> = NexarClient::bootstrap_local(2, adapter)
        .await?
        .into_iter()
        .map(Arc::new)
        .collect();

    let c0 = Arc::clone(&clients[0]);
    let c1 = Arc::clone(&clients[1]);

    // Rank 0 sends, rank 1 receives. Tag 42 matches sender and receiver.
    let send_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let size = send_data.len() * std::mem::size_of::<f32>();
    let send_ptr = send_data.as_ptr() as u64;

    let mut recv_buf: Vec<f32> = vec![0.0; 4];
    let recv_ptr = recv_buf.as_mut_ptr() as u64;

    // send and recv must run concurrently — send blocks until the stream is
    // written, recv blocks until data arrives.
    let sender = tokio::spawn(async move { unsafe { c0.send(send_ptr, size, 1, 42).await } });
    let receiver = tokio::spawn(async move { unsafe { c1.recv(recv_ptr, size, 0, 42).await } });

    sender.await.unwrap()?;
    receiver.await.unwrap()?;

    println!("rank 1 received: {recv_buf:?}");
    // Output: rank 1 received: [1.0, 2.0, 3.0, 4.0]

    Ok(())
}
