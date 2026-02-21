//! Manual cluster setup with seed and worker nodes.
//!
//! Instead of `bootstrap_local`, this shows how to set up a cluster the way
//! you would across real machines: start a seed node, connect workers to it,
//! then form peer-to-peer connections.
//!
//! ```bash
//! cargo run --example seed_worker
//! ```

use nexar::transport::PeerConnection;
use nexar::{CpuAdapter, NexarClient, SeedNode, WorkerNode};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> nexar::Result<()> {
    let world_size = 2u32;

    // Step 1: Start the seed node on a local address.
    let seed_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let seed = SeedNode::bind(seed_addr, world_size)?;
    let seed_addr = seed.local_addr();
    println!("seed listening on {seed_addr}");

    // Step 2: Workers connect to the seed concurrently.
    let seed_handle = tokio::spawn(async move { seed.form_cluster().await });

    let mut workers = Vec::new();
    for _ in 0..world_size {
        let addr = seed_addr;
        workers.push(tokio::spawn(async move { WorkerNode::connect(addr).await }));
    }

    // Wait for cluster formation.
    let _seed_result = seed_handle.await.unwrap()?;

    let mut worker_nodes = Vec::new();
    for w in workers {
        worker_nodes.push(w.await.unwrap()?);
    }

    for w in &worker_nodes {
        println!(
            "worker rank={} world_size={} peers={:?}",
            w.rank, w.world_size, w.peers
        );
    }

    // Step 3: Establish peer-to-peer mesh connections using mTLS credentials
    // from the seed. (In a real deployment, each worker runs on a different
    // machine and connects to peers over the network.)
    let adapter: Arc<dyn nexar::DeviceAdapter> = Arc::new(CpuAdapter::new());
    let mut clients = Vec::new();

    for worker in &worker_nodes {
        let peer_map: HashMap<u32, PeerConnection> = HashMap::new();
        // In production, you'd iterate worker.peers and call
        // PeerConnection::connect() to each peer address using the mTLS
        // certs (worker.ca_cert, worker.node_cert, worker.node_key).
        // For this example, we show the structure.
        let client = NexarClient::new(
            worker.rank,
            worker.world_size,
            peer_map,
            Arc::clone(&adapter),
        );
        clients.push(client);
    }

    println!("cluster formed: {} nodes", clients.len());
    for c in &clients {
        println!("  rank {} / world_size {}", c.rank(), c.world_size());
    }

    Ok(())
}
