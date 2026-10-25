//! A cluster of separate processes: one seed process and N-1 workers.
//!
//! Start the seed (rank 0) first, then one `join` per remaining rank, each in
//! its own terminal or on its own host:
//!
//! ```bash
//! export NEXAR_CLUSTER_TOKEN=change-me
//! cargo run -p nexar --example seed_worker -- seed 127.0.0.1:7000 2
//! cargo run -p nexar --example seed_worker -- join 127.0.0.1:7000 2
//! ```
//!
//! Across hosts, use the seed host's IP as the seed address, and pass this
//! host's IP as the optional fourth argument so peers can dial it:
//!
//! ```bash
//! cargo run -p nexar --example seed_worker -- seed 10.0.0.1:7000 2 10.0.0.1:7001
//! cargo run -p nexar --example seed_worker -- join 10.0.0.1:7000 2 10.0.0.2:7001
//! ```
//!
//! The fourth argument is the mesh bind address. Its port carries QUIC (UDP)
//! and the TCP bulk sidecar. Every process must use the same token and world size.

use nexar::{ClusterConfig, CpuAdapter, DataType, NexarClient, ReduceOp};
use std::net::SocketAddr;
use std::sync::Arc;

fn usage() -> ! {
    eprintln!("usage: seed_worker <seed|join> <seed-addr> <world-size> [mesh-bind-addr]");
    std::process::exit(2);
}

#[tokio::main]
async fn main() -> nexar::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        usage();
    }
    let seed_addr: SocketAddr = args[1].parse().unwrap_or_else(|_| usage());
    let world_size: u32 = args[2].parse().unwrap_or_else(|_| usage());
    let bind: SocketAddr = match args.get(3) {
        Some(a) => a.parse().unwrap_or_else(|_| usage()),
        None => SocketAddr::new(seed_addr.ip(), 0),
    };

    let cfg = ClusterConfig::new(world_size).with_bind_addr(bind);
    let adapter = Arc::new(CpuAdapter::new());
    let client = match args[0].as_str() {
        "seed" => NexarClient::bootstrap_seed(seed_addr, &cfg, adapter).await?,
        "join" => NexarClient::bootstrap_join(seed_addr, &cfg, adapter).await?,
        _ => usage(),
    };
    println!(
        "rank {} of {} joined the mesh",
        client.rank(),
        client.world_size()
    );

    let mut data = vec![(client.rank() + 1) as f32; 4];
    unsafe {
        client
            .all_reduce(
                data.as_mut_ptr() as u64,
                data.len(),
                DataType::F32,
                ReduceOp::Sum,
            )
            .await?;
    }
    println!("rank {} allreduce result: {data:?}", client.rank());

    // Shut down before exit: closing a connection discards data peers have not
    // acknowledged. It returns once every peer shut down too.
    client.shutdown().await
}
