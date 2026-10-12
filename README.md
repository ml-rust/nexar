<div align="center">

# nexar

<h3>Distributed runtime for Rust. QUIC transport and 11 collectives, with no MPI.</h3>

<p>
Replaces MPI for inter-node communication. Point-to-point messages, collectives, and RPC run over QUIC without `mpirun` or an MPI library.
</p>

<p>
  <a href="https://docs.rs/nexar"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/nexar"><strong>Crate</strong></a>
  ·
  <a href="#collectives"><strong>Collectives</strong></a>
  ·
  <a href="#architecture"><strong>Architecture</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://github.com/ml-rust/nexar/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ml-rust/nexar/ci.yml?branch=main&label=ci" alt="CI status">
  </a>
  <a href="https://crates.io/crates/nexar">
    <img src="https://img.shields.io/crates/v/nexar" alt="crates.io">
  </a>
  <a href="https://docs.rs/nexar">
    <img src="https://img.shields.io/docsrs/nexar" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/nexar" alt="License">
  </a>
  <a href="https://github.com/ml-rust/nexar/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/nexar?style=social" alt="GitHub stars">
  </a>
</p>

</div>

nexar handles the network layer for distributed Rust programs. It provides point-to-point transfers, collectives, elastic membership, and fault recovery. Workers find each other through a seed node and form a direct peer-to-peer mesh.

## Quick Start

Add the crates to your project:

```bash
cargo add nexar
cargo add tokio --features full
```

Start a 4-rank local cluster and run an in-place allreduce:

```rust
use nexar::{NexarClient, CpuAdapter, DataType, ReduceOp};
use std::sync::Arc;

#[tokio::main]
async fn main() -> nexar::Result<()> {
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(4, adapter).await?;

    // Each client holds rank 0..3 with connections to all peers.
    let client = &clients[0];
    assert_eq!(client.rank(), 0);
    assert_eq!(client.world_size(), 4);

    // Run allreduce in-place across all ranks.
    let mut data = vec![1.0f32; 1024];
    unsafe {
        client.all_reduce(
            data.as_mut_ptr() as u64,
            1024,
            DataType::F32,
            ReduceOp::Sum,
        ).await?;
    }

    Ok(())
}
```

The raw-pointer collective methods are `unsafe`. The caller must keep the buffer valid and sized for `count` elements of `dtype` until the call returns.

## Collectives

`NexarClient` exposes 11 collectives. Each one has an `_nb` variant that returns a `CollectiveHandle`. Call `wait` on the handle to finish the operation.

| Operation        | Algorithm                                                            |
| ---------------- | -------------------------------------------------------------------- |
| `all_reduce`     | Ring. Basic ring for small worlds, pipelined ring for large messages |
| `broadcast`      | Flat fan-out for small worlds, tree for larger worlds                |
| `all_gather`     | Ring                                                                 |
| `reduce_scatter` | Ring                                                                 |
| `reduce`         | Binomial tree                                                        |
| `all_to_all`     | Each rank sends a distinct chunk to every other rank                 |
| `gather`         | Root receives from all peers concurrently                            |
| `scatter`        | Root sends one chunk to each rank                                    |
| `scan`           | Hillis-Steele prefix, O(log N) rounds                                |
| `exclusive_scan` | Exclusive variant of `scan`                                          |
| `barrier`        | Two-phase below 5 ranks, dissemination at 5 or more ranks            |

The client also exposes `all_reduce_host`, `all_reduce_device`, and the other `_host` and `_device` variants. The `_device` variants take memory managed by a `DeviceAdapter`.

## Features

- **Point-to-point:** tagged `send` and `recv` between any two ranks.
- **Elastic membership:** `rebuild_adding` and `rebuild_excluding` change the communicator at run time.
- **Fault recovery:** `RecoveryPolicy::Automatic`, `Manual`, or `Abort`. Healthy ranks agree on the failed set and rebuild.
- **Sparse topologies:** `TopologyStrategy::FullMesh`, `KRegular { degree }`, and `Hypercube`. Traffic to non-neighbors relays through peers.
- **Compressed allreduce:** `TopKCompressor` and random-K gradient compression.
- **RPC:** `register_rpc` registers a handler by function ID. `rpc` and `rpc_with_timeout` call it on a rank. Responses match requests by request ID.
- **Device adapters:** the `DeviceAdapter` trait stages device memory for network I/O. `CpuAdapter` ships with nexar. `CudaAdapter` ships with `nexar-rdma` behind the `gpudirect` feature.

## Examples

Runnable examples live in [`nexar/examples/`](nexar/examples/):

| Example                                        | What it shows                                         |
| ---------------------------------------------- | ----------------------------------------------------- |
| [`send_recv`](nexar/examples/send_recv.rs)     | Point-to-point tagged send and recv between two ranks |
| [`allreduce`](nexar/examples/allreduce.rs)     | Ring allreduce (Sum) across 4 ranks                   |
| [`broadcast`](nexar/examples/broadcast.rs)     | Tree broadcast from root to all ranks                 |
| [`barrier`](nexar/examples/barrier.rs)         | Barrier synchronization with staggered arrivals       |
| [`rpc`](nexar/examples/rpc.rs)                 | Register a remote function and call it across ranks   |
| [`seed_worker`](nexar/examples/seed_worker.rs) | Manual cluster setup with seed and worker nodes       |

Run an example from the repository root:

```bash
cargo run -p nexar --example send_recv
cargo run -p nexar --example allreduce
```

## Architecture

```
seed node (discovery only, no data routing)
    │
    ├── worker 0 ──── worker 1
    │       \            /
    │        \          /
    │         worker 2 ── worker 3
    │              ...
    └── direct peer-to-peer mesh
```

Workers connect to the seed to get a rank and a peer list. They then open direct QUIC connections to their peers. The seed is not on the data path.

Each peer connection runs a router. The router accepts incoming QUIC streams and sends each one to a typed lane:

| Lane            | Traffic                            | Consumer                      |
| --------------- | ---------------------------------- | ----------------------------- |
| `rpc_requests`  | Incoming RPC calls                 | Dispatcher serve loop         |
| `rpc_responses` | RPC replies, matched by request ID | `rpc()` caller via oneshot    |
| `control`       | Barrier, heartbeat, join and leave | Barrier logic, health monitor |
| `data`          | Point-to-point `send` and `recv`   | Application code              |
| `raw`           | Bulk byte streams                  | Tensor transfers              |

Each lane has its own bounded channel.

### Stream protocol

Each QUIC unidirectional stream starts with a 1-byte tag:

| Tag    | Stream content                                                                |
| ------ | ----------------------------------------------------------------------------- |
| `0x01` | Framed message: 8-byte little-endian length, then a serialized `NexarMessage` |
| `0x02` | Raw bytes: 8-byte little-endian length, then the payload                      |
| `0x03` | Raw bytes for a split communicator: `comm_id` prefix, then length and payload |
| `0x04` | Raw bytes for a tagged send: tag prefix, then length and payload              |

Messages use rkyv serialization.

## Crates

| Crate                      | What it provides                                             | Native dependencies                  |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------ |
| `nexar`                    | Core runtime: QUIC transport, collectives, RPC               | `ring` builds bundled C and assembly |
| `nexar-rdma`               | RDMA transport for InfiniBand and RoCE                       | `libibverbs` (rdma-core)             |
| `nexar-rdma` + `gpudirect` | GPU memory to RDMA NIC, `CudaAdapter`                        | `libibverbs` and CUDA runtime        |
| `nexar-nccl`               | Hierarchical communicator: NCCL intra-node, nexar inter-node | CUDA runtime and NCCL                |

Build each crate from the repository root:

```bash
cargo build -p nexar --release
cargo build -p nexar-rdma --release
cargo build -p nexar-rdma --release --features gpudirect
cargo build -p nexar-nccl --release
```

Requires Rust 1.89 or later. See [CONTRIBUTING.md](CONTRIBUTING.md) for test and lint commands.

## When to Use nexar

**Use nexar when:**

- You need collectives such as allreduce or broadcast across machines.
- You want async, non-blocking communication in Rust.
- You want to avoid MPI installation, `mpirun`, and C FFI.
- You are building distributed ML training or inference.

**Use NCCL directly when:**

- All GPUs sit in one machine. NCCL uses the local interconnect without a network hop.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
