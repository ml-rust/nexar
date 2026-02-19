# nexar

[![Crates.io](https://img.shields.io/crates/v/nexar.svg)](https://crates.io/crates/nexar) [![docs.rs](https://docs.rs/nexar/badge.svg)](https://docs.rs/nexar) [![CI](https://github.com/ml-rust/nexar/actions/workflows/ci.yml/badge.svg)](https://github.com/ml-rust/nexar/actions) [![License](https://img.shields.io/crates/l/nexar.svg)](LICENSE)

Distributed runtime for Rust. QUIC transport, stream-multiplexed messaging, built-in collectives. No C dependencies.

nexar replaces MPI for inter-node communication. It handles the network layer — point-to-point transfers, allreduce, broadcast, barrier — so your distributed application doesn't have to shell out to `mpirun` or link against libfabric.

## Why not MPI?

MPI works. It's also a C library with decades of accumulated complexity, a rigid process launcher, TCP-based transports that suffer from head-of-line blocking, and an implicit assumption that you'll manage serialization yourself.

nexar takes a different approach:

- **QUIC transport** (via quinn). Multiplexed streams mean a stalled tensor transfer doesn't block your barrier. TLS is built into the protocol.
- **No process launcher.** A lightweight seed node handles discovery. Workers connect, get a rank, and form a direct peer-to-peer mesh. Nodes can join and leave.
- **No C dependencies.** Pure Rust, compiles with `cargo build`. No `libmpi`, no `libfabric`, no `libucp`.
- **Async-native.** Built on tokio. Send and receive overlap naturally.

## What it provides

**Point-to-point:**

- `send` / `recv` — tagged messages between any two ranks

**Collectives:**

- `ring_allreduce` — scatter-reduce + allgather over a ring
- `tree_broadcast` — fan-out from root
- `ring_allgather` — ring-based gather
- `ring_reduce_scatter` — ring-based reduce-scatter
- `two_phase_barrier` — distributed synchronization with timeout

**RPC:**

- Register handlers by function ID, call them by rank. Responses are matched per-request, so concurrent RPCs don't interfere.

**Device abstraction:**

- `DeviceAdapter` trait lets GPU backends stage memory for network I/O without nexar knowing anything about CUDA or ROCm. A `CpuAdapter` is included.

## Quick start

Add to `Cargo.toml`:

```toml
[dependencies]
nexar = "<use-latest-version>"
tokio = { version = "1", features = ["full"] }
```

Bootstrap a local cluster and run allreduce:

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

## Examples

Runnable examples in [`examples/`](examples/):

| Example                                  | What it shows                                                         |
| ---------------------------------------- | --------------------------------------------------------------------- |
| [`send_recv`](examples/send_recv.rs)     | Point-to-point tagged send/recv between two ranks                     |
| [`allreduce`](examples/allreduce.rs)     | Ring-allreduce (Sum) across 4 ranks                                   |
| [`broadcast`](examples/broadcast.rs)     | Tree broadcast from root to all ranks                                 |
| [`barrier`](examples/barrier.rs)         | Barrier synchronization with staggered arrivals                       |
| [`rpc`](examples/rpc.rs)                 | Register a remote function, call it across ranks                      |
| [`seed_worker`](examples/seed_worker.rs) | Manual cluster setup with seed/worker nodes (real deployment pattern) |

Run any example with:

```bash
cargo run --example send_recv
cargo run --example allreduce
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

Workers connect to the seed to get a rank and peer list, then establish direct QUIC connections to every other worker. The seed is not on the data path.

Each peer connection runs a **router** — a background task that accepts incoming QUIC streams and dispatches them to typed channels:

| Lane            | Traffic                             | Consumer                      |
| --------------- | ----------------------------------- | ----------------------------- |
| `rpc_requests`  | Incoming RPC calls                  | Dispatcher serve loop         |
| `rpc_responses` | RPC replies (matched by request ID) | `rpc()` caller via oneshot    |
| `control`       | Barrier, heartbeat, join/leave      | Barrier logic, health monitor |
| `data`          | Point-to-point `send`/`recv`        | Application code              |
| `raw`           | Bulk byte streams                   | Tensor transfers              |

Lanes are independent. A full `data` channel doesn't block `control` messages.

## Stream protocol

Every QUIC unidirectional stream starts with a 1-byte tag:

- `0x01` — framed message (8-byte LE length prefix + serialized `NexarMessage`)
- `0x02` — raw bytes (8-byte LE length prefix + payload)

Messages are serialized with rkyv (zero-copy deserialization). Maximum message size is 4 GiB.

## When to use nexar

**Use nexar when:**

- You need collectives (allreduce, broadcast) across machines
- You want async, non-blocking communication in Rust
- You don't want to deal with MPI installation, `mpirun`, or C FFI
- You're building distributed ML training or inference

**Don't use nexar when:**

- Your GPUs are on the same machine — use NCCL directly (NVLink is 10-100x faster than any network)
- You need RDMA / GPUDirect — nexar uses standard UDP/QUIC
- You're already happy with MPI

## Building

```bash
cargo build --release
cargo test
cargo clippy --all-targets
```

Requires Rust 1.85+.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
