# nexar-rdma

RDMA and GPUDirect transport extensions for [nexar](https://crates.io/crates/nexar).

Provides InfiniBand/RoCE kernel-bypass transport as a drop-in extension to nexar's QUIC-based communication. After bootstrapping a nexar cluster, establish an RDMA mesh over the existing connections for zero-copy, OS-bypass data transfers.

## Features

| Feature      | What it enables                                   | Extra dependencies          |
| ------------ | ------------------------------------------------- | --------------------------- |
| *(default)*  | RDMA transport via ibverbs (InfiniBand/RoCE)      | `libibverbs` (rdma-core)    |
| `gpudirect`  | GPU memory ↔ NIC directly, `CudaAdapter`          | `libibverbs` + CUDA runtime |

## Usage

```toml
[dependencies]
nexar-rdma = "<use-latest-version>"

# For GPUDirect:
# nexar-rdma = { version = "<use-latest-version>", features = ["gpudirect"] }
```

```rust
use nexar_rdma::{bootstrap::establish_rdma_mesh, PeerConnectionRdmaExt};

// After nexar bootstrap, each rank layers RDMA on top:
establish_rdma_mesh(&client).await?;
```

With the `gpudirect` feature, `CudaAdapter` implements nexar's `DeviceAdapter` trait for GPU memory, and `NexarClientRdmaExt` adds RDMA-accelerated collective operations.

## Shutdown

Every rank calls `NexarClient::shutdown` before exit, after its last RDMA and QUIC transfer. Shutdown drains QUIC streams and TCP sidecars only, so await every RDMA send and receive first. The RDMA state lives in the client's peer connections and drops with them.

`shutdown` takes the client by value, so it needs sole ownership. A client shared as `Arc<NexarClient>` (for example by a `nexar-nccl` `HierarchicalComm`) needs every other holder dropped first:

```rust
drop(comm); // every holder of an Arc clone
let client = std::sync::Arc::into_inner(client)
    .expect("another Arc<NexarClient> clone is still alive");
client.shutdown().await?;
```

## Building

```bash
# RDMA (requires libibverbs / rdma-core)
cargo build -p nexar-rdma --release

# GPUDirect RDMA (requires libibverbs + CUDA)
cargo build -p nexar-rdma --release --features gpudirect
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
