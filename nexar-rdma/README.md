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

// After nexar bootstrap, layer RDMA on top:
establish_rdma_mesh(&clients).await;
```

With the `gpudirect` feature, `CudaAdapter` implements nexar's `DeviceAdapter` trait for GPU memory, and `NexarClientRdmaExt` adds RDMA-accelerated collective operations.

## Building

```bash
# RDMA (requires libibverbs / rdma-core)
cargo build -p nexar-rdma --release

# GPUDirect RDMA (requires libibverbs + CUDA)
cargo build -p nexar-rdma --release --features gpudirect
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
