# nexar-nccl

Hierarchical communicator combining NCCL for intra-node GPU-GPU with [nexar](https://crates.io/crates/nexar) for inter-node transport.

This is the standard 2D decomposition used by Megatron-LM and DeepSpeed: NCCL handles NVLink/NVSwitch traffic within a node, nexar handles QUIC/RDMA traffic between nodes.

## How it works

Each node elects a lead rank. Collectives run in two (or three) phases:

1. **Intra-node** — NCCL reduces across local GPUs (NVLink speed)
2. **Inter-node** — nexar reduces across lead ranks (network speed)
3. **Intra-node** — NCCL broadcasts the result back to local GPUs

Large messages use a reduce-scatter / allgather variant to minimize the data sent over the network.

## Collectives

- `allreduce` — hierarchical 2D allreduce (small-message and large-message paths)
- `broadcast` — NCCL intra + nexar inter + NCCL intra
- `allgather` — NCCL allgather + nexar allgather + NCCL broadcast
- `reduce` — NCCL reduce to lead + nexar reduce across leads
- `barrier` — CUDA stream sync + nexar barrier on leads

## Usage

```toml
[dependencies]
nexar-nccl = "<use-latest-version>"
```

```rust
use nexar_nccl::{form_hierarchical_comm, HierarchicalComm};

// All ranks call collectively after nexar bootstrap:
let comm = unsafe { form_hierarchical_comm(nexar_client, cuda_stream).await? };

// Then use like any communicator:
unsafe { comm.allreduce(ptr, count, dtype, op).await? };
```

## Building

Requires CUDA runtime and NCCL.

```bash
cargo build -p nexar-nccl --release
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
