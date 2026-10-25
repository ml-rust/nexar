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
use std::sync::Arc;

// All ranks call collectively after nexar bootstrap:
// `nexar_client` is an `Arc<NexarClient>`.
let comm = unsafe { form_hierarchical_comm(Arc::clone(&nexar_client), cuda_stream).await? };

// Then use like any communicator:
unsafe { comm.allreduce(ptr, count, dtype, op).await? };
```

## Shutdown

Every rank calls `NexarClient::shutdown` before exit. It takes the client by value, so it needs sole ownership. `HierarchicalComm` holds an `Arc<NexarClient>` and a split inter-node client, so drop the communicator first:

```rust
drop(comm); // releases its Arc clone and the inter-node split client
let client = std::sync::Arc::into_inner(nexar_client)
    .expect("another Arc<NexarClient> clone is still alive");
client.shutdown().await?;
```

`Arc::into_inner` returns `None` while any other clone is alive. Drop every clone before the call.

## Building

Requires CUDA runtime and NCCL.

```bash
cargo build -p nexar-nccl --release
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
