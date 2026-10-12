# Contributing to nexar

Thank you for your interest in nexar.

## Before You Start

Open an issue before you start anything non-trivial, such as a new collective, transport, or public API.
Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

## Build and Test

Run these commands from the repository root. CI runs the same checks.

```bash
cargo fmt --all --check
cargo clippy -p nexar --all-targets -- -D warnings
cargo clippy -p nexar-rdma --all-targets -- -D warnings
cargo check -p nexar-nccl
cargo doc -p nexar --no-deps
cargo test -p nexar
cargo test -p nexar --doc
```

Hardware-dependent crates need extra setup:

- `nexar-rdma` tests need InfiniBand or RoCE hardware and `libibverbs`.
- `nexar-nccl` CUDA tests need an NVIDIA GPU and NCCL. Set `NEXAR_NCCL_TEST=1` to run them.

## Conventions

- `mod.rs` files contain only `pub mod` and `pub use`.
- Put one concern in one file, such as `allreduce.rs` and `broadcast.rs`.
- Keep files under 500 lines. Split a file at 300 lines when a logical boundary exists.
- Put collective algorithms in `collective/` with no transport details.
- Put QUIC connection and stream code in `transport/`.
- Do not use `.unwrap()` in library code.
- Give every new public API a doc comment.
- Place unit tests in the same file as the code under `#[cfg(test)]`.
- Place integration tests in `nexar/tests/` and test the public API only.

## Pull Requests

- Include tests for new behavior and for each bug fix.
- Run the checks in [Build and Test](#build-and-test) before you open the pull request.
- Describe the change, the reason for it, and the commands you ran.
