//! Remote procedure calls between ranks.
//!
//! Rank 1 registers an "add" handler (fn_id=1). Rank 0 calls it remotely,
//! sending two f32s and receiving their sum.
//!
//! ```bash
//! cargo run --example rpc
//! ```

use nexar::{CpuAdapter, NexarClient};
use std::sync::Arc;

const FN_ADD: u16 = 1;

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

    // Rank 1: register a handler that adds two f32 values.
    c1.register_rpc(
        FN_ADD,
        Arc::new(|args: &[u8]| -> Vec<u8> {
            let a = f32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = f32::from_le_bytes(args[4..8].try_into().unwrap());
            let sum = a + b;
            sum.to_le_bytes().to_vec()
        }),
    )
    .await;

    // Rank 1: serve incoming RPC requests in the background.
    let server = {
        let c1 = Arc::clone(&c1);
        tokio::spawn(async move {
            let dispatcher = c1.rpc_dispatcher();
            // Serve one request from rank 0.
            let msg = c1.recv_rpc_request(0).await.unwrap();
            if let nexar::NexarMessage::Rpc {
                req_id,
                fn_id,
                payload,
            } = msg
            {
                dispatcher
                    .handle_request(c1.peer(0).unwrap(), req_id, fn_id, &payload)
                    .await
                    .unwrap();
            }
        })
    };

    // Rank 0: call the remote function on rank 1.
    let mut args = Vec::new();
    args.extend_from_slice(&3.0f32.to_le_bytes());
    args.extend_from_slice(&4.0f32.to_le_bytes());

    let result = c0.rpc(1, FN_ADD, &args).await?;
    let sum = f32::from_le_bytes(result[0..4].try_into().unwrap());
    println!("rank 0 called add(3.0, 4.0) on rank 1 => {sum}");
    // Output: rank 0 called add(3.0, 4.0) on rank 1 => 7

    server.await.unwrap();

    Ok(())
}
