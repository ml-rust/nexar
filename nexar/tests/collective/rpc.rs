use nexar::rpc::RpcHandler;
use nexar::{CpuAdapter, NexarClient};
use std::sync::Arc;

#[tokio::test]
async fn test_rpc_call_and_response() {
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(2, adapter).await.unwrap();
    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();

    let handler: RpcHandler = Arc::new(|args: &[u8]| {
        let mut result: Vec<u8> = args.iter().rev().copied().collect();
        result.push(0xFF);
        result
    });
    clients[1].register_rpc(42, handler).await;

    let c0 = Arc::clone(&clients[0]);
    let c1 = Arc::clone(&clients[1]);

    let caller = tokio::spawn(async move { c0.rpc(1, 42, &[1, 2, 3]).await.unwrap() });

    let responder = tokio::spawn(async move {
        let dispatcher = c1.rpc_dispatcher();
        let peer = Arc::clone(c1.peer(0).unwrap());
        let msg = c1.recv_rpc_request(0).await.unwrap();
        match msg {
            nexar::NexarMessage::Rpc {
                req_id,
                fn_id,
                payload,
            } => {
                dispatcher
                    .handle_request(&peer, req_id, fn_id, &payload)
                    .await
                    .unwrap();
            }
            other => panic!("expected Rpc, got {other:?}"),
        }
    });

    let result = caller.await.unwrap();
    responder.await.unwrap();

    assert_eq!(result, vec![3, 2, 1, 0xFF]);
}
