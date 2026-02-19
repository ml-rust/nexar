use nexar::{CpuAdapter, NexarClient};
use std::sync::Arc;

/// Helper: run a collective operation across N clients concurrently.
/// Keeps all clients alive until every task completes.
pub async fn run_collective<F, Fut>(world_size: u32, f: F)
where
    F: Fn(Arc<NexarClient>) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let adapter = Arc::new(CpuAdapter::new());
    let clients = NexarClient::bootstrap_local(world_size, adapter)
        .await
        .unwrap();
    let clients: Vec<Arc<NexarClient>> = clients.into_iter().map(Arc::new).collect();

    let f = Arc::new(f);
    let mut handles = Vec::new();
    for c in &clients {
        let c = Arc::clone(c);
        let f = Arc::clone(&f);
        handles.push(tokio::spawn(async move { f(c).await }));
    }
    for h in handles {
        h.await.unwrap();
    }
    // `clients` dropped here — all tasks already complete.
}
