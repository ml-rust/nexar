use super::helpers::run_collective;

#[tokio::test]
async fn test_barrier_4_nodes() {
    run_collective(4, |client| async move {
        client.barrier().await.unwrap();
    })
    .await;
}

#[tokio::test]
async fn test_barrier_5_nodes_dissemination() {
    run_collective(5, |client| async move {
        client.barrier().await.unwrap();
    })
    .await;
}

#[tokio::test]
async fn test_barrier_2_nodes_double() {
    run_collective(2, |client| async move {
        client.barrier().await.unwrap();
        client.barrier().await.unwrap();
    })
    .await;
}
