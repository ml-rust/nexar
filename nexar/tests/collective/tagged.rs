use super::helpers::run_collective;
use std::sync::Arc;

/// Concurrent tagged sends/recvs with different tags — verify no cross-talk.
#[tokio::test]
async fn test_tagged_send_recv_no_crosstalk() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        if rank == 0 {
            // Send two messages with different tags concurrently.
            let data_a = vec![1u8, 2, 3, 4];
            let data_b = vec![10u8, 20, 30, 40];
            let (ra, rb) = tokio::join!(
                client.send_bytes_tagged(1, 100, &data_a),
                client.send_bytes_tagged(1, 200, &data_b),
            );
            ra.unwrap();
            rb.unwrap();
        } else {
            // Receive on both tags — order should match tags, not arrival.
            let (ra, rb) = tokio::join!(
                client.recv_bytes_tagged(0, 100),
                client.recv_bytes_tagged(0, 200),
            );
            let buf_a = ra.unwrap();
            let buf_b = rb.unwrap();
            assert_eq!(&*buf_a, &[1u8, 2, 3, 4]);
            assert_eq!(&*buf_b, &[10u8, 20, 30, 40]);
        }
    })
    .await;
}

/// Tagged send/recv with 4 nodes: each sends to next with unique tags.
#[tokio::test]
async fn test_tagged_ring_4_nodes() {
    run_collective(4, |client| async move {
        let rank = client.rank();
        let world = client.world_size();
        let next = (rank + 1) % world;
        let prev = (rank + world - 1) % world;
        let tag = 500 + rank as u64;
        let prev_tag = 500 + prev as u64;

        let send_data = vec![rank as u8; 8];
        let (send_r, recv_r) = tokio::join!(
            client.send_bytes_tagged(next, tag, &send_data),
            client.recv_bytes_tagged(prev, prev_tag),
        );
        send_r.unwrap();
        let received = recv_r.unwrap();
        assert_eq!(received.len(), 8);
        assert!(received.iter().all(|&b| b == prev as u8));
    })
    .await;
}

/// Multiple concurrent tagged sends between the same pair of ranks.
#[tokio::test]
async fn test_tagged_concurrent_multiple_tags() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        if rank == 0 {
            let mut handles = Vec::new();
            for tag in 0..10u64 {
                let data = vec![tag as u8; 16];
                let client = Arc::clone(&client);
                handles.push(tokio::spawn(async move {
                    client.send_bytes_tagged(1, tag, &data).await.unwrap();
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        } else {
            let mut handles = Vec::new();
            for tag in 0..10u64 {
                let client = Arc::clone(&client);
                handles.push(tokio::spawn(async move {
                    let buf = client.recv_bytes_tagged(0, tag).await.unwrap();
                    assert_eq!(buf.len(), 16);
                    assert!(buf.iter().all(|&b| b == tag as u8));
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        }
    })
    .await;
}

/// Test that untagged sends don't interfere with tagged receives.
#[tokio::test]
async fn test_tagged_and_untagged_isolation() {
    run_collective(2, |client| async move {
        let rank = client.rank();
        if rank == 0 {
            // Send one untagged, one tagged.
            client.send_bytes(1, &[0xAA; 4]).await.unwrap();
            client.send_bytes_tagged(1, 42, &[0xBB; 4]).await.unwrap();
        } else {
            // Receive tagged first (should not pick up the untagged message).
            let (tagged_r, untagged_r) =
                tokio::join!(client.recv_bytes_tagged(0, 42), client.recv_bytes(0),);
            let tagged = tagged_r.unwrap();
            let untagged = untagged_r.unwrap();
            assert_eq!(&*tagged, &[0xBB; 4]);
            assert_eq!(&*untagged, &[0xAA; 4]);
        }
    })
    .await;
}
