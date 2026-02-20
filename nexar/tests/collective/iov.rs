use super::helpers::run_collective;
use nexar::IoVec;

/// Send 3 non-contiguous regions, recv into 3 regions, verify integrity.
#[tokio::test]
async fn test_send_recv_iov_3_regions() {
    run_collective(2, |client| async move {
        let rank = client.rank();

        if rank == 0 {
            // 3 separate buffers (non-contiguous).
            let buf_a = [1u8, 2, 3, 4];
            let buf_b = [10u8, 20];
            let buf_c = [100u8, 200, 255];

            let regions = vec![
                IoVec {
                    ptr: buf_a.as_ptr() as u64,
                    len: buf_a.len(),
                },
                IoVec {
                    ptr: buf_b.as_ptr() as u64,
                    len: buf_b.len(),
                },
                IoVec {
                    ptr: buf_c.as_ptr() as u64,
                    len: buf_c.len(),
                },
            ];

            unsafe { client.send_iov(&regions, 1, 0).await.unwrap() };
        } else {
            // Receive into 3 separate regions.
            let mut recv_a = vec![0u8; 4];
            let mut recv_b = vec![0u8; 2];
            let mut recv_c = vec![0u8; 3];

            let regions = vec![
                IoVec {
                    ptr: recv_a.as_mut_ptr() as u64,
                    len: recv_a.len(),
                },
                IoVec {
                    ptr: recv_b.as_mut_ptr() as u64,
                    len: recv_b.len(),
                },
                IoVec {
                    ptr: recv_c.as_mut_ptr() as u64,
                    len: recv_c.len(),
                },
            ];

            unsafe { client.recv_iov(&regions, 0, 0).await.unwrap() };

            assert_eq!(recv_a, vec![1, 2, 3, 4]);
            assert_eq!(recv_b, vec![10, 20]);
            assert_eq!(recv_c, vec![100, 200, 255]);
        }
    })
    .await;
}

/// Single region IoVec (degenerate case — should still work).
#[tokio::test]
async fn test_iov_single_region() {
    run_collective(2, |client| async move {
        let rank = client.rank();

        if rank == 0 {
            let buf = [42u8; 16];
            let regions = vec![IoVec {
                ptr: buf.as_ptr() as u64,
                len: buf.len(),
            }];
            unsafe { client.send_iov(&regions, 1, 1).await.unwrap() };
        } else {
            let mut recv = vec![0u8; 16];
            let regions = vec![IoVec {
                ptr: recv.as_mut_ptr() as u64,
                len: recv.len(),
            }];
            unsafe { client.recv_iov(&regions, 0, 1).await.unwrap() };
            assert!(recv.iter().all(|&b| b == 42));
        }
    })
    .await;
}

/// IoVec with mixed sizes: 1-byte, 1024-byte, and 8-byte regions.
#[tokio::test]
async fn test_iov_mixed_sizes() {
    run_collective(2, |client| async move {
        let rank = client.rank();

        if rank == 0 {
            let tiny = [0xFFu8];
            let big = vec![0xABu8; 1024];
            let medium = [0xCDu8; 8];

            let regions = vec![
                IoVec {
                    ptr: tiny.as_ptr() as u64,
                    len: tiny.len(),
                },
                IoVec {
                    ptr: big.as_ptr() as u64,
                    len: big.len(),
                },
                IoVec {
                    ptr: medium.as_ptr() as u64,
                    len: medium.len(),
                },
            ];

            unsafe { client.send_iov(&regions, 1, 2).await.unwrap() };
        } else {
            let mut r_tiny = vec![0u8; 1];
            let mut r_big = vec![0u8; 1024];
            let mut r_medium = vec![0u8; 8];

            let regions = vec![
                IoVec {
                    ptr: r_tiny.as_mut_ptr() as u64,
                    len: r_tiny.len(),
                },
                IoVec {
                    ptr: r_big.as_mut_ptr() as u64,
                    len: r_big.len(),
                },
                IoVec {
                    ptr: r_medium.as_mut_ptr() as u64,
                    len: r_medium.len(),
                },
            ];

            unsafe { client.recv_iov(&regions, 0, 2).await.unwrap() };

            assert_eq!(r_tiny, vec![0xFF]);
            assert!(r_big.iter().all(|&b| b == 0xAB));
            assert!(r_medium.iter().all(|&b| b == 0xCD));
        }
    })
    .await;
}
