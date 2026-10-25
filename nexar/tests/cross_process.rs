//! Cross-process bootstrap: each rank runs in its own OS process.
//!
//! The test binary re-runs itself once per rank. `NEXAR_XPROC_ROLE` selects
//! the role in a child. The seed child prints its bound address on stdout,
//! and the parent passes it to the worker children. Each child forms the
//! cluster, runs collectives, and exits non-zero on any mismatch.

use nexar::{ClusterConfig, CpuAdapter, DataType, NexarClient, NexarConfig, ReduceOp, SeedNode};
use std::net::SocketAddr;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::Instant;

const TEST_NAME: &str = "cross_process_cluster";
const ROLE_ENV: &str = "NEXAR_XPROC_ROLE";
const SEED_ENV: &str = "NEXAR_XPROC_SEED";
const SEED_MARKER: &str = "NEXAR_XPROC_SEED_ADDR=";
const WORLD: u32 = 3;
const CHILD_TIMEOUT: Duration = Duration::from_secs(90);

fn config() -> ClusterConfig {
    ClusterConfig::new(WORLD)
        .with_bind_addr("127.0.0.1:0".parse().unwrap())
        .with_cluster_token("cross-process-test-token")
        .with_join_timeout(Duration::from_secs(60))
        .with_nexar(NexarConfig::default())
}

/// Check collectives across processes. Panics, and so exits non-zero, on a mismatch.
async fn exercise(client: NexarClient, parent_pid: u32) {
    let client = Arc::new(client);
    let (rank, n) = (client.rank(), client.world_size());
    assert_eq!(n, WORLD);

    let mut data = vec![(rank + 1) as f32; 4096];
    unsafe {
        client
            .all_reduce(
                data.as_mut_ptr() as u64,
                data.len(),
                DataType::F32,
                ReduceOp::Sum,
            )
            .await
            .unwrap();
    }
    let expected = (n * (n + 1) / 2) as f32;
    assert!(
        data.iter().all(|&v| v == expected),
        "rank {rank}: allreduce {}",
        data[0]
    );

    // Every rank must run in its own process: gather the process ids.
    let pid = [std::process::id() as u64];
    let mut pids = vec![0u64; n as usize];
    unsafe {
        client
            .all_gather(
                pid.as_ptr() as u64,
                pids.as_mut_ptr() as u64,
                1,
                DataType::U64,
            )
            .await
            .unwrap();
    }
    let mut distinct = pids.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), n as usize, "rank {rank}: pids {pids:?}");
    assert!(
        !pids.contains(&(parent_pid as u64)),
        "a rank ran in the parent"
    );

    let (next, prev) = ((rank + 1) % n, (rank + n - 1) % n);
    let payload = [rank * 10 + 7; 32];
    let mut got = [0u32; 32];
    let sender = Arc::clone(&client);
    let send_ptr = payload.as_ptr() as u64;
    let send = tokio::spawn(async move { unsafe { sender.send(send_ptr, 128, next, 42).await } });
    unsafe {
        client
            .recv(got.as_mut_ptr() as u64, 128, prev, 42)
            .await
            .unwrap()
    };
    send.await.unwrap().unwrap();
    assert_eq!(got, [prev * 10 + 7; 32], "rank {rank}: tagged recv");

    client.barrier().await.unwrap();
    // Shut down before exit: closing discards data peers have not acknowledged.
    let client = Arc::into_inner(client).expect("no other client handle is left");
    client.shutdown().await.unwrap();
}

fn run_child(role: &str) {
    let parent_pid: u32 = std::env::var("NEXAR_XPROC_PARENT")
        .unwrap()
        .parse()
        .unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let cfg = config();
        let client = match role {
            "seed" => {
                let seed = SeedNode::from_config("127.0.0.1:0".parse().unwrap(), &cfg).unwrap();
                println!("{SEED_MARKER}{}", seed.local_addr());
                NexarClient::bootstrap_seed_with(seed, &cfg, Arc::new(CpuAdapter::new()))
                    .await
                    .unwrap()
            }
            "worker" => {
                let seed: SocketAddr = std::env::var(SEED_ENV).unwrap().parse().unwrap();
                NexarClient::bootstrap_join(seed, &cfg, Arc::new(CpuAdapter::new()))
                    .await
                    .unwrap()
            }
            other => panic!("unknown {ROLE_ENV}={other}"),
        };
        exercise(client, parent_pid).await;
    });
}

fn spawn_child(role: &str, seed: Option<SocketAddr>) -> Child {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
        .env(ROLE_ENV, role)
        .env("NEXAR_XPROC_PARENT", std::process::id().to_string())
        .kill_on_drop(true)
        .stdout(if seed.is_none() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        });
    if let Some(addr) = seed {
        cmd.env(SEED_ENV, addr.to_string());
    }
    cmd.spawn().unwrap()
}

/// Read the seed child's stdout until it prints its address. Keep draining it after.
async fn seed_addr(child: &mut Child) -> SocketAddr {
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let found = tokio::time::timeout(CHILD_TIMEOUT, async {
        while let Some(line) = lines.next_line().await.unwrap() {
            // libtest can print its own text before the marker on the same line.
            if let Some(at) = line.find(SEED_MARKER) {
                return line[at + SEED_MARKER.len()..].trim().to_string();
            }
            println!("[seed] {line}");
        }
        panic!("seed child exited without printing its address");
    })
    .await
    .expect("seed child printed no address in time");
    tokio::spawn(async move {
        while let Ok(Some(line)) = lines.next_line().await {
            println!("[seed] {line}");
        }
    });
    found.parse().unwrap()
}

/// Wait for `child` until `deadline`, killing it on timeout. Returns whether it succeeded.
async fn wait_until(child: &mut Child, deadline: Instant) -> bool {
    match tokio::time::timeout_at(deadline, child.wait()).await {
        Ok(status) => status.unwrap().success(),
        Err(_) => {
            let _ = child.kill().await;
            false
        }
    }
}

#[test]
fn cross_process_cluster() {
    if let Ok(role) = std::env::var(ROLE_ENV) {
        run_child(&role);
        return;
    }
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut seed = spawn_child("seed", None);
        let addr = seed_addr(&mut seed).await;
        let mut workers: Vec<Child> = (1..WORLD)
            .map(|_| spawn_child("worker", Some(addr)))
            .collect();

        let deadline = Instant::now() + CHILD_TIMEOUT;
        let seed_ok = wait_until(&mut seed, deadline).await;
        let mut workers_ok = Vec::new();
        for w in &mut workers {
            workers_ok.push(wait_until(w, deadline).await);
        }
        assert!(seed_ok, "seed process failed");
        assert!(
            workers_ok.iter().all(|&ok| ok),
            "worker processes: {workers_ok:?}"
        );
    });
}
