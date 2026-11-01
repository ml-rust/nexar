# Changelog

## [Unreleased]

## [0.2.0] - 2026-11-01

Selective ordered receives, one-process-per-rank bootstrap, a clean shutdown, and bounds on every peer-announced length.

### Breaking

- **Wire protocol** — `PROTOCOL_VERSION = 2`. A v1 node cannot join a v2 seed. Byte streams use stream types `0x05` (data) and `0x06` (skip notice). Types `0x02`, `0x03`, and `0x04` are gone.
- **`NexarMessage`** — `Hello` gains `requested_rank: Option<Rank>`, and `listen_addr` holds an IP address. `Welcome` gains `observed_addr` and has an empty `peers` list during formation. New messages: `MeshListen`, `PeerTable`, `Abort`, `PeerHello`, `PeerHelloAck`.
- **`#[non_exhaustive]`** — `NexarError`, `AbortReason`, `NexarMessage`, `LaneKey`, and `ControlKey`. A `match` outside nexar needs a wildcard arm. `NexarError` gains `FrameTooLarge`, `RankConflict`, `JoinTimeout`, `ShutdownIncomplete`, `BootstrapAborted`, and `TokenRequired`.
- **`NexarConfig`** — new public fields `max_frame_bytes`, `recv_queue_depth`, and `recv_inflight_bytes`. End a struct literal with `..NexarConfig::default()`.
- **Receives** — every receive names the source rank first: `PeerRouter::recv_lane(rank, LaneKey)`, `PeerRouter::recv_control(rank, ControlKey)`, `RelayDeliveries::recv_control(src, ControlKey)`, `RelayDeliveries::recv_data(src, tag)`.
- **`PeerRouter`** — its fields are private. `recv_data`, `recv_raw`, `register_comm`, `register_tag`, and `remove_tag` are gone. Use `recv_lane`. `take_relay_rx` returns `None` once taken.
- **Signatures** — `tcp_bulk_connect` and `tcp_bulk_connect_tls` take `max_frame_bytes`. `send_or_relay_message` and `send_or_relay_tagged` take a `RelayRoute`. `start_relay_listeners` takes a `RelayTopology`. `ElasticManager::new` takes a `NodeCredentials`.
- **Seed token** — `SeedNode::bind_local` and `SeedNode::bind_with_token` return `TokenRequired` for a non-loopback address without a token. Before, `bind_local("0.0.0.0:7000")` let any host join. An empty token counts as none.
- **`SeedNode::bind`** — no longer logs its generated token. Read it with `SeedNode::cluster_token`.
- **`SeedNode::form_cluster`** — runs the whole formation, including the mesh exchange and ready barrier. A `WorkerNode::connect` caller must use `NexarClient::bootstrap_join`, or the seed stalls until its timeout.
- **`WorkerNode`** — its fields are private. It closes its seed connection on drop.
- **Communicator tags** — a split or rebuilt communicator uses its own tag space on QUIC and on the TCP sidecar. Before, it shared the root communicator's tags.

### Added

- **Selective ordered receive** — a receive for `(src, tag)` takes only that tag. Messages on one byte lane from one sender arrive in send order. Control, RPC, and relayed traffic are selective and unordered. New: `LaneKey`, `ControlKey`, `RouterLimits`, `PeerRouter::spawn_with_limits`, `PeerConnection::send_lane`, `PeerConnection::send_lane_best_effort`.
- **`NexarConfig::recv_queue_depth`** — env `NEXAR_RECV_QUEUE_DEPTH`, default 256. Messages each receive key queues per peer.
- **`NexarConfig::recv_inflight_bytes`** — env `NEXAR_RECV_INFLIGHT_BYTES`, default 4 GiB. Payload bytes each peer's router reads at once.
- **`NexarConfig::max_frame_bytes`** — env `NEXAR_MAX_FRAME_BYTES`, default 4 GiB. The largest message or TCP sidecar frame a rank accepts. Set the same value on every rank. Also `TcpBulkTransport::from_stream_with_limit`, `from_split_with_limit`, and `tcp_bulk_accept_stream`.
- **Cross-host bootstrap** — `NexarClient::bootstrap_seed`, `bootstrap_seed_with`, and `bootstrap_join` run one process per rank. `ClusterConfig` holds the world size, addresses, requested rank, token, and join timeout. See the `seed_worker` example.
- **Seed and worker API** — `SeedNode::from_config`, `bind_with_token`, `world_size`, `cluster_token`. `NodeCredentials`, `WorkerNode::credentials`, `RelayRoute`, `RelayTopology`, and `AbortReason`, which carries a formation error to every rank.
- **`NexarClient::shutdown`** and **`shutdown_with_timeout`** — flush and exchange a done byte with every direct peer, then close. Call on every rank before exit. A timeout or bad signal returns `ShutdownIncomplete`. `BulkTransport::shutdown` is a default no-op.
- **`nexar-rdma` and `nexar-nccl` shutdown** — drop every holder of the `Arc<NexarClient>`, such as a `HierarchicalComm`, then call `Arc::into_inner` and `shutdown`.

### Fixed

- **Logged cluster token** — before, `SeedNode::bind` without `NEXAR_CLUSTER_TOKEN` logged its generated token at warn level.
- **Unbounded allocation** — before, the QUIC router and the TCP bulk receiver allocated the length a peer announced, up to 4 GiB per message. Both refuse a length past `max_frame_bytes` and reserve bytes in `recv_inflight_bytes` first. A bootstrap control frame no longer allocates its announced size.
- **Reorder state growth** — before, the router accepted any sequence number. It refuses one 65536 or more past the lowest unknown.
- **Connection flood** — before, the seed started a handshake task per connection and a bootstrap control reader queued without limit. At most 64 handshakes run, and a refused worker retries. The reader queue holds 64 messages.
- **Lost and swapped messages** — before, `recv`, barriers, recovery, and elastic checkpoints took the next message and lost it on a tag mismatch. QUIC could also swap two sends. Receives now wait for their own key.
- **Stalled tags** — before, one slow tag stalled every tag from that peer.
- **Cancelled send** — before, a send cancelled by a timeout blocked its lane. The sender now sends a skip notice.
- **Peer disconnect** — a blocked receiver gets `PeerDisconnected` after the queued messages, on every lane.
- **Split and grown communicators** — before, split and rebuilt communicators failed tagged receives and barriers with `UnknownPeer` when the TCP sidecar was off. A grown communicator in a sparse topology also replaced the parent's relay deliveries.
- **Heartbeats** — before, a received heartbeat failed the next barrier with an unexpected message. Received heartbeats are dropped.
- **TCP bulk sidecar** — before, a frame was lost when it arrived as a receiver registered, and a receiver on a closed sidecar waited forever. It now gets `FrameTooLarge` or `Transport`. Sparse topologies with the sidecar now bootstrap.
- **`nexar-rdma` build** — before, it did not build with libclang 22 and newer. The `gpudirect` feature also did not compile.

### Changed

- **Bootstrap** — `bootstrap_local` and `bootstrap_elastic` use the same per-rank mesh path as `bootstrap_join`. `bootstrap_local` honors `NEXAR_TOPOLOGY`.
- **Seed** — a garbage, stalled, or rejected connection no longer stops formation. A bad Hello gets an `Abort`. The seed honors requested ranks. `WorkerNode::connect` retries until the join timeout, then returns `JoinTimeout`.
- **Token policy** — a non-loopback bind, seed, or advertise address without a cluster token is an error.
- **Certificates** — bootstrap leaf certificates name `localhost`, the advertised IP, and the observed IP. Mesh connections check the certificate against the dialed IP.
- **TCP bulk sidecars** — they use mTLS with the cluster certificates. Sparse topologies connect neighbors only. A sidecar config mismatch between two ranks fails formation.
- **Dropping a client** — dropping the last client that shares a mesh closes its connections at once.
- **Client endpoints** — they bind the wildcard address of the peer's address family instead of loopback.

### Removed

- **`nexar-rdma` dependencies** — `ibverbs-sys` and the cmake build requirement. The build script generates libibverbs bindings with bindgen 0.72.
