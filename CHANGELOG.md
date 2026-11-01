# Changelog

## [Unreleased]

## [0.2.0] - 2026-11-01

### Breaking

- Wire protocol version 2 (`PROTOCOL_VERSION = 2`). A v1 node cannot join a v2 seed, and the seed rejects it.
- `NexarMessage::Hello` has a new `requested_rank: Option<Rank>` field. `listen_addr` now holds an IP address, not a socket address.
- `NexarMessage::Welcome` has a new `observed_addr` field. During formation its `peers` list is empty: the `PeerTable` message carries the peer addresses.
- New messages: `MeshListen`, `PeerTable`, `Abort`, `PeerHello`, `PeerHelloAck`. A `match` on `NexarMessage` needs arms for them.
- `NexarError` gains `FrameTooLarge`, `RankConflict`, `JoinTimeout`, `ShutdownIncomplete`, `BootstrapAborted`, and `TokenRequired`. An exhaustive `match` on `NexarError` stops compiling.
- `NexarError`, `AbortReason`, `NexarMessage`, `LaneKey`, and `ControlKey` are `#[non_exhaustive]`. A `match` outside nexar needs a wildcard arm. New variants no longer break such a `match`.
- `NexarConfig` gains the public fields `max_frame_bytes`, `recv_queue_depth`, and `recv_inflight_bytes`. A struct literal that lists every field stops compiling: end it with `..NexarConfig::default()`.
- `tcp_bulk_connect` and `tcp_bulk_connect_tls` take a `max_frame_bytes` argument.
- Every receive names the source rank first, as `recv(ptr, size, src, tag)` does: `PeerRouter::recv_lane(rank, LaneKey)`, `PeerRouter::recv_control(rank, ControlKey)`, `RelayDeliveries::recv_control(src, ControlKey)`, and `RelayDeliveries::recv_data(src, tag)`.
- `SeedNode::bind_local` and `SeedNode::bind_with_token` return `TokenRequired` for a non-loopback address without a token. Before, `bind_local("0.0.0.0:7000")` let any host join and get a leaf certificate. An empty token counts as no token.
- `SeedNode::bind` no longer logs its generated token. Read it with `SeedNode::cluster_token`.
- `send_or_relay_message` and `send_or_relay_tagged` take a `RelayRoute` in place of the rank, peers, routing table, strategy, and world size. `start_relay_listeners` takes a `RelayTopology`.
- `ElasticManager::new` takes a `NodeCredentials` in place of three `Vec<u8>` arguments.
- `SeedNode::form_cluster` now runs the whole formation: join, mesh address exchange, and ready barrier. A `WorkerNode::connect` caller that does not form the mesh stalls the seed until its timeout. Use `NexarClient::bootstrap_join`.
- `WorkerNode` has private fields and closes its seed connection on drop. Code outside nexar can no longer construct it.
- Bootstrap-time leaf certificates name `localhost`, the advertised IP, and the observed IP. Mesh connections verify the certificate against the dialed IP.
- Byte streams use a new wire format under protocol version 2: stream type `0x05` carries lane kind, `comm_id`, tag, a per-connection sequence number, and length. Type `0x06` is a skip notice. Types `0x02`, `0x03`, and `0x04` are gone, and point-to-point `send` no longer sends a framed `NexarMessage::Data` to a direct peer.
- `PeerRouter`: the public fields `rpc_requests`, `control`, `data`, `raw`, and `relay` are private. `recv_data`, `recv_raw`, `register_comm`, `register_tag`, and `remove_tag` are gone: use `recv_lane(rank, LaneKey)`. `recv_control` takes `(rank, ControlKey)`. `take_relay_rx` returns `None` once taken.
- Tagged sends of a split or rebuilt communicator use that communicator's own tag space, on QUIC and on the TCP sidecar. Before, they shared the root communicator's tags.

### Added

- Selective, ordered receive. A receive for `(src, tag)` takes only messages with that tag and never fails on another tag's message. Messages of one byte lane (`send`/`recv` tag, `send_bytes_tagged` tag, `send_bytes`, per communicator) from one sender arrive in send order. Control messages, RPC, and relayed traffic are selective and unordered.
- `NexarConfig::recv_queue_depth` (env `NEXAR_RECV_QUEUE_DEPTH`, default 256): messages each receive key queues per peer. A full key holds back only its own streams.
- `NexarConfig::recv_inflight_bytes` (env `NEXAR_RECV_INFLIGHT_BYTES`, default 4 GiB): payload bytes each peer's router reads at once. A message reserves its length before the router allocates its buffer. A message that does not fit waits unread in its QUIC stream. A message longer than the budget reads alone.
- `LaneKey`, `ControlKey`, `PeerRouter::spawn_with_limits`, `RouterLimits`, `PeerRouter::recv_lane`, `RelayDeliveries::recv_data`, `PeerConnection::send_lane`, and `PeerConnection::send_lane_best_effort`.
- `NexarClient::bootstrap_seed`, `bootstrap_seed_with`, and `bootstrap_join`: one process per rank, across hosts.
- `ClusterConfig`: world size, bind and advertise addresses, requested rank, cluster token, join timeout, and the client's `NexarConfig`.
- `SeedNode::from_config`, `SeedNode::bind_with_token`, `SeedNode::world_size`, and `SeedNode::cluster_token`.
- `NodeCredentials` and `WorkerNode::credentials`: the mTLS credentials a rank receives from the seed.
- `RelayRoute` and `RelayTopology`: the routing context of relay sends and relay listeners.
- `AbortReason` carries a formation failure to every rank.
- `tcp_bulk_accept_stream(stream, tls, max_frame_bytes)`: build a sidecar from an accepted `TcpStream`.
- `NexarConfig::max_frame_bytes` (env `NEXAR_MAX_FRAME_BYTES`, default 4 GiB, the QUIC router's message limit): the largest TCP bulk sidecar frame a rank accepts. A longer frame closes the sidecar, and receivers get the new `NexarError::FrameTooLarge`. Set the same value on every rank.
- `TcpBulkTransport::from_stream_with_limit` and `from_split_with_limit`.
- The `seed_worker` example runs one process per rank.
- `NexarClient::shutdown` and `shutdown_with_timeout`: flush sidecars, exchange a `DONE` byte (`0xD0`) with every direct peer, then close. Each rank waits for the peer's `DONE` byte or a deliberate close. Data sent before the call reaches every peer that is still running. A timeout returns the new `ShutdownIncomplete` error. A wrong byte, a stream finished before the byte, or a reset stream also returns it, with the peer in `bad_signal`. Call it on every rank before exit.
- `nexar-rdma` and `nexar-nccl` document shutdown: drop every holder of an `Arc<NexarClient>`, such as a `HierarchicalComm`, then call `Arc::into_inner` and `shutdown`.
- `BulkTransport::shutdown`, default no-op. `TcpBulkTransport` flushes, sends TLS `close_notify`, and sends FIN.

### Changed

- `bootstrap_local` and `bootstrap_elastic` run the same per-rank mesh path as `bootstrap_join`. Each rank dials and accepts on one UDP socket.
- `bootstrap_local` now honors `NEXAR_TOPOLOGY` from the environment.
- The seed accepts each connection in its own task. A garbage, stalled, or rejected connection no longer stops formation. A bad Hello gets an `Abort` reply.
- The seed honors requested ranks and assigns the rest in arrival order. Hellos that arrive after the cluster fills get an `Abort`.
- `WorkerNode::connect` retries until the join timeout and fails with `JoinTimeout` when no Welcome arrives.
- A non-loopback bind, seed, or advertise address without a cluster token is an error. The bootstrap connection does not verify the seed's certificate, so the token is the only defense.
- TCP bulk sidecars use mTLS with the cluster certificates. In sparse topologies, sidecars connect neighbors only. A sidecar config mismatch between two ranks fails formation.
- Dropping the last client that shares a mesh (root, split, and rebuilt clients share one) closes its connections at once. Before, router tasks kept dropped connections open.
- A worker closes its seed connection explicitly on every exit path, so the seed no longer waits for acknowledgements that never come on abort paths.
- Client endpoints bind the wildcard address of the seed's or peer's address family instead of loopback.

### Fixed

- The QUIC router allocated the length a peer announced, up to 4 GiB per message and 512 messages at once. It now refuses a message longer than `NexarConfig::max_frame_bytes` without allocating: it stops the stream, logs `FrameTooLarge`, and the sender's write fails. It allocates a payload only after it reserves the bytes in `recv_inflight_bytes`.
- The router accepted a sequence number of any distance past the lowest unknown one, so a hostile peer could grow its reorder state without bound. It now refuses a number 65536 or more past it.
- `recv`, `recv_iov`, barriers, recovery, and elastic checkpoints took the next message from a peer and failed on a tag or kind mismatch, losing that message. QUIC delivers each message on its own stream, so two sends could also arrive swapped. Receives now wait for their own key, in send order where the lane is ordered.
- The router held its tag-map lock while it waited for room in a full tag channel, so one slow tag stalled every tag from that peer. A full key now waits on its own stream only.
- A send cancelled by a timeout no longer leaves a message that blocks its lane: the sender sends a skip notice.
- A receiver blocked on a peer that disconnects gets `PeerDisconnected` after the queued messages, on every lane.
- Split and rebuilt communicators failed tagged receives and barriers with `UnknownPeer` when the TCP sidecar was off: they had no routers. They now share the parent's routers.
- A grown communicator in a sparse topology replaced the relay deliveries the parent's relay listeners still fill.
- The router queued received heartbeats on the control lane, so a barrier that ran after a heartbeat failed with an unexpected message. Received heartbeats are now dropped.
- `nexar-rdma` builds with libclang 22 and newer. It no longer depends on `ibverbs-sys`. Its build script generates the libibverbs bindings from the system headers with bindgen 0.72. bindgen 0.71, which `ibverbs-sys` 0.3.2 requires, emitted opaque `ibv_mr`, `ibv_qp`, `ibv_cq` and other structs under libclang 22+, so the layout assertions did not compile. The build no longer needs cmake.
- `nexar-rdma` with the `gpudirect` feature compiles again. The blocking GPUDirect send and receive paths now take the queue-pair lock with `blocking_lock`.
- `SeedNode::bind` without `NEXAR_CLUSTER_TOKEN` logged its generated token at warn level, so the secret reached every log sink. The seed no longer logs it.
- `TcpBulkTransport` lost a frame when it arrived while a receiver for its tag registered, and two concurrent receivers for one tag could orphan one of them.
- Sparse topologies with the TCP bulk sidecar enabled failed to bootstrap.
- A bootstrap control frame announcing a huge payload no longer allocates that payload.
- The TCP bulk receiver allocated a frame's buffer from the announced length, so one frame header could force a 4 GiB allocation. It now reads in 4 MiB steps and grows the buffer as data arrives.
- A receiver on a closed TCP bulk sidecar waited forever on a tagged channel. It now gets an error: `FrameTooLarge` after an oversize frame, `Transport` otherwise.
- A flood of connections to the seed started one handshake task per connection. At most 64 handshakes run at once, and the seed refuses the rest. A refused worker retries.
- A bootstrap control reader queued messages without limit. Its queue now holds 64 messages, and the reader waits for the consumer.
