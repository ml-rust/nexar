# Changelog

## [Unreleased]

### Breaking

- Wire protocol version 2 (`PROTOCOL_VERSION = 2`). A v1 node cannot join a v2 seed, and the seed rejects it.
- `NexarMessage::Hello` has a new `requested_rank: Option<Rank>` field. `listen_addr` now holds an IP address, not a socket address.
- `NexarMessage::Welcome` has a new `observed_addr` field. During formation its `peers` list is empty: the `PeerTable` message carries the peer addresses.
- New messages: `MeshListen`, `PeerTable`, `Abort`, `PeerHello`, `PeerHelloAck`. A `match` on `NexarMessage` needs arms for them. A `match` on `NexarError` needs an arm for `FrameTooLarge`.
- `SeedNode::form_cluster` now runs the whole formation: join, mesh address exchange, and ready barrier. A `WorkerNode::connect` caller that does not form the mesh stalls the seed until its timeout. Use `NexarClient::bootstrap_join`.
- `WorkerNode` has private fields and closes its seed connection on drop. Code outside nexar can no longer construct it.
- Bootstrap-time leaf certificates name `localhost`, the advertised IP, and the observed IP. Mesh connections verify the certificate against the dialed IP.

### Added

- `NexarClient::bootstrap_seed`, `bootstrap_seed_with`, and `bootstrap_join`: one process per rank, across hosts.
- `ClusterConfig`: world size, bind and advertise addresses, requested rank, cluster token, join timeout, and the client's `NexarConfig`.
- `SeedNode::from_config`, `SeedNode::bind_with_token`, and `SeedNode::world_size`.
- Errors `RankConflict`, `JoinTimeout`, `BootstrapAborted`, and `TokenRequired`. `AbortReason` carries a formation failure to every rank.
- `tcp_bulk_accept_stream`: build a sidecar from an accepted `TcpStream`.
- `NexarConfig::max_frame_bytes` (env `NEXAR_MAX_FRAME_BYTES`, default 4 GiB, the QUIC router's message limit): the largest TCP bulk sidecar frame a rank accepts. A longer frame closes the sidecar, and receivers get the new `NexarError::FrameTooLarge`. Set the same value on every rank.
- `TcpBulkTransport::from_stream_with_limit` and `from_split_with_limit`.
- The `seed_worker` example runs one process per rank.
- `NexarClient::shutdown` and `shutdown_with_timeout`: flush sidecars, exchange a done signal with every direct peer, then close. Data sent before the call reaches every peer that is still running. A timeout returns the new `ShutdownIncomplete` error. Call it on every rank before exit.
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
- `tcp_bulk_connect`, `tcp_bulk_connect_tls`, and `tcp_bulk_accept_stream` take a `max_frame_bytes` argument.
- Client endpoints bind the wildcard address of the seed's or peer's address family instead of loopback.

### Fixed

- The router queued received heartbeats on the control lane, so a barrier that ran after a heartbeat failed with an unexpected message. Received heartbeats are now dropped.

- `SeedNode::bind` without `NEXAR_CLUSTER_TOKEN` logged a hex token that workers could not use. The seed now checks the logged hex string.
- `TcpBulkTransport` lost a frame when it arrived while a receiver for its tag registered, and two concurrent receivers for one tag could orphan one of them.
- Sparse topologies with the TCP bulk sidecar enabled failed to bootstrap.
- A bootstrap control frame announcing a huge payload no longer allocates that payload.
- The TCP bulk receiver allocated a frame's buffer from the announced length, so one frame header could force a 4 GiB allocation. It now reads in 4 MiB steps and grows the buffer as data arrives.
- A receiver on a closed TCP bulk sidecar waited forever on a tagged channel. It now gets an error: `FrameTooLarge` after an oversize frame, `Transport` otherwise.
- A flood of connections to the seed started one handshake task per connection. At most 64 handshakes run at once, and the seed refuses the rest. A refused worker retries.
- A bootstrap control reader queued messages without limit. Its queue now holds 64 messages, and the reader waits for the consumer.
