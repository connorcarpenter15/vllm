# OpenEngine v1 gRPC service (vLLM side)

This module is the vLLM-side implementation of the vendor-neutral **OpenEngine
v1** gRPC contract (`openengine/proto/openengine.proto`, vendored at
`vllm/rust/proto/openengine.proto`). It is the wire API consumed by the Dynamo
**Rust** vLLM sidecar (`dynamo/lib/vllm-sidecar/`).

It is a **sibling** of the `vllm`-native Generate service in `../mod.rs` and is
backed by the same [`crate::state::AppState`] / [`vllm_text::TextLlm`] facade.
The Generate service (`../{mod,convert,tests}.rs`) is the line-by-line template
for this module — keep them structurally parallel.

## Files

| File | Role |
|---|---|
| `mod.rs` | `OpenEngineServiceImpl` — implements the generated `open_engine_server::OpenEngine` trait. `pb` = `tonic::include_proto!("openengine.v1")`. |
| `convert.rs` | OpenEngine `GenerateRequest` ↔ `vllm_text::TextRequest`, and `DecodedTextEvent` → `GenerateResponse` oneof. |
| `tests.rs` | Wire-shape tests against the `mock-engine` double (mirror of `../tests.rs`). |
| `CLAUDE.md` | This file. |

## Discovery comes from the engine handshake, not from flags

The defining constraint of the sidecar architecture: the out-of-process
frontend learns **everything** about the engine over the wire. None of it is a
CLI flag. All discovery metadata is sourced from
[`EngineCoreClient::ready_response`] (the engine startup handshake,
`engine-core-client/src/protocol/handshake.rs`), which the Python EngineCore
populates from its `vllm_config`:

- **Role** — derived from the engine's authoritative `kv_role`
  (`kv_producer` → prefill, `kv_consumer` → decode, else aggregated). Never a
  flag. See `convert::role_from_kv_role`.
- **Parallelism** — `tensor_parallel_size`, `pipeline_parallel_size`,
  `data_parallel_size`, `data_parallel_rank`.
- **Caps** — `block_size`, `max_num_seqs`, `max_num_batched_tokens`,
  `max_model_len` (+ `num_gpu_blocks` summed across engines).
- **KV connector / events** — `kv_connector`, `kv_engine_id`,
  `kv_events_publisher`/`endpoint`/`topic`.

The **only** new vLLM CLI flags are `--openengine-port` / `--openengine-host`
(transport binding, wired in `cmd/src/cli.rs` + `server/src/{config,lib}.rs`).

## RPC mapping

| RPC | Maps to |
|---|---|
| `Generate` (server-stream) | `TextLlm::generate`. Per `TextDelta` → `Token{token_ids,text}`; terminal → `Finished{reason,usage}`, or `PrefillReady{kv_session}` for the prefill role. Mid-stream failure → `EngineError`. `stream=false` collapses to the terminal output only (`intermediate` flag). `media` (if present) → `MediaContentPart`s via `media_parts_from_request`, then `ChatLlm::prepare_media` fetches/preprocesses and expands the placeholder markers in `token_ids` (so multimodal requests must use token-ids input); runs before `mark_prefill_request` so prefill encodes the media. |
| `GetEngineInfo` | `engine_name="vllm"`, version + role + parallelism + kv_connector from the handshake. |
| `GetModelInfo` | model id / served names + caps from the handshake; most capability bools are static (text/token-ids/logprobs/guided = true; lora = false). `supports_multimodal` reflects the loaded backend (`ChatLlm::supports_multimodal()`: true when a `MultimodalModelInfo` resolved, e.g. a VLM). |
| `GetLoad` | `running_requests` from `AppState::server_load`; deeper scheduler stats (queue depth, KV usage) are **not yet a queryable snapshot** → reported as 0 (follow-up). |
| `Health` | liveness from `EngineCoreClient::is_healthy`; `include_inference_probe` runs a bounded 1-token greedy generate. |
| `Abort` | `EngineCoreClient::abort([id])`, idempotent (unknown id → `ABORTED`). `abort_all` → `UNSUPPORTED` (frontend does not retain the in-flight id set). |
| `Drain` (server-stream) | polls `server_load` to 0 or deadline; `STARTED` → `IN_PROGRESS`* → `COMPLETE`/`FAILED`. |
| `GetKvConnectorInfo` | from the handshake `kv_connector`. |
| `GetKvEventSources` | surfaces the ZMQ publisher (`kv_events_*`) for KV-aware routing; populates the routable `endpoint_addr` (`KvEndpoint{host,port,protocol}`) — a bind wildcard (`*`/`0.0.0.0`/`::`) is rewritten to this node's advertised IP so a remote router can connect. Empty when no publisher. |
| `SubscribeKvEvents` / `SubscribeRuntimeEvents` | `UNIMPLEMENTED` in v1 (consumers subscribe to the ZMQ source directly via `GetKvEventSources`). |

## Disaggregation (KV session) — Phase 3

- **Prefill role**: the terminal `Finished.kv_transfer_params` (the connector's
  handoff metadata) is packed into `PrefillReady.kv_session.attributes_struct`
  as a typed `google.protobuf.Struct` (`convert::kv_transfer_params_to_kv_session`
  → `json_to_prost_struct`).
- **Decode role**: the request's `kv_session.attributes_struct` is converted
  back to JSON (`convert::prost_struct_to_json`) and lifted into the
  engine-core request's `kv_transfer_params` via `vllm_xargs`.

The typed Struct preserves value types end-to-end (ints, bools, arrays) with no
stringify/reparse — `number_to_json` recovers integral `f64`s back to JSON
integers so connector params like `remote_port`/`tp_size` keep their int type.
The exact session contents track the NixlConnector path; the shape here is the
stable envelope.

## Testing

`tests.rs` stands up the tonic service over the `mock-engine` double (the
`../tests.rs` scaffold: `IpcNamespace` + `spawn_mock_engine_task` +
`FakeTextBackend`) and asserts the wire shape for the happy-path stream, the
discovery RPCs, abort, drain, and the unimplemented subscriptions. Real-engine
coverage is the cluster smoke matrix (plan Phase 5) — there is no local GPU.
