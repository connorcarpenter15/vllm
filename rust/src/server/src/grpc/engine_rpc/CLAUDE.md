# Private engine RPC service

This module implements vLLM's private RPC API for out-of-process frontends.
The schema lives at `rust/proto/engine.proto`.

It is a **sibling** of the `vllm`-native Generate service in `../mod.rs` and is
backed by the same [`crate::state::AppState`] / [`vllm_text::TextLlm`] facade.
The Generate service (`../{mod,convert,tests}.rs`) is the line-by-line template
for this module — keep them structurally parallel.

## Files

| File | Role |
|---|---|
| `mod.rs` | `EngineServiceImpl` — implements the generated `engine_server::Engine` trait. `pb` = `tonic::include_proto!("vllm.engine.v1")`. |
| `convert.rs` | engine RPC `GenerateRequest` ↔ `vllm_text::TextRequest`, and `DecodedTextEvent` → `GenerateResponse` oneof. |
| `tests.rs`, `tests/` | Shared mock harness plus focused discovery, generation, lifecycle, LoRA, media, and topology tests. |
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
- **Caps** — `block_size`, per-engine `num_gpu_blocks`, `max_num_seqs`,
  `max_num_batched_tokens`, `max_model_len`, and `max_loras`.
- **KV connector / events** — `kv_connector`, `kv_engine_id`,
  `kv_events_publisher`/`endpoint`/`topic`.

The **only** new vLLM CLI flags are `--engine-rpc-port` / `--engine-rpc-host`
(transport binding, wired in `cmd/src/cli.rs` + `server/src/{config,lib}.rs`).

## RPC mapping

| RPC | Maps to |
|---|---|
| `Generate` (server-stream) | `TextLlm::generate`. Per `TextDelta` → `Token{token_ids,text}`; terminal → `Finished{reason,usage}`, or `PrefillReady{kv_session}` for the prefill role. Mid-stream failure → `EngineError`. `stream=false` collapses to the terminal output only (`intermediate` flag). `media` (if present) → `MediaContentPart`s via `media_parts_from_request`, then `ChatLlm::prepare_media` fetches/preprocesses and expands the placeholder markers in `token_ids` (so multimodal requests must use token-ids input); runs before `mark_prefill_request` so prefill encodes the media. |
| `GetEngineInfo` | `engine_name="vllm"`, version + role + parallelism + kv_connector from the handshake. |
| `GetModelInfo` | Model names, per-engine capacity, effective parser names, and LoRA/multimodal capability. |
| `Health` | liveness from `EngineCoreClient::is_healthy`; reports `DRAINING` after admission closes. `include_inference_probe` runs a bounded 1-token greedy generate. |
| `Abort` | `EngineCoreClient::abort([id])`, idempotent (unknown id → `ABORTED`). `abort_all` → `UNSUPPORTED` (frontend does not retain the in-flight id set). |
| `Drain` | irreversibly closes engine-RPC admission, then reports `IN_PROGRESS` or `COMPLETE` from `server_load`. Callers poll it; the RPC does not wait. |
| `LoadLora` / `UnloadLora` / `ListLoras` | Manage the engine-local adapter registry. Disabled engines return `FAILED_PRECONDITION`; `Generate.lora_name` selects a loaded adapter. |
| `GetKvConnectorInfo` | from the handshake `kv_connector`. |
| `GetKvEventSources` | surfaces the ZMQ publisher (`kv_events_*`) for KV-aware routing; populates the routable `endpoint_addr` (`KvEndpoint{host,port,protocol}`) — a bind wildcard (`*`/`0.0.0.0`/`::`) is rewritten to this node's advertised IP so a remote router can connect. Empty when no publisher. |

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

`tests.rs` provides the shared tonic/mock-engine harness. Focused modules under
`tests/` cover discovery, generation, lifecycle, LoRA, media, and topology.
Real-engine coverage runs on the cluster; there is no local GPU.
