# OpenEngine server (vLLM side)

Serves the OpenEngine v1 gRPC contract on top of a vLLM `AsyncLLM`, so a
Dynamo sidecar worker can drive this engine across a process boundary. This
module is the vLLM half of the sidecar integration; the Dynamo half is
`dynamo/components/src/dynamo/vllm/sidecar/`.

## Scope discipline (keep the fork edit minimal)

The vLLM fork change is intentionally tiny:

- **Everything self-contained here** in `entrypoints/openengine/`.
- The **only** edits outside this dir:
  - `entrypoints/openai/cli_args.py` — define `--openengine-port` /
    `--openengine-host` flags.
  - `entrypoints/openai/api_server.py` — a small hook: when
    `--openengine-port` is set, instantiate
    `OpenEngineServer(engine_client=async_llm, host=..., port=...)`, start it
    as a background asyncio task, and register a lifespan shutdown that awaits
    `server.shutdown(grace=...)`.

**Wrap `AsyncLLM` strictly through its `EngineClient` ABC** (`generate`,
`abort`, `check_health`, …). Do **not** edit `v1/engine/`, `kv_events.py`, or
`kv_transfer/`. If you need data from those, read it off the already-built
`vllm_config` the server is handed at construction.

## Files (planned)

```
openengine/
  __init__.py      exports OpenEngineServer, OpenEngineServicer
  server.py        OpenEngineServer(engine_client, host, port): grpc.aio lifecycle, Drain
  servicer.py      OpenEngineServicer(OpenEngineServicer): one EngineClient, all RPCs
  _openengine/     GENERATED stubs (openengine_pb2*.py) — do NOT hand-edit
  CLAUDE.md        this file
```

`_openengine/` is generated from the canonical proto:
from the `openengine/` peer dir run
`./gen.sh ../vllm/vllm/entrypoints/openengine/_openengine` (see
`openengine/README.md`). Regenerate after any proto change. Build artifact.
Add `grpcio` (runtime) + `grpcio-tools` (dev) to vLLM's deps.

## RPC implementation notes

- **Generate** — map `GenerateRequest.token_ids|prompt` + `sampling` + `stop`
  → `TokensPrompt` + `SamplingParams`. For decode role, lift
  `kv_session.attributes` into the engine call as `kv_transfer_params` (mirror
  `entrypoints/serve/disagg/serving.py` so behavior matches the HTTP disagg
  path). `async for out in engine_client.generate(...)`: emit
  `GenerateResponse(token=TokenOutput(...))` per delta; prefill role emits
  `prefill_ready=PrefillReady(kv_session=...)` first; on stop emit
  `finished=GenerationFinished(...)` + `Usage`. Map errors → `EngineError`.
- **GetEngineInfo** — from `vllm_config`: `engine_name="vllm"`, version, role
  from `--kv-role`/`--kv-rank`, `ParallelismInfo` from
  `vllm_config.parallel_config`, `KvConnectorInfo` from
  `vllm_config.kv_transfer_config`.
- **GetModelInfo** — from `model_config` / `cache_config` /
  `scheduler_config`.
- **GetLoad** — drive off the engine stat snapshot. v1: populate
  `running_requests`, `queued_requests`, `used_kv_blocks`, `total_kv_blocks`;
  leave the rest zeroed rather than block the milestone.
- **Health** — `engine_client.check_health()`; on `include_inference_probe`,
  run a 1-token canary and report `HealthCheck(name="inference_probe", ...)`.
- **Abort** — `engine_client.abort(request_id)`, stable `ABORT_STATUS_*`
  mapping. `abort_all` is best-effort over tracked active request IDs.
- **Drain** — stop accepting new `Generate`, stream `DrainResponse(state,
  in_flight_requests, open_kv_sessions, ...)` until 0 or `deadline_ms`; if
  `abort_after_deadline`, abort the remainder and emit
  `DRAIN_STATE_COMPLETE`.
- **GetKvConnectorInfo** — snapshot of `vllm_config.kv_transfer_config`.
- **GetKvEventSources** — **the recommended v1 KV-event path.** If
  `kv_events_config.publisher == "zmq"`, return one `KvEventSource(transport=
  "zmq", endpoint=..., topic="", replay_endpoint=..., data_parallel_rank=...)`
  per DP rank, surfacing vLLM's existing `ZmqEventPublisher` config verbatim.
- **SubscribeKvEvents** — `UNIMPLEMENTED` in v1 (covered by
  `GetKvEventSources`). Follow-up.
- **SubscribeRuntimeEvents** — `UNIMPLEMENTED` in v1.

## Reuse (don't reinvent)

- `AsyncLLM.from_vllm_config`, `AsyncEngineArgs.from_cli_args`,
  `AsyncLLM.generate/abort/check_health` — wrap, don't reimplement.
- `ZmqEventPublisher` config (endpoint, replay_endpoint, buffer_steps, DP-rank
  offset) — surface verbatim through `GetKvEventSources`.
- `entrypoints/serve/disagg/serving.py` `kv_transfer_params` pack/unpack — copy
  that logic so disagg behavior is identical to the existing HTTP path.

## Testing

- Unit: `OpenEngineServicer` against a fake `EngineClient`; assert proto wire
  shape for happy path, abort, drain (CPU-only, local).
- Smoke (cluster): `vllm serve --model <small> --openengine-port 50051`, then
  call each RPC via `grpcurl` / the generated stub. Real-engine smoke runs on
  computelab or lyris (see root `CLAUDE.md`).
