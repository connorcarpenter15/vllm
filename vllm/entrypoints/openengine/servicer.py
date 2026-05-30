# SPDX-License-Identifier: Apache-2.0
"""OpenEngine v1 gRPC servicer backed by a vLLM ``EngineClient``.

This module is deliberately **vLLM-free**: it talks to the engine only through
the duck-typed ``EngineClient`` async API (``generate``, ``abort``,
``check_health``) and reads all static metadata from an injected
:class:`~vllm.entrypoints.openengine.config.EngineServeConfig`. The single
vLLM-touching step -- turning an OpenEngine ``GenerateRequest`` into a vLLM
``(prompt, SamplingParams)`` pair -- is injected as ``input_builder`` (default
:func:`vllm.entrypoints.openengine._translate.build_engine_inputs`).

Keeping it vLLM-free lets the Generate/Abort/Drain logic -- the bug-prone part
-- be unit-tested on CPU against a fake engine client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional

from .config import EngineServeConfig, LoadSnapshot
from ._openengine import openengine_pb2 as pb
from ._openengine import openengine_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)

# Attribute key under which the vLLM kv_transfer_params handoff blob is carried
# inside KvSessionRef.attributes. The decode side reverses this.
KV_TRANSFER_PARAMS_ATTR = "kv_transfer_params"

_ROLE_PREFILL = pb.ENGINE_ROLE_PREFILL
_ROLE_DECODE = pb.ENGINE_ROLE_DECODE

# vLLM finish-reason string -> OpenEngine FinishReason enum.
_FINISH_REASON_MAP = {
    "stop": pb.FINISH_REASON_STOP,
    "length": pb.FINISH_REASON_LENGTH,
    "abort": pb.FINISH_REASON_CANCELLED,
    "cancel": pb.FINISH_REASON_CANCELLED,
    "cancelled": pb.FINISH_REASON_CANCELLED,
    "error": pb.FINISH_REASON_ERROR,
}


def map_finish_reason(reason: Optional[str]) -> int:
    if not reason:
        return pb.FINISH_REASON_UNSPECIFIED
    return _FINISH_REASON_MAP.get(str(reason).lower(), pb.FINISH_REASON_STOP)


def kv_session_from_transfer_params(
    session_id: str,
    transfer_backend: str,
    kv_transfer_params: Any,
    dp_rank: int = 0,
) -> "pb.KvSessionRef":
    """Wrap vLLM ``kv_transfer_params`` into an OpenEngine ``KvSessionRef``.

    The blob is JSON-encoded into ``attributes[kv_transfer_params]`` so the
    decode peer can recover it verbatim -- mirroring how vLLM's HTTP disagg
    path threads the same dict through.
    """
    attrs = {}
    if kv_transfer_params is not None:
        attrs[KV_TRANSFER_PARAMS_ATTR] = json.dumps(kv_transfer_params)
    return pb.KvSessionRef(
        session_id=session_id,
        transfer_backend=transfer_backend,
        dp_rank=dp_rank,
        attributes=attrs,
    )


def transfer_params_from_kv_session(kv_session: "pb.KvSessionRef") -> Optional[dict]:
    """Inverse of :func:`kv_session_from_transfer_params` (decode side)."""
    if kv_session is None:
        return None
    blob = kv_session.attributes.get(KV_TRANSFER_PARAMS_ATTR)
    if not blob:
        return None
    return json.loads(blob)


class OpenEngineServicer(pb_grpc.OpenEngineServicer):
    def __init__(
        self,
        engine_client: Any,
        config: EngineServeConfig,
        *,
        input_builder: Callable[..., Any],
        load_provider: Optional[Callable[[], LoadSnapshot]] = None,
    ) -> None:
        self._engine = engine_client
        self._config = config
        self._input_builder = input_builder
        self._load_provider = load_provider
        # request_id -> True for in-flight Generate calls (abort_all / drain).
        self._active: dict[str, bool] = {}
        self._draining = False

    # ----- core inference -------------------------------------------------

    async def Generate(self, request, context) -> AsyncIterator["pb.GenerateResponse"]:
        request_id = request.request_id or uuid.uuid4().hex
        role = self._config.role
        is_prefill = role == _ROLE_PREFILL

        if self._draining:
            yield pb.GenerateResponse(
                request_id=request_id,
                error=pb.EngineError(
                    code=pb.ERROR_CODE_DRAINING,
                    message="engine is draining; not accepting new requests",
                ),
            )
            return

        try:
            prompt, sampling_params = self._input_builder(request, self._config)
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed wire error
            logger.warning("Generate input build failed for %s: %s", request_id, exc)
            yield pb.GenerateResponse(
                request_id=request_id,
                error=pb.EngineError(
                    code=pb.ERROR_CODE_INVALID_ARGUMENT, message=str(exc)
                ),
            )
            return

        prompt_tokens = (
            len(request.token_ids.ids) if request.HasField("token_ids") else 0
        )
        total_completion = 0
        prefill_ready_sent = False

        self._active[request_id] = True
        kv_transfer_params: Any = None
        try:
            gen = self._engine.generate(prompt, sampling_params, request_id)
            async for res in gen:
                outputs = getattr(res, "outputs", None)
                # vLLM populates kv_transfer_params on the final RequestOutput
                # (once prefill completes), not the first — capture the latest.
                # Do this BEFORE the empty-outputs guard: a producer's terminal
                # RequestOutput can carry kv_transfer_params with no new tokens.
                res_kv = getattr(res, "kv_transfer_params", None)
                if res_kv is not None:
                    kv_transfer_params = res_kv

                if not outputs:
                    if is_prefill and getattr(res, "finished", False):
                        # Terminal producer marker (no tokens, KV ready).
                        if not prefill_ready_sent:
                            kv_session = kv_session_from_transfer_params(
                                session_id=request_id,
                                transfer_backend=self._config.kv_transfer_backend,
                                kv_transfer_params=kv_transfer_params,
                            )
                            yield pb.GenerateResponse(
                                request_id=request_id,
                                prefill_ready=pb.PrefillReady(
                                    kv_session=kv_session
                                ),
                            )
                            prefill_ready_sent = True
                        total = prompt_tokens + total_completion
                        yield pb.GenerateResponse(
                            request_id=request_id,
                            finished=pb.GenerationFinished(
                                reason=pb.FINISH_REASON_STOP,
                                message="stop",
                            ),
                            usage=pb.Usage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=total_completion,
                                total_tokens=total,
                            ),
                        )
                        break
                    yield pb.GenerateResponse(
                        request_id=request_id,
                        error=pb.EngineError(
                            code=pb.ERROR_CODE_INTERNAL,
                            message="no outputs from vLLM engine",
                        ),
                    )
                    break

                for output in outputs:
                    delta = list(getattr(output, "token_ids", None) or [])
                    finish_reason = getattr(output, "finish_reason", None)

                    if delta:
                        total_completion += len(delta)
                        yield pb.GenerateResponse(
                            request_id=request_id,
                            token=pb.TokenOutput(token_ids=delta),
                        )

                    if finish_reason:
                        # Prefill role: KV is ready for the decode peer to pull
                        # once prefill finishes. Emit PrefillReady here (not on
                        # the first chunk) so kv_transfer_params is populated.
                        if is_prefill and not prefill_ready_sent:
                            kv_session = kv_session_from_transfer_params(
                                session_id=request_id,
                                transfer_backend=self._config.kv_transfer_backend,
                                kv_transfer_params=kv_transfer_params,
                            )
                            yield pb.GenerateResponse(
                                request_id=request_id,
                                prefill_ready=pb.PrefillReady(
                                    kv_session=kv_session
                                ),
                            )
                            prefill_ready_sent = True

                        total = prompt_tokens + total_completion
                        yield pb.GenerateResponse(
                            request_id=request_id,
                            finished=pb.GenerationFinished(
                                reason=map_finish_reason(finish_reason),
                                message=str(finish_reason),
                            ),
                            usage=pb.Usage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=total_completion,
                                total_tokens=total,
                            ),
                        )
        except asyncio.CancelledError:
            await self._safe_abort(request_id)
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed wire error
            logger.exception("Generate failed for %s", request_id)
            yield pb.GenerateResponse(
                request_id=request_id,
                error=pb.EngineError(code=pb.ERROR_CODE_INTERNAL, message=str(exc)),
            )
        finally:
            self._active.pop(request_id, None)

    # ----- metadata -------------------------------------------------------

    async def GetEngineInfo(self, request, context) -> "pb.EngineInfo":
        c = self._config
        return pb.EngineInfo(
            engine_name=c.engine_name,
            engine_version=c.engine_version,
            api_version=c.api_version,
            role=c.role,
            instance_id=c.instance_id,
            supported_models=c.supported_models,
            parallelism=pb.ParallelismInfo(
                tensor_parallel_size=c.parallelism.tensor_parallel_size,
                pipeline_parallel_size=c.parallelism.pipeline_parallel_size,
                data_parallel_size=c.parallelism.data_parallel_size,
                data_parallel_rank=c.parallelism.data_parallel_rank,
                data_parallel_start_rank=c.parallelism.data_parallel_start_rank,
            ),
            kv_connector=self._kv_connector_info(),
        )

    async def GetModelInfo(self, request, context) -> "pb.ModelInfo":
        m = self._config.model
        return pb.ModelInfo(
            model_id=m.model_id,
            served_model_name=m.served_model_name,
            served_model_aliases=m.served_model_aliases,
            max_context_length=m.max_context_length,
            max_output_tokens=m.max_output_tokens,
            kv_block_size=m.kv_block_size,
            total_kv_blocks=m.total_kv_blocks,
            max_running_requests=m.max_running_requests,
            max_batched_tokens=m.max_batched_tokens,
            tokenizer_modes=m.tokenizer_modes,
            supports_text_input=m.supports_text_input,
            supports_token_ids_input=m.supports_token_ids_input,
            supports_logprobs=m.supports_logprobs,
            supports_guided_decoding=m.supports_guided_decoding,
            supports_lora=m.supports_lora,
            supports_multimodal=m.supports_multimodal,
        )

    async def GetLoad(self, request, context) -> "pb.LoadInfo":
        snap = self._load_provider() if self._load_provider else LoadSnapshot(
            running_requests=len(self._active),
            total_kv_blocks=self._config.model.total_kv_blocks,
        )
        return pb.LoadInfo(
            instance_id=self._config.instance_id,
            timestamp_unix_nanos=time.time_ns(),
            running_requests=snap.running_requests,
            queued_requests=snap.queued_requests,
            active_kv_sessions=snap.active_kv_sessions,
            used_kv_blocks=snap.used_kv_blocks,
            total_kv_blocks=snap.total_kv_blocks,
            running_tokens=snap.running_tokens,
            waiting_tokens=snap.waiting_tokens,
            prefill_batch_size=snap.prefill_batch_size,
            decode_batch_size=snap.decode_batch_size,
        )

    def _kv_connector_info(self) -> "pb.KvConnectorInfo":
        k = self._config.kv_connector
        return pb.KvConnectorInfo(
            enabled=k.enabled,
            transfer_backend=k.transfer_backend,
            supported_protocols=k.supported_protocols,
            supports_remote_prefill=k.supports_remote_prefill,
            supports_decode_pull=k.supports_decode_pull,
            supports_abort_cleanup=k.supports_abort_cleanup,
            supports_drain=k.supports_drain,
            schema_version=k.schema_version,
        )

    async def GetKvConnectorInfo(self, request, context) -> "pb.KvConnectorInfo":
        return self._kv_connector_info()

    async def GetKvEventSources(
        self, request, context
    ) -> "pb.GetKvEventSourcesResponse":
        wanted = set(request.data_parallel_ranks)
        sources = []
        for s in self._config.kv_event_sources:
            if wanted and s.data_parallel_rank not in wanted:
                continue
            sources.append(
                pb.KvEventSource(
                    transport=s.transport,
                    endpoint=s.endpoint,
                    topic=s.topic,
                    replay_endpoint=s.replay_endpoint,
                    data_parallel_rank=s.data_parallel_rank,
                    encoding=s.encoding,
                    schema_version=s.schema_version,
                    buffer_steps=s.buffer_steps,
                    hwm=s.hwm,
                    max_queue_size=s.max_queue_size,
                )
            )
        return pb.GetKvEventSourcesResponse(sources=sources)

    async def SubscribeKvEvents(self, request, context):
        # v1: covered by GetKvEventSources compatibility path.
        await context.abort(
            _UNIMPLEMENTED,
            "SubscribeKvEvents not implemented in v1; use GetKvEventSources",
        )
        return
        yield  # pragma: no cover - makes this an async generator

    async def SubscribeRuntimeEvents(self, request, context):
        await context.abort(
            _UNIMPLEMENTED, "SubscribeRuntimeEvents not implemented in v1"
        )
        return
        yield  # pragma: no cover

    # ----- health / lifecycle --------------------------------------------

    async def Health(self, request, context) -> "pb.HealthResponse":
        checks = []
        try:
            await self._engine.check_health()
            engine_state = pb.HEALTH_STATE_READY
            checks.append(
                pb.HealthCheck(name="engine", state=pb.HEALTH_STATE_READY)
            )
        except Exception as exc:  # noqa: BLE001
            engine_state = pb.HEALTH_STATE_NOT_READY
            checks.append(
                pb.HealthCheck(
                    name="engine",
                    state=pb.HEALTH_STATE_NOT_READY,
                    message=str(exc),
                )
            )

        state = pb.HEALTH_STATE_DRAINING if self._draining else engine_state
        return pb.HealthResponse(state=state, checks=checks)

    async def Abort(self, request, context) -> "pb.AbortResponse":
        if request.abort_all:
            ids = list(self._active.keys())
            for rid in ids:
                await self._safe_abort(rid)
            return pb.AbortResponse(
                status=pb.ABORT_STATUS_ABORTED,
                message=f"aborted {len(ids)} request(s)",
            )

        rid = request.request_id
        if rid and rid not in self._active:
            return pb.AbortResponse(
                status=pb.ABORT_STATUS_NOT_FOUND,
                message=f"request {rid} not in-flight",
            )
        await self._safe_abort(rid)
        return pb.AbortResponse(status=pb.ABORT_STATUS_ABORTED)

    async def Drain(self, request, context) -> AsyncIterator["pb.DrainResponse"]:
        if request.stop_accepting_new_requests:
            self._draining = True

        deadline_ms = request.deadline_ms or 0
        start = time.monotonic()
        yield pb.DrainResponse(
            state=pb.DRAIN_STATE_STARTED,
            in_flight_requests=len(self._active),
        )

        while self._active:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if deadline_ms and elapsed_ms >= deadline_ms:
                if request.abort_after_deadline:
                    for rid in list(self._active.keys()):
                        await self._safe_abort(rid)
                    yield pb.DrainResponse(
                        state=pb.DRAIN_STATE_COMPLETE,
                        in_flight_requests=0,
                        message="deadline reached; aborted remaining requests",
                    )
                    return
                yield pb.DrainResponse(
                    state=pb.DRAIN_STATE_FAILED,
                    in_flight_requests=len(self._active),
                    message="deadline reached with requests still in flight",
                )
                return
            yield pb.DrainResponse(
                state=pb.DRAIN_STATE_IN_PROGRESS,
                in_flight_requests=len(self._active),
            )
            await asyncio.sleep(0.05)

        yield pb.DrainResponse(state=pb.DRAIN_STATE_COMPLETE, in_flight_requests=0)

    # ----- helpers --------------------------------------------------------

    async def _safe_abort(self, request_id: str) -> None:
        if not request_id:
            return
        try:
            await self._engine.abort(request_id)
        except Exception:  # noqa: BLE001 - abort is best-effort
            logger.debug("abort of %s raised (ignored)", request_id, exc_info=True)


# grpc status code for UNIMPLEMENTED, imported lazily to keep import light.
try:  # pragma: no cover
    import grpc

    _UNIMPLEMENTED = grpc.StatusCode.UNIMPLEMENTED
except Exception:  # pragma: no cover
    _UNIMPLEMENTED = None
