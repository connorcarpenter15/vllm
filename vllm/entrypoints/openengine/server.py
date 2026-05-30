# SPDX-License-Identifier: Apache-2.0
"""OpenEngine v1 gRPC server lifecycle, wrapping a vLLM ``AsyncLLM``.

``OpenEngineServer`` is the vLLM-side glue:

* reads ``vllm_config`` to build the (vLLM-free) :class:`EngineServeConfig`,
* constructs an :class:`OpenEngineServicer` bound to the live ``EngineClient``,
* owns the ``grpc.aio`` server lifecycle (``start`` / ``shutdown``).

It is wired into ``vllm serve`` from ``entrypoints/openai/api_server.py`` when
``--openengine-port`` is set; see that file's lifespan hook.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import grpc

from .config import (
    EngineServeConfig,
    KvConnectorConfig,
    KvEventSourceConfig,
    ModelConfig,
    ParallelismConfig,
)
from ._openengine import openengine_pb2 as pb
from ._openengine import openengine_pb2_grpc as pb_grpc
from ._translate import build_engine_inputs
from .servicer import OpenEngineServicer

logger = logging.getLogger(__name__)


def _role_from_kv_config(kv_transfer_config: Any) -> int:
    """Derive the OpenEngine role from vLLM's kv_transfer_config.kv_role."""
    if kv_transfer_config is None:
        return pb.ENGINE_ROLE_AGGREGATED
    role = getattr(kv_transfer_config, "kv_role", None)
    if role == "kv_producer":
        return pb.ENGINE_ROLE_PREFILL
    if role == "kv_consumer":
        return pb.ENGINE_ROLE_DECODE
    return pb.ENGINE_ROLE_AGGREGATED


def _kv_event_sources_from_config(vllm_config: Any) -> list[KvEventSourceConfig]:
    """Surface vLLM's existing ZMQ KV-event publisher as OpenEngine sources.

    This is the recommended v1 KV-event path: advertise the ZMQ endpoints so
    Dynamo's existing subscriber consumes them unchanged.
    """
    kv_events_config = getattr(vllm_config, "kv_events_config", None)
    if kv_events_config is None:
        return []
    if not getattr(kv_events_config, "enable_kv_cache_events", False):
        return []
    if getattr(kv_events_config, "publisher", None) != "zmq":
        return []

    parallel = getattr(vllm_config, "parallel_config", None)
    dp_size = getattr(parallel, "data_parallel_size", 1) or 1
    dp_start = getattr(parallel, "data_parallel_rank", 0) or 0

    base_endpoint = getattr(kv_events_config, "endpoint", "")
    replay_endpoint = getattr(kv_events_config, "replay_endpoint", "") or ""
    buffer_steps = getattr(kv_events_config, "buffer_steps", 0) or 0
    hwm = getattr(kv_events_config, "hwm", 0) or 0
    max_queue_size = getattr(kv_events_config, "max_queue_size", 0) or 0

    # vLLM offsets the publisher port per DP rank; reuse its helper if present.
    try:
        from vllm.distributed.kv_events import ZmqEventPublisher

        def _endpoint_for(rank: int) -> str:
            return ZmqEventPublisher.offset_endpoint_port(
                base_endpoint, data_parallel_rank=rank
            ).replace("*", "127.0.0.1")
    except Exception:  # noqa: BLE001 - fall back to the unoffset endpoint
        def _endpoint_for(rank: int) -> str:
            return base_endpoint.replace("*", "127.0.0.1")

    sources = []
    for rank in range(dp_start, dp_start + dp_size):
        sources.append(
            KvEventSourceConfig(
                transport="zmq",
                endpoint=_endpoint_for(rank),
                replay_endpoint=replay_endpoint,
                data_parallel_rank=rank,
                encoding="msgpack",
                buffer_steps=buffer_steps,
                hwm=hwm,
                max_queue_size=max_queue_size,
            )
        )
    return sources


def build_serve_config(
    vllm_config: Any,
    *,
    instance_id: str = "",
    role: Optional[int] = None,
) -> EngineServeConfig:
    """Extract an :class:`EngineServeConfig` from a built ``vllm_config``.

    Defensive ``getattr`` throughout: this runs inside vLLM where the config
    objects exist, but field layout shifts between vLLM releases.
    """
    try:
        from vllm.version import __version__ as vllm_version
    except Exception:  # noqa: BLE001
        vllm_version = ""

    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)

    if role is None:
        role = _role_from_kv_config(kv_transfer_config)

    model_id = getattr(model_config, "model", "") if model_config else ""
    served = (
        getattr(model_config, "served_model_name", None) or model_id
        if model_config
        else ""
    )

    model = ModelConfig(
        model_id=model_id,
        served_model_name=served or model_id,
        max_context_length=int(getattr(model_config, "max_model_len", 0) or 0),
        kv_block_size=int(getattr(cache_config, "block_size", 0) or 0),
        total_kv_blocks=int(getattr(cache_config, "num_gpu_blocks", 0) or 0),
        max_running_requests=int(getattr(scheduler_config, "max_num_seqs", 0) or 0),
        max_batched_tokens=int(
            getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
        ),
        supports_multimodal=bool(
            getattr(model_config, "is_multimodal_model", False)
        ),
    )

    parallelism = ParallelismConfig(
        tensor_parallel_size=int(
            getattr(parallel_config, "tensor_parallel_size", 1) or 1
        ),
        pipeline_parallel_size=int(
            getattr(parallel_config, "pipeline_parallel_size", 1) or 1
        ),
        data_parallel_size=int(getattr(parallel_config, "data_parallel_size", 1) or 1),
        data_parallel_rank=int(getattr(parallel_config, "data_parallel_rank", 0) or 0),
        data_parallel_start_rank=int(
            getattr(parallel_config, "data_parallel_rank", 0) or 0
        ),
    )

    transfer_backend = (
        getattr(kv_transfer_config, "kv_connector", "") if kv_transfer_config else ""
    )
    kv_connector = KvConnectorConfig(
        enabled=kv_transfer_config is not None,
        transfer_backend=transfer_backend or "",
        supports_remote_prefill=True,
        supports_decode_pull=True,
        supports_abort_cleanup=True,
        supports_drain=True,
    )

    return EngineServeConfig(
        engine_name="vllm",
        engine_version=vllm_version,
        role=role,
        instance_id=instance_id,
        supported_models=[m for m in [served, model_id] if m],
        parallelism=parallelism,
        kv_connector=kv_connector,
        model=model,
        kv_event_sources=_kv_event_sources_from_config(vllm_config),
        kv_transfer_backend=transfer_backend or "nixl",
    )


class OpenEngineServer:
    def __init__(
        self,
        engine_client: Any,
        vllm_config: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 50051,
        instance_id: str = "",
        role: Optional[int] = None,
        max_message_mb: int = 64,
    ) -> None:
        self._host = host
        self._port = port
        self._config = build_serve_config(
            vllm_config, instance_id=instance_id, role=role
        )
        self._servicer = OpenEngineServicer(
            engine_client,
            self._config,
            input_builder=build_engine_inputs,
        )
        opts = [
            ("grpc.max_send_message_length", max_message_mb * 1024 * 1024),
            ("grpc.max_receive_message_length", max_message_mb * 1024 * 1024),
        ]
        self._server = grpc.aio.server(options=opts)
        pb_grpc.add_OpenEngineServicer_to_server(self._servicer, self._server)
        self._bound_port = self._server.add_insecure_port(f"{host}:{port}")

    @property
    def config(self) -> EngineServeConfig:
        return self._config

    @property
    def port(self) -> int:
        return self._bound_port

    async def start(self) -> None:
        await self._server.start()
        logger.info(
            "OpenEngine gRPC server listening on %s:%s (role=%s, model=%s)",
            self._host,
            self._bound_port,
            self._config.role,
            self._config.model.served_model_name,
        )

    async def wait_for_termination(self) -> None:
        await self._server.wait_for_termination()

    async def shutdown(self, grace: float = 5.0) -> None:
        logger.info("OpenEngine gRPC server shutting down (grace=%ss)", grace)
        await self._server.stop(grace)
