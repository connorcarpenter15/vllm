# SPDX-License-Identifier: Apache-2.0
"""Static configuration the OpenEngine servicer reports to clients.

This module is intentionally **free of any ``vllm`` import** so the servicer
(which consumes these dataclasses) stays unit-testable on CPU without vLLM.
``server.py`` is the only place that reads ``vllm_config`` and builds an
``EngineServeConfig`` from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParallelismConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    data_parallel_start_rank: int = 0


@dataclass
class KvConnectorConfig:
    enabled: bool = False
    transfer_backend: str = ""
    supported_protocols: list[str] = field(default_factory=list)
    supports_remote_prefill: bool = False
    supports_decode_pull: bool = False
    supports_abort_cleanup: bool = False
    supports_drain: bool = False
    schema_version: int = 0


@dataclass
class KvEventSourceConfig:
    transport: str = "zmq"
    endpoint: str = ""
    topic: str = ""
    replay_endpoint: str = ""
    data_parallel_rank: int = 0
    encoding: str = "msgpack"
    schema_version: int = 0
    buffer_steps: int = 0
    hwm: int = 0
    max_queue_size: int = 0


@dataclass
class ModelConfig:
    model_id: str = ""
    served_model_name: str = ""
    served_model_aliases: list[str] = field(default_factory=list)
    max_context_length: int = 0
    max_output_tokens: int = 0
    kv_block_size: int = 0
    total_kv_blocks: int = 0
    max_running_requests: int = 0
    max_batched_tokens: int = 0
    tokenizer_modes: list[str] = field(default_factory=list)
    supports_text_input: bool = True
    supports_token_ids_input: bool = True
    supports_logprobs: bool = True
    supports_guided_decoding: bool = False
    supports_lora: bool = False
    supports_multimodal: bool = False


@dataclass
class EngineServeConfig:
    """Everything the servicer needs to answer metadata RPCs.

    ``role`` is the OpenEngine ``EngineRole`` enum value (int). Kept as a plain
    int so this module needs no proto import either.
    """

    engine_name: str = "vllm"
    engine_version: str = ""
    api_version: str = "openengine.v1"
    role: int = 1  # ENGINE_ROLE_AGGREGATED
    instance_id: str = ""
    supported_models: list[str] = field(default_factory=list)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    kv_connector: KvConnectorConfig = field(default_factory=KvConnectorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    kv_event_sources: list[KvEventSourceConfig] = field(default_factory=list)
    # Backend label stamped onto KvSessionRef.transfer_backend during the
    # prefill->decode handoff (e.g. "nixl").
    kv_transfer_backend: str = "nixl"


@dataclass
class LoadSnapshot:
    """Point-in-time load, returned by an optional load provider."""

    running_requests: int = 0
    queued_requests: int = 0
    active_kv_sessions: int = 0
    used_kv_blocks: int = 0
    total_kv_blocks: int = 0
    running_tokens: int = 0
    waiting_tokens: int = 0
    prefill_batch_size: int = 0
    decode_batch_size: int = 0
