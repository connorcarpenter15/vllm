# SPDX-License-Identifier: Apache-2.0
"""vLLM-touching translation: OpenEngine ``GenerateRequest`` -> vLLM inputs.

This is the ONLY module in the OpenEngine server that imports ``vllm``. The
servicer injects :func:`build_engine_inputs` as its ``input_builder`` so the
servicer itself stays vLLM-free and CPU-testable.
"""

from __future__ import annotations

from typing import Any

from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind

from .config import EngineServeConfig
from .servicer import transfer_params_from_kv_session

# OpenEngine EngineRole enum values (avoid importing pb here twice; mirror).
_ROLE_PREFILL = 2
_ROLE_DECODE = 3


def build_sampling_params(request, config: EngineServeConfig) -> SamplingParams:
    s = request.sampling
    stop_text: list[str] = []
    stop_token_ids: list[int] = []
    for cond in request.stop:
        which = cond.WhichOneof("condition")
        if which == "stop_text":
            stop_text.append(cond.stop_text)
        elif which == "stop_token_id":
            stop_token_ids.append(cond.stop_token_id)

    kwargs: dict[str, Any] = {}
    # proto3 scalars default to 0; only forward meaningful overrides.
    if s.temperature:
        kwargs["temperature"] = s.temperature
    if s.top_p:
        kwargs["top_p"] = s.top_p
    if s.top_k:
        kwargs["top_k"] = s.top_k
    if s.frequency_penalty:
        kwargs["frequency_penalty"] = s.frequency_penalty
    if s.presence_penalty:
        kwargs["presence_penalty"] = s.presence_penalty
    if s.max_tokens:
        kwargs["max_tokens"] = s.max_tokens
    if s.seed:
        kwargs["seed"] = s.seed
    if stop_text:
        kwargs["stop"] = stop_text
    if stop_token_ids:
        kwargs["stop_token_ids"] = stop_token_ids

    # Stream deltas: AsyncLLM defaults to CUMULATIVE, where each iteration
    # re-emits all tokens so far. The servicer forwards output.token_ids as a
    # per-chunk delta, so DELTA is required to avoid quadratic re-emission.
    kwargs["output_kind"] = RequestOutputKind.DELTA

    params = SamplingParams(**kwargs)

    role = config.role
    if role == _ROLE_PREFILL:
        # Mirror the in-process VllmLLMEngine prefill path: cap to one token
        # and flag the connector to expect a remote decode pull.
        if params.extra_args is None:
            params.extra_args = {}
        kv_defaults = {
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        caller_kv = params.extra_args.get("kv_transfer_params", {})
        params.extra_args["kv_transfer_params"] = {
            **kv_defaults,
            **caller_kv,
            "do_remote_decode": True,
        }
        params.max_tokens = 1
        params.min_tokens = 1
    elif role == _ROLE_DECODE:
        kv_params = transfer_params_from_kv_session(request.kv_session)
        if kv_params is None:
            raise ValueError(
                "decode request missing kv_session.kv_transfer_params; the "
                "prefill peer must populate it for NixlConnector to pull KV"
            )
        if params.extra_args is None:
            params.extra_args = {}
        params.extra_args["kv_transfer_params"] = kv_params

    return params


def build_engine_inputs(request, config: EngineServeConfig):
    """Return ``(prompt, SamplingParams)`` for ``EngineClient.generate``."""
    which = request.WhichOneof("input")
    if which == "token_ids":
        prompt = TokensPrompt(prompt_token_ids=list(request.token_ids.ids))
    elif which == "prompt":
        # Text input path: hand the raw string to vLLM's tokenizer.
        prompt = request.prompt
    else:
        raise ValueError("GenerateRequest must set either token_ids or prompt")

    sampling_params = build_sampling_params(request, config)
    return prompt, sampling_params
