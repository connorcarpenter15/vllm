# SPDX-License-Identifier: Apache-2.0
"""Fake vLLM ``EngineClient`` for CPU servicer/wire tests."""

from __future__ import annotations

import asyncio
from typing import Any, Optional


class FakeCompletionOutput:
    def __init__(
        self,
        token_ids: list[int],
        finish_reason: Optional[str] = None,
        text: str = "",
        index: int = 0,
    ) -> None:
        self.token_ids = token_ids
        self.finish_reason = finish_reason
        self.text = text
        self.index = index


class FakeRequestOutput:
    def __init__(
        self,
        outputs: list[FakeCompletionOutput],
        prompt_token_ids: Optional[list[int]] = None,
        kv_transfer_params: Any = None,
    ) -> None:
        self.outputs = outputs
        self.prompt_token_ids = prompt_token_ids or []
        self.kv_transfer_params = kv_transfer_params
        self.finished = any(o.finish_reason for o in outputs)


class FakeEngineClient:
    """Replays a scripted list of RequestOutputs from ``generate``."""

    def __init__(
        self,
        script: list[FakeRequestOutput],
        *,
        healthy: bool = True,
        per_chunk_delay: float = 0.0,
    ) -> None:
        self._script = script
        self.healthy = healthy
        self._per_chunk_delay = per_chunk_delay
        self.aborted: list[str] = []
        self.generate_calls: list[tuple[Any, Any, str]] = []

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.generate_calls.append((prompt, sampling_params, request_id))
        for ro in self._script:
            if self._per_chunk_delay:
                await asyncio.sleep(self._per_chunk_delay)
            else:
                await asyncio.sleep(0)
            yield ro

    async def abort(self, request_id) -> None:
        self.aborted.append(request_id)

    async def check_health(self) -> None:
        if not self.healthy:
            raise RuntimeError("engine unhealthy")


def fake_input_builder(request, config):
    """Stand-in for the vLLM input builder; ignores vLLM types."""
    which = request.WhichOneof("input")
    prompt = list(request.token_ids.ids) if which == "token_ids" else request.prompt
    return prompt, {"sampling": "fake"}
