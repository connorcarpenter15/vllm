# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the OpenEngine servicer (no vLLM/GPU required)."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import _bootstrap  # noqa: E402
from fakes import (  # noqa: E402
    FakeCompletionOutput,
    FakeEngineClient,
    FakeRequestOutput,
    fake_input_builder,
)

servicer_mod, pb = _bootstrap.load()
config_mod = __import__(
    "_oe_standalone.config", fromlist=["EngineServeConfig", "ModelConfig"]
)

OpenEngineServicer = servicer_mod.OpenEngineServicer
EngineServeConfig = config_mod.EngineServeConfig
ModelConfig = config_mod.ModelConfig


class _Ctx:
    """Minimal grpc servicer context stand-in for direct method calls."""

    def __init__(self) -> None:
        self.aborted = None

    async def abort(self, code, details):
        self.aborted = (code, details)
        raise RuntimeError(f"aborted: {details}")


def _agg_config(**kw):
    return EngineServeConfig(
        engine_name="vllm",
        engine_version="test",
        role=pb.ENGINE_ROLE_AGGREGATED,
        model=ModelConfig(model_id="m", served_model_name="m", total_kv_blocks=100),
        **kw,
    )


def _gen_request(token_ids=(1, 2, 3), max_tokens=8):
    return pb.GenerateRequest(
        request_id="req-1",
        model="m",
        token_ids=pb.TokenIds(ids=list(token_ids)),
        sampling=pb.SamplingParams(max_tokens=max_tokens),
    )


async def _collect(agen):
    return [x async for x in agen]


def test_generate_happy_path():
    script = [
        FakeRequestOutput([FakeCompletionOutput([10])]),
        FakeRequestOutput([FakeCompletionOutput([11, 12])]),
        FakeRequestOutput(
            [FakeCompletionOutput([13], finish_reason="stop")],
            prompt_token_ids=[1, 2, 3],
        ),
    ]
    engine = FakeEngineClient(script)
    svc = OpenEngineServicer(
        engine, _agg_config(), input_builder=fake_input_builder
    )

    responses = asyncio.run(_collect(svc.Generate(_gen_request(), _Ctx())))

    tokens = [
        list(r.token.token_ids) for r in responses if r.WhichOneof("event") == "token"
    ]
    assert tokens == [[10], [11, 12], [13]]

    finished = [r for r in responses if r.WhichOneof("event") == "finished"]
    assert len(finished) == 1
    assert finished[0].finished.reason == pb.FINISH_REASON_STOP
    assert finished[0].usage.prompt_tokens == 3
    assert finished[0].usage.completion_tokens == 4
    assert finished[0].usage.total_tokens == 7


def test_generate_length_finish():
    script = [
        FakeRequestOutput(
            [FakeCompletionOutput([10], finish_reason="length")],
            prompt_token_ids=[1],
        )
    ]
    svc = OpenEngineServicer(
        FakeEngineClient(script), _agg_config(), input_builder=fake_input_builder
    )
    responses = asyncio.run(_collect(svc.Generate(_gen_request(), _Ctx())))
    finished = [r for r in responses if r.WhichOneof("event") == "finished"]
    assert finished[0].finished.reason == pb.FINISH_REASON_LENGTH


def test_generate_no_outputs_errors():
    script = [FakeRequestOutput([])]
    svc = OpenEngineServicer(
        FakeEngineClient(script), _agg_config(), input_builder=fake_input_builder
    )
    responses = asyncio.run(_collect(svc.Generate(_gen_request(), _Ctx())))
    errors = [r for r in responses if r.WhichOneof("event") == "error"]
    assert errors and errors[0].error.code == pb.ERROR_CODE_INTERNAL


def test_generate_input_build_failure():
    def bad_builder(request, config):
        raise ValueError("bad input")

    svc = OpenEngineServicer(
        FakeEngineClient([]), _agg_config(), input_builder=bad_builder
    )
    responses = asyncio.run(_collect(svc.Generate(_gen_request(), _Ctx())))
    assert len(responses) == 1
    assert responses[0].error.code == pb.ERROR_CODE_INVALID_ARGUMENT


def test_prefill_emits_prefill_ready_with_kv_session():
    kv_params = {"remote_block_ids": [1, 2], "remote_engine_id": "eng-7"}
    script = [
        FakeRequestOutput(
            [FakeCompletionOutput([99], finish_reason="stop")],
            prompt_token_ids=[1, 2, 3],
            kv_transfer_params=kv_params,
        )
    ]
    cfg = _agg_config()
    cfg.role = pb.ENGINE_ROLE_PREFILL
    cfg.kv_transfer_backend = "nixl"
    svc = OpenEngineServicer(
        FakeEngineClient(script), cfg, input_builder=fake_input_builder
    )
    responses = asyncio.run(_collect(svc.Generate(_gen_request(), _Ctx())))

    ready = [r for r in responses if r.WhichOneof("event") == "prefill_ready"]
    assert len(ready) == 1
    sess = ready[0].prefill_ready.kv_session
    assert sess.session_id == "req-1"
    assert sess.transfer_backend == "nixl"
    recovered = servicer_mod.transfer_params_from_kv_session(sess)
    assert recovered == kv_params


def test_kv_session_roundtrip():
    params = {"a": 1, "b": [2, 3], "c": None}
    sess = servicer_mod.kv_session_from_transfer_params("s1", "nixl", params, dp_rank=2)
    assert sess.dp_rank == 2
    assert servicer_mod.transfer_params_from_kv_session(sess) == params


def test_abort_known_and_unknown():
    svc = OpenEngineServicer(
        FakeEngineClient([]), _agg_config(), input_builder=fake_input_builder
    )
    # Unknown request id -> NOT_FOUND.
    resp = asyncio.run(
        svc.Abort(pb.AbortRequest(request_id="nope"), _Ctx())
    )
    assert resp.status == pb.ABORT_STATUS_NOT_FOUND

    # Mark one active, then abort it.
    svc._active["live"] = True
    resp = asyncio.run(svc.Abort(pb.AbortRequest(request_id="live"), _Ctx()))
    assert resp.status == pb.ABORT_STATUS_ABORTED
    assert "live" in svc._engine.aborted


def test_abort_all():
    engine = FakeEngineClient([])
    svc = OpenEngineServicer(engine, _agg_config(), input_builder=fake_input_builder)
    svc._active.update({"a": True, "b": True})
    resp = asyncio.run(svc.Abort(pb.AbortRequest(abort_all=True), _Ctx()))
    assert resp.status == pb.ABORT_STATUS_ABORTED
    assert set(engine.aborted) == {"a", "b"}


def test_health_ready_and_not_ready():
    svc = OpenEngineServicer(
        FakeEngineClient([], healthy=True),
        _agg_config(),
        input_builder=fake_input_builder,
    )
    resp = asyncio.run(svc.Health(pb.HealthRequest(), _Ctx()))
    assert resp.state == pb.HEALTH_STATE_READY

    svc2 = OpenEngineServicer(
        FakeEngineClient([], healthy=False),
        _agg_config(),
        input_builder=fake_input_builder,
    )
    resp2 = asyncio.run(svc2.Health(pb.HealthRequest(), _Ctx()))
    assert resp2.state == pb.HEALTH_STATE_NOT_READY


def test_drain_completes_when_idle():
    svc = OpenEngineServicer(
        FakeEngineClient([]), _agg_config(), input_builder=fake_input_builder
    )
    responses = asyncio.run(
        _collect(svc.Drain(pb.DrainRequest(stop_accepting_new_requests=True), _Ctx()))
    )
    assert responses[0].state == pb.DRAIN_STATE_STARTED
    assert responses[-1].state == pb.DRAIN_STATE_COMPLETE
    assert svc._draining is True


def test_drain_rejects_new_generate():
    svc = OpenEngineServicer(
        FakeEngineClient([]), _agg_config(), input_builder=fake_input_builder
    )
    svc._draining = True
    responses = asyncio.run(_collect(svc.Generate(_gen_request(), _Ctx())))
    assert len(responses) == 1
    assert responses[0].error.code == pb.ERROR_CODE_DRAINING


def test_get_engine_info_and_model_info():
    svc = OpenEngineServicer(
        FakeEngineClient([]), _agg_config(), input_builder=fake_input_builder
    )
    info = asyncio.run(svc.GetEngineInfo(pb.GetEngineInfoRequest(), _Ctx()))
    assert info.engine_name == "vllm"
    assert info.role == pb.ENGINE_ROLE_AGGREGATED
    minfo = asyncio.run(svc.GetModelInfo(pb.GetModelInfoRequest(), _Ctx()))
    assert minfo.served_model_name == "m"
    assert minfo.total_kv_blocks == 100


def test_get_load_defaults():
    svc = OpenEngineServicer(
        FakeEngineClient([]), _agg_config(), input_builder=fake_input_builder
    )
    load = asyncio.run(svc.GetLoad(pb.GetLoadRequest(), _Ctx()))
    assert load.total_kv_blocks == 100
    assert load.running_requests == 0


def test_kv_event_sources_filtering():
    from _oe_standalone.config import KvEventSourceConfig

    cfg = _agg_config(
        kv_event_sources=[
            KvEventSourceConfig(endpoint="tcp://h:5557", data_parallel_rank=0),
            KvEventSourceConfig(endpoint="tcp://h:5558", data_parallel_rank=1),
        ]
    )
    svc = OpenEngineServicer(
        FakeEngineClient([]), cfg, input_builder=fake_input_builder
    )
    resp = asyncio.run(
        svc.GetKvEventSources(
            pb.GetKvEventSourcesRequest(data_parallel_ranks=[1]), _Ctx()
        )
    )
    assert len(resp.sources) == 1
    assert resp.sources[0].data_parallel_rank == 1
