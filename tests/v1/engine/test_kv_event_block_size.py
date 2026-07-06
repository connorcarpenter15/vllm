# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for KV-event block-size selection advertised in the engine
handshake (``EngineCoreReadyResponse.block_size``).

Under the hybrid KV-cache manager ``cache_config.block_size`` is reset to the
*minimum* group block size, which need not match the granularity at which
prefix-cache events are published. Out-of-process frontends (the OpenEngine
sidecar's KV router) index those events, so the handshake must advertise the
*main-attention* group's block size, not the hybrid minimum.
"""

from vllm.v1.engine.core import select_kv_event_block_size


def test_hybrid_prefers_main_attention_over_minimum():
    # The hybrid manager would reset cache_config.block_size to the min (4);
    # the published events use the MLA group's size (256). Selection must
    # return the main-attention size so the router's index aligns.
    metadata = [
        {"group_idx": 0, "kind": "sliding_window", "block_size": 4},
        {"group_idx": 1, "kind": "mla_attention", "block_size": 256},
    ]
    assert select_kv_event_block_size(metadata, fallback_block_size=4) == 256


def test_full_attention_selected():
    metadata = [{"group_idx": 0, "kind": "full_attention", "block_size": 16}]
    assert select_kv_event_block_size(metadata, fallback_block_size=16) == 16


def test_sink_full_attention_selected():
    metadata = [
        {"group_idx": 0, "kind": "mamba", "block_size": 8},
        {"group_idx": 1, "kind": "sink_full_attention", "block_size": 128},
    ]
    assert select_kv_event_block_size(metadata, fallback_block_size=8) == 128


def test_no_main_attention_group_falls_back():
    metadata = [
        {"group_idx": 0, "kind": "mamba", "block_size": 8},
        {"group_idx": 1, "kind": "sliding_window", "block_size": 4},
    ]
    assert select_kv_event_block_size(metadata, fallback_block_size=32) == 32


def test_empty_metadata_falls_back():
    assert select_kv_event_block_size([], fallback_block_size=16) == 16


def test_main_attention_without_block_size_falls_back():
    metadata = [{"group_idx": 0, "kind": "full_attention", "block_size": None}]
    assert select_kv_event_block_size(metadata, fallback_block_size=64) == 64
