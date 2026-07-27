// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::protocol::OpaqueValue;
use crate::protocol::dtype::ModelDtype;

/// Decoded engine startup-handshake payload sent on the handshake socket.
///
/// Original Python payload construction:
/// <https://github.com/vllm-project/vllm/blob/c8d98f81f6/vllm/v1/engine/core.py#L1000-L1035>
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadyMessage {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub local: Option<bool>,
    #[serde(default)]
    pub headless: Option<bool>,
    #[serde(default)]
    pub parallel_config_hash: Option<String>,
}

/// Post-initialization configuration sent from each engine on the input socket
/// registration message, after the handshake completes.
///
/// Contains values that may differ from the original config (e.g.
/// `max_model_len` after KV cache auto-fitting, `num_gpu_blocks` after
/// profiling).
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/c9340e6f35/vllm/v1/engine/__init__.py#L68-L80>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreReadyResponse {
    /// Engine-reported maximum model context length (auto-fitted after
    /// KV cache profiling and may differ from the original config value).
    pub max_model_len: u64,
    /// Number of GPU blocks available for KV cache on this engine.
    pub num_gpu_blocks: u64,
    /// KV cache block size (tokens per block).
    pub block_size: u64,
    /// DP coordinator stats publish address, if applicable.
    pub dp_stats_address: Option<String>,
    /// Effective model dtype after Python vLLM resolves `--dtype`.
    pub dtype: ModelDtype,
    /// Python vLLM version reported by the engine process.
    pub vllm_version: String,
    /// World size (TP * PP) from the parallel config.
    pub world_size: u64,
    /// Data parallelism size from the parallel config.
    pub data_parallel_size: u64,
    /// Total KV cache capacity in tokens, if reported.
    pub kv_cache_size_tokens: Option<u64>,
    /// Maximum achievable request concurrency given the KV cache, if reported.
    pub kv_cache_max_concurrency: Option<f64>,
    /// Tensor-parallel size of this engine.
    #[serde(default = "one_u32")]
    pub tensor_parallel_size: u32,
    /// Pipeline-parallel size of this engine.
    #[serde(default = "one_u32")]
    pub pipeline_parallel_size: u32,
    /// This engine's data-parallel rank.
    #[serde(default)]
    pub data_parallel_rank: u32,
    /// Scheduler cap on concurrently running sequences.
    #[serde(default)]
    pub max_num_seqs: u64,
    /// Scheduler cap on batched tokens per step.
    #[serde(default)]
    pub max_num_batched_tokens: u64,
    /// Configured KV connector name, if any.
    #[serde(default)]
    pub kv_connector: Option<String>,
    /// KV transfer role (`kv_producer`, `kv_consumer`, or `kv_both`).
    #[serde(default)]
    pub kv_role: Option<String>,
    /// KV connector engine identifier.
    #[serde(default)]
    pub kv_engine_id: Option<String>,
    /// KV-event publisher backend (`null` or `zmq`).
    #[serde(default)]
    pub kv_events_publisher: Option<String>,
    /// ZMQ endpoint used by the KV-event publisher.
    #[serde(default)]
    pub kv_events_endpoint: Option<String>,
    /// Optional ZMQ replay endpoint used by the KV-event publisher.
    #[serde(default)]
    pub kv_events_replay_endpoint: Option<String>,
    /// Topic used by the KV-event publisher.
    #[serde(default)]
    pub kv_events_topic: Option<String>,
    /// Number of event batches retained for replay.
    #[serde(default)]
    pub kv_events_buffer_steps: u64,
    /// Publisher high-water mark.
    #[serde(default)]
    pub kv_events_hwm: u64,
    /// Maximum publisher queue size.
    #[serde(default)]
    pub kv_events_max_queue_size: u64,
    /// Whether dynamic LoRA was enabled at engine startup.
    #[serde(default)]
    pub supports_lora: bool,
    /// Maximum number of active LoRA adapters.
    #[serde(default)]
    pub max_loras: u32,
}

fn one_u32() -> u32 {
    1
}

/// Frontend-owned ZMQ addresses that are sent to the engine during startup
/// handshake initialization.
///
/// Original Python definition (`EngineZmqAddresses`):
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/utils.py#L53-L67>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeAddresses {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub coordinator_input: Option<String>,
    pub coordinator_output: Option<String>,
    pub frontend_stats_publish_address: Option<String>,
}

/// Startup handshake payload sent from the frontend to initialize an engine
/// after receiving `HELLO`.
///
/// Original Python definition (`EngineHandshakeMetadata`):
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/utils.py#L69-L77>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeInitMessage {
    pub addresses: HandshakeAddresses,
    pub parallel_config: BTreeMap<String, OpaqueValue>,
}
