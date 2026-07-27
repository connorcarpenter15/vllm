// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenEngine sibling server backed by the same [`crate::state::AppState`] as
//! the OpenAI and native vLLM gRPC frontends.

mod convert;
#[cfg(test)]
mod service_tests;
mod struct_json;

pub(crate) mod pb {
    tonic::include_proto!("openengine.v1");
}

use std::collections::BTreeMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tonic::{Request, Response, Status};
use tracing::info;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextOutputStreamExt as _, TextRequest};

use self::convert::TRANSFER_BACKEND;
use crate::lora::{ActivateLoraError, LoadExactLoraError, UnloadLoraError};
use crate::state::AppState;

pub(crate) use pb::control_server::ControlServer;
pub(crate) use pb::inference_server::InferenceServer;

const SCHEMA_REVISION: u32 = 1;
const MINIMUM_CLIENT_REVISION: u32 = 1;
const SCHEMA_RELEASE: &str = env!("OPENENGINE_SCHEMA_RELEASE");
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(25);
const INFERENCE_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[derive(Clone)]
pub(crate) struct OpenEngineService {
    state: Arc<AppState>,
    instance_id: String,
    advertise_host: String,
    role: pb::EngineRole,
    active_connector_sessions: Arc<AtomicU64>,
    abort_cleanups: Arc<AbortCleanupTasks>,
}

struct ConnectorSessionGuard(Arc<AtomicU64>);

impl ConnectorSessionGuard {
    fn new(counter: Arc<AtomicU64>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for ConnectorSessionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

struct AbortCleanupTasks {
    stopping: AtomicBool,
    shutdown: CancellationToken,
    tasks: TaskTracker,
    pending: AtomicU64,
}

impl AbortCleanupTasks {
    fn new() -> Self {
        Self {
            stopping: AtomicBool::new(false),
            shutdown: CancellationToken::new(),
            tasks: TaskTracker::new(),
            pending: AtomicU64::new(0),
        }
    }

    fn pending(&self) -> u64 {
        self.pending.load(Ordering::SeqCst)
    }

    fn track(self: &Arc<Self>, state: Arc<AppState>, baseline: BTreeMap<u32, u64>) {
        if baseline.is_empty() || self.stopping.load(Ordering::SeqCst) {
            return;
        }

        // Obtain the tracking token before the second stopping check. This
        // closes the race where shutdown could otherwise observe an empty task
        // set immediately before a new cleanup poller is spawned.
        let tracked = self.tasks.token();
        if self.stopping.load(Ordering::SeqCst) {
            drop(tracked);
            return;
        }

        self.pending.fetch_add(1, Ordering::SeqCst);
        let cleanup = self.clone();
        tokio::spawn(async move {
            let _tracked = tracked;
            let _pending = PendingAbortCleanup(cleanup.clone());
            loop {
                if cleanup.shutdown.is_cancelled() {
                    return;
                }
                let loads = state
                    .engine_core_client()
                    .engine_loads()
                    .into_iter()
                    .map(|load| (load.engine_index, load))
                    .collect::<BTreeMap<_, _>>();
                let cleaned = baseline.iter().all(|(engine_index, generation)| {
                    loads.get(engine_index).is_some_and(|load| {
                        load.update_generation > *generation
                            && load.running_requests == 0
                            && load.queued_requests == 0
                    })
                });
                if cleaned {
                    return;
                }
                tokio::select! {
                    biased;
                    _ = cleanup.shutdown.cancelled() => return,
                    _ = tokio::time::sleep(DRAIN_POLL_INTERVAL) => {}
                }
            }
        });
    }

    async fn shutdown(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        self.shutdown.cancel();
        self.tasks.close();
        self.tasks.wait().await;
    }
}

struct PendingAbortCleanup(Arc<AbortCleanupTasks>);

impl Drop for PendingAbortCleanup {
    fn drop(&mut self) {
        self.0.pending.fetch_sub(1, Ordering::SeqCst);
    }
}

impl OpenEngineService {
    pub(crate) fn new(state: Arc<AppState>, advertise_host: String) -> anyhow::Result<Self> {
        let ready = state.engine_core_client().ready_responses();
        anyhow::ensure!(
            !ready.is_empty(),
            "OpenEngine requires at least one engine rank"
        );
        let role = convert::role_from_kv_role(ready[0].kv_role.as_deref()).ok_or_else(|| {
            anyhow::anyhow!(
                "vLLM OpenEngine does not support KV role `{}`",
                ready[0].kv_role.as_deref().unwrap_or_default()
            )
        })?;
        for response in ready.iter().skip(1) {
            let rank_role =
                convert::role_from_kv_role(response.kv_role.as_deref()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "vLLM OpenEngine does not support KV role `{}`",
                        response.kv_role.as_deref().unwrap_or_default()
                    )
                })?;
            anyhow::ensure!(
                rank_role == role,
                "OpenEngine requires a uniform KV role across local engine ranks"
            );
        }
        anyhow::ensure!(
            ready
                .iter()
                .all(|response| response.data_parallel_size == ready[0].data_parallel_size),
            "OpenEngine requires consistent data-parallel discovery across ranks"
        );
        if role != pb::EngineRole::Aggregated {
            anyhow::ensure!(
                ready.iter().all(|response| {
                    response
                        .kv_connector
                        .as_deref()
                        .is_some_and(|connector| connector.contains("Nixl"))
                }),
                "vLLM OpenEngine disaggregation requires NixlConnector"
            );
        }
        Ok(Self {
            state,
            instance_id: uuid::Uuid::new_v4().to_string(),
            advertise_host,
            role,
            active_connector_sessions: Arc::new(AtomicU64::new(0)),
            abort_cleanups: Arc::new(AbortCleanupTasks::new()),
        })
    }

    fn ready(&self) -> &EngineCoreReadyResponse {
        self.state.engine_core_client().ready_responses()[0]
    }

    fn supports_lora(&self) -> bool {
        self.state
            .engine_core_client()
            .ready_responses()
            .iter()
            .all(|response| response.supports_lora)
    }

    fn active_scheduler_baseline(&self) -> BTreeMap<u32, u64> {
        self.state
            .engine_core_client()
            .engine_loads()
            .into_iter()
            .filter(|load| {
                load.frontend_inflight > 0
                    || load.running_requests.saturating_add(load.queued_requests) > 0
            })
            .map(|load| (load.engine_index, load.update_generation))
            .collect()
    }

    async fn abort_and_track(&self, request_ids: &[String]) -> vllm_chat::Result<()> {
        let baseline = self.active_scheduler_baseline();
        let result = self.state.chat.abort(request_ids).await;
        self.abort_cleanups.track(self.state.clone(), baseline);
        result
    }

    pub(crate) async fn shutdown_background_tasks(&self) {
        self.abort_cleanups.shutdown().await;
    }

    fn kv_connector_info(&self) -> pb::KvConnectorInfo {
        let enabled = self.ready().kv_connector.is_some();
        pb::KvConnectorInfo {
            enabled: Some(enabled),
            transfer_backend: if enabled { TRANSFER_BACKEND } else { "" }.to_string(),
            local_endpoints: Vec::new(),
            supported_protocols: if enabled {
                vec![TRANSFER_BACKEND.to_string()]
            } else {
                Vec::new()
            },
            supports_remote_prefill: Some(enabled),
            supports_decode_pull: Some(enabled),
            supports_abort_cleanup: Some(enabled),
            schema_version: enabled.then_some(1),
        }
    }

    fn lora_adapter(request: &LoraRequest) -> pb::LoraAdapter {
        pb::LoraAdapter {
            lora_id: i64::try_from(request.lora_int_id).unwrap_or(i64::MAX),
            lora_name: request.lora_name.clone(),
            source_path: request.lora_path.clone(),
        }
    }

    async fn run_inference_probe(&self, model: &str) -> (pb::HealthState, String) {
        if !model.is_empty()
            && model != self.state.chat.model_id()
            && !self.state.served_model_names().iter().any(|name| name == model)
        {
            return (
                pb::HealthState::Degraded,
                format!("model `{model}` not found"),
            );
        }
        if self.role != pb::EngineRole::Aggregated {
            return (
                pb::HealthState::Ready,
                "engine-core readiness used because a disaggregated probe requires its peer"
                    .to_string(),
            );
        }
        let Some(_work_guard) = self.state.try_admit_engine_work() else {
            return (
                pb::HealthState::NotReady,
                "inference probe rejected by engine admission".to_string(),
            );
        };
        let request = TextRequest {
            request_id: format!("openengine-health-{}", uuid::Uuid::new_v4()),
            prompt: Prompt::Text("hi".to_string()),
            mm_features: None,
            sampling_params: SamplingParams {
                temperature: Some(0.0),
                max_tokens: Some(1),
                ..SamplingParams::default()
            },
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            priority: 0,
            cache_salt: None,
            add_special_tokens: true,
            data_parallel_rank: None,
            reasoning_parser_kwargs: None,
            lora_request: None,
            arrival_time: None,
        };
        let probe = async {
            let stream = self.state.chat.text().generate(request).await?;
            stream.collect_output().await
        };
        match tokio::time::timeout(INFERENCE_PROBE_TIMEOUT, probe).await {
            Ok(Ok(_)) => (pb::HealthState::Ready, String::new()),
            Ok(Err(error)) => (pb::HealthState::Degraded, error.to_report_string()),
            Err(_) => (
                pb::HealthState::Degraded,
                format!(
                    "inference probe timed out after {}s",
                    INFERENCE_PROBE_TIMEOUT.as_secs()
                ),
            ),
        }
    }
}

#[tonic::async_trait]
impl pb::inference_server::Inference for OpenEngineService {
    type GenerateStream = ResponseStream<pb::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let work_guard = self
            .state
            .try_admit_engine_work()
            .ok_or_else(|| Status::unavailable("engine is draining"))?;
        let (target_dp_rank, priority) = convert::metadata_options(&request)?;
        let proto_request = request.into_inner();
        let lora_name = proto_request.lora_name.clone();
        let handoff_dp_rank = target_dp_rank.unwrap_or(self.ready().data_parallel_rank);
        let mut text_request = convert::to_text_request(
            proto_request,
            self.role,
            target_dp_rank,
            priority,
            self.ready().data_parallel_size,
            &self.state,
        )
        .await?;

        let lora_lease = if lora_name.is_empty() {
            None
        } else {
            if !self.supports_lora() {
                return Err(Status::failed_precondition(
                    "vLLM was not started with dynamic LoRA support",
                ));
            }
            let (lora_request, lease) =
                self.state.activate_lora(&lora_name).await.map_err(activate_lora_status)?;
            text_request.lora_request = Some(lora_request);
            lease
        };

        let request_id = text_request.request_id.clone();
        info!(%request_id, role = ?self.role, "openengine generate");
        let stream = self
            .state
            .chat
            .text()
            .generate(text_request)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        let stream = crate::lora::hold_lora_lease(stream, lora_lease);
        let role = self.role;
        let connector_session = (role != pb::EngineRole::Aggregated)
            .then(|| ConnectorSessionGuard::new(self.active_connector_sessions.clone()));
        let (tx, rx) = mpsc::channel(32);
        let service = self.clone();
        tokio::spawn(async move {
            let _work_guard = work_guard;
            let _connector_session = connector_session;
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                let terminal_error = event.is_err();
                let responses = match event {
                    Ok(event) => {
                        convert::event_to_responses(event, &request_id, role, handoff_dp_rank)
                    }
                    Err(error) => vec![convert::engine_error(
                        &request_id,
                        pb::ErrorCode::Internal,
                        error.to_report_string(),
                        None,
                    )],
                };
                for response in responses {
                    if tx.send(Ok(response)).await.is_err() {
                        let _ = service.abort_and_track(std::slice::from_ref(&request_id)).await;
                        return;
                    }
                }
                if terminal_error {
                    return;
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for OpenEngineService {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        let ready = self.state.engine_core_client().ready_responses();
        let first = ready[0];
        let total_blocks = ready.iter().map(|response| response.num_gpu_blocks).sum();
        let max_requests = ready.iter().map(|response| response.max_num_seqs).sum();
        let max_batched_tokens = ready.iter().map(|response| response.max_num_batched_tokens).sum();
        let start_rank = ready.iter().map(|response| response.data_parallel_rank).min();
        Ok(Response::new(pb::ServerInfo {
            engine_name: "vllm".to_string(),
            engine_version: first.vllm_version.clone(),
            engine_role: self.role as i32,
            instance_id: self.instance_id.clone(),
            // This process hosts one canonical model. Public names beyond the
            // primary served name are aliases reported by ModelInfo, not
            // additional models that require explicit sidecar selection.
            supported_models: vec![self.state.chat.model_id().to_string()],
            parallelism: Some(pb::ParallelismInfo {
                tensor_parallel_size: Some(first.tensor_parallel_size),
                pipeline_parallel_size: Some(first.pipeline_parallel_size),
                data_parallel_size: Some(first.data_parallel_size.min(u64::from(u32::MAX)) as u32),
                data_parallel_rank: (ready.len() == 1).then_some(first.data_parallel_rank),
                data_parallel_start_rank: start_rank,
                decode_context_parallel_size: Some(1),
            }),
            kv_connector: Some(self.kv_connector_info()),
            schema_revision: SCHEMA_REVISION,
            minimum_client_revision: MINIMUM_CLIENT_REVISION,
            schema_release: SCHEMA_RELEASE.to_string(),
            capacity: Some(pb::DeploymentCapacity {
                kv_block_size: Some(first.block_size.min(u64::from(u32::MAX)) as u32),
                total_kv_blocks: Some(total_blocks),
                max_running_requests: Some(max_requests),
                max_batched_tokens: Some(max_batched_tokens),
                max_loras: self.supports_lora().then_some(first.max_loras),
            }),
            extra: None,
        }))
    }

    async fn get_model_info(
        &self,
        request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let requested = request.into_inner().model;
        if !requested.is_empty()
            && requested != self.state.chat.model_id()
            && !self.state.served_model_names().iter().any(|model| model == &requested)
        {
            return Err(Status::not_found(format!("model `{requested}` not found")));
        }
        let supports_image =
            self.state.chat.supported_modalities().contains(&vllm_chat::Modality::Image);
        let max_context = self
            .state
            .engine_core_client()
            .ready_responses()
            .iter()
            .map(|response| response.max_model_len)
            .min()
            .unwrap_or_default()
            .min(u64::from(u32::MAX)) as u32;
        let logprob_modes = vec![
            pb::CandidateTokenSelectionMode::TopN as i32,
            pb::CandidateTokenSelectionMode::All as i32,
        ];
        let mut output_logprob_modes = logprob_modes.clone();
        output_logprob_modes.push(pb::CandidateTokenSelectionMode::TokenIds as i32);
        Ok(Response::new(pb::ModelInfo {
            model_id: self.state.chat.model_id().to_string(),
            served_model_name: self.state.primary_model_name().to_string(),
            served_model_aliases: self.state.served_model_names().iter().skip(1).cloned().collect(),
            max_context_length: Some(max_context),
            max_output_tokens: Some(max_context),
            tokenizer_modes: vec![self.state.tokenizer_mode().to_string()],
            supports_text_input: Some(true),
            supports_token_ids_input: Some(true),
            generation: Some(pb::GenerationCapabilities {
                prompt_logprobs: Some(pb::LogprobCapabilities {
                    supported: Some(true),
                    candidate_selection_modes: logprob_modes,
                    max_top_n: None,
                }),
                output_logprobs: Some(pb::LogprobCapabilities {
                    supported: Some(true),
                    candidate_selection_modes: output_logprob_modes,
                    max_top_n: None,
                }),
                guided_decoding: Some(pb::GuidedDecodingCapabilities {
                    supported: Some(true),
                    modes: vec![
                        pb::GuidedDecodingMode::JsonSchema as i32,
                        pb::GuidedDecodingMode::Regex as i32,
                        pb::GuidedDecodingMode::EbnfGrammar as i32,
                        pb::GuidedDecodingMode::StructuralTag as i32,
                        pb::GuidedDecodingMode::Choice as i32,
                        pb::GuidedDecodingMode::JsonObject as i32,
                    ],
                }),
                max_num_sequences: Some(1),
                supports_priority: Some(true),
                supports_stop_in_output: Some(true),
                supports_cache_salt: Some(true),
                supports_prefix_cache_bypass: Some(true),
            }),
            supports_lora: Some(self.supports_lora()),
            supports_multimodal: Some(supports_image),
            reasoning_parser: self
                .state
                .chat
                .reasoning_parser_name()
                .unwrap_or_default()
                .to_string(),
            tool_call_parser: self
                .state
                .chat
                .tool_call_parser_name()
                .unwrap_or_default()
                .to_string(),
            extra: None,
        }))
    }

    async fn get_load(
        &self,
        request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        let include_per_rank = request.into_inner().include_per_rank;
        let client = self.state.engine_core_client();
        let ready = client.ready_responses();
        let total_blocks = ready.iter().map(|response| response.num_gpu_blocks).sum();
        let engine_loads_by_index = client
            .engine_loads()
            .into_iter()
            .map(|load| (load.engine_index, load))
            .collect::<BTreeMap<_, _>>();
        let engine_loads = client
            .engine_indices()
            .into_iter()
            .zip(ready.iter())
            .filter_map(|(engine_index, response)| {
                engine_loads_by_index
                    .get(&engine_index)
                    .copied()
                    .map(|load| (response.data_parallel_rank, load))
            })
            .collect::<BTreeMap<_, _>>();
        let scheduler_complete = ready.iter().all(|response| {
            engine_loads
                .get(&response.data_parallel_rank)
                .is_some_and(|load| load.kv_cache_usage.is_some())
        });
        let used_blocks = ready
            .iter()
            .map(|response| {
                engine_loads
                    .get(&response.data_parallel_rank)
                    .and_then(|load| used_kv_blocks(response.num_gpu_blocks, load.kv_cache_usage))
            })
            .collect::<Option<Vec<_>>>()
            .map(|blocks| blocks.into_iter().sum());
        let load = self.state.server_load().min(u64::from(u32::MAX)) as u32;
        let pending_cleanup = self.abort_cleanups.pending();
        let running_requests = if scheduler_complete {
            engine_loads
                .values()
                .map(|load| load.running_requests)
                .sum::<u64>()
                .min(u64::from(u32::MAX)) as u32
        } else {
            load
        }
        .max(load)
        .max(pending_cleanup.min(u64::from(u32::MAX)) as u32);
        let queued_requests = scheduler_complete.then(|| {
            engine_loads
                .values()
                .map(|load| load.queued_requests)
                .sum::<u64>()
                .min(u64::from(u32::MAX)) as u32
        });
        let ranks = if include_per_rank {
            ready
                .iter()
                .map(|response| {
                    let rank_load = engine_loads.get(&response.data_parallel_rank);
                    pb::RankLoadInfo {
                        data_parallel_rank: Some(response.data_parallel_rank),
                        running_requests: rank_load
                            .map(|load| load.running_requests.min(u64::from(u32::MAX)) as u32),
                        queued_requests: rank_load
                            .map(|load| load.queued_requests.min(u64::from(u32::MAX)) as u32),
                        used_kv_blocks: rank_load.and_then(|load| {
                            used_kv_blocks(response.num_gpu_blocks, load.kv_cache_usage)
                        }),
                        total_kv_blocks: Some(response.num_gpu_blocks),
                        prefill_batch_size: None,
                        decode_batch_size: None,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Response::new(pb::LoadInfo {
            instance_id: self.instance_id.clone(),
            timestamp_unix_nanos: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .and_then(|duration| u64::try_from(duration.as_nanos()).ok()),
            running_requests: Some(running_requests),
            queued_requests,
            // This is the number of OpenEngine disaggregated streams whose
            // scheduler cleanup is not yet acknowledged, not a lifetime NIXL
            // transfer-session inventory.
            active_kv_sessions: Some(if self.role == pb::EngineRole::Aggregated {
                0
            } else {
                self.active_connector_sessions
                    .load(Ordering::SeqCst)
                    .saturating_add(pending_cleanup)
                    .min(u64::from(u32::MAX)) as u32
            }),
            used_kv_blocks: used_blocks,
            total_kv_blocks: Some(total_blocks),
            running_tokens: None,
            waiting_tokens: None,
            prefill_batch_size: None,
            decode_batch_size: None,
            ranks,
            attributes: None,
        }))
    }

    async fn health(
        &self,
        request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        let request = request.into_inner();
        let expected_role = pb::EngineRole::try_from(request.role).unwrap_or_default();
        let role_matches =
            expected_role == pb::EngineRole::Unspecified || expected_role == self.role;
        let engine_healthy = self.state.engine_core_client().is_healthy();
        let mut state = if !engine_healthy || !role_matches {
            pb::HealthState::NotReady
        } else {
            pb::HealthState::Ready
        };
        let mut checks = vec![
            pb::HealthCheck {
                name: "engine".to_string(),
                state: if engine_healthy {
                    pb::HealthState::Ready as i32
                } else {
                    pb::HealthState::NotReady as i32
                },
                message: self
                    .state
                    .engine_core_client()
                    .health_error()
                    .map(|error| error.to_string())
                    .unwrap_or_default(),
            },
            pb::HealthCheck {
                name: "role".to_string(),
                state: if role_matches {
                    pb::HealthState::Ready as i32
                } else {
                    pb::HealthState::NotReady as i32
                },
                message: if role_matches {
                    String::new()
                } else {
                    format!("expected {expected_role:?}, serving {:?}", self.role)
                },
            },
        ];
        if request.include_inference_probe && state == pb::HealthState::Ready {
            let (probe_state, message) = self.run_inference_probe(&request.model).await;
            checks.push(pb::HealthCheck {
                name: "inference_probe".to_string(),
                state: probe_state as i32,
                message,
            });
            if probe_state != pb::HealthState::Ready {
                state = probe_state;
            }
        }
        Ok(Response::new(pb::HealthResponse {
            state: state as i32,
            checks,
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        use pb::abort_request::Target;
        let ids = match request.into_inner().target {
            Some(Target::RequestId(request_id)) if !request_id.is_empty() => vec![request_id],
            Some(Target::KvSession(session)) if !session.session_id.is_empty() => {
                vec![session.session_id]
            }
            Some(Target::AllRequests(_)) => Vec::new(),
            _ => return Err(Status::invalid_argument("abort target is required")),
        };
        self.abort_and_track(&ids)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        Ok(Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: String::new(),
        }))
    }

    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        if !self.supports_lora() {
            return Err(Status::failed_precondition(
                "vLLM was not started with dynamic LoRA support",
            ));
        }
        let adapter = request
            .into_inner()
            .adapter
            .ok_or_else(|| Status::invalid_argument("adapter is required"))?;
        if adapter.lora_id <= 0
            || adapter.lora_name.trim().is_empty()
            || adapter.source_path.is_empty()
        {
            return Err(Status::invalid_argument(
                "adapter id, name, and source_path are required; id must be positive",
            ));
        }
        let source_path = std::fs::canonicalize(&adapter.source_path).map_err(|error| {
            Status::invalid_argument(format!(
                "adapter source_path `{}` is not accessible: {error}",
                adapter.source_path
            ))
        })?;
        if !source_path.is_dir() {
            return Err(Status::invalid_argument(
                "adapter source_path must be a directory",
            ));
        }
        validate_lora_source(&source_path)?;
        let lora_request = LoraRequest::new(
            adapter.lora_name,
            adapter.lora_id as u64,
            source_path.to_string_lossy().into_owned(),
            false,
            false,
        );
        let (registered, already_loaded) =
            self.state.register_lora_exact(lora_request).await.map_err(load_exact_status)?;
        Ok(Response::new(pb::LoadLoraResponse {
            adapter: Some(Self::lora_adapter(&registered)),
            already_loaded,
        }))
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        let lora_name = request.into_inner().lora_name;
        if lora_name.is_empty() {
            return Err(Status::invalid_argument("lora_name is required"));
        }
        let removed = self.state.logical_unload_lora(&lora_name).await.map_err(unload_status)?;
        Ok(Response::new(pb::UnloadLoraResponse {
            adapter: Some(Self::lora_adapter(&removed)),
        }))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        Ok(Response::new(pb::ListLorasResponse {
            adapters: self
                .state
                .served_lora_requests()
                .await
                .iter()
                .map(Self::lora_adapter)
                .collect(),
        }))
    }

    async fn get_kv_event_sources(
        &self,
        request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let requested = request.into_inner().data_parallel_ranks;
        let sources = self
            .state
            .engine_core_client()
            .ready_responses()
            .into_iter()
            .filter(|response| {
                requested.is_empty() || requested.contains(&response.data_parallel_rank)
            })
            .filter(|response| response.kv_events_publisher.as_deref() == Some("zmq"))
            .filter_map(|response| self.kv_event_source(response))
            .collect();
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
    }

    type SubscribeKvEventsStream = ResponseStream<pb::SubscribeKvEventsResponse>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented(
            "vLLM advertises direct ZMQ/msgpack KV event sources",
        ))
    }
}

impl OpenEngineService {
    fn kv_event_source(&self, ready: &EngineCoreReadyResponse) -> Option<pb::KvEventSource> {
        let endpoint = offset_endpoint_port(
            ready.kv_events_endpoint.as_deref()?,
            ready.data_parallel_rank,
        );
        let endpoint_addr = parse_zmq_endpoint(&endpoint, &self.advertise_host)?;
        let replay_endpoint = ready
            .kv_events_replay_endpoint
            .as_deref()
            .map(|endpoint| offset_endpoint_port(endpoint, ready.data_parallel_rank))
            .and_then(|endpoint| connectable_zmq_uri(&endpoint, &self.advertise_host))
            .unwrap_or_default();
        Some(pb::KvEventSource {
            transport: "zmq".to_string(),
            endpoint_addr: Some(endpoint_addr),
            topic: ready.kv_events_topic.clone().unwrap_or_default(),
            replay_endpoint,
            data_parallel_rank: Some(ready.data_parallel_rank),
            encoding: "msgpack".to_string(),
            schema_version: Some(1),
            buffer_steps: Some(ready.kv_events_buffer_steps.min(u64::from(u32::MAX)) as u32),
            hwm: Some(ready.kv_events_hwm.min(u64::from(u32::MAX)) as u32),
            max_queue_size: Some(ready.kv_events_max_queue_size.min(u64::from(u32::MAX)) as u32),
        })
    }
}

fn used_kv_blocks(total_blocks: u64, usage: Option<f64>) -> Option<u64> {
    let usage = usage?;
    if !usage.is_finite() || usage < 0.0 {
        return None;
    }
    Some((usage.min(1.0) * total_blocks as f64).round() as u64)
}

fn validate_lora_source(source_path: &Path) -> Result<(), Status> {
    let config_path = source_path.join("adapter_config.json");
    let config = std::fs::read(&config_path).map_err(|error| {
        Status::invalid_argument(format!(
            "LoRA directory is missing readable adapter_config.json at `{}`: {error}",
            config_path.display()
        ))
    })?;
    let config: serde_json::Value = serde_json::from_slice(&config).map_err(|error| {
        Status::invalid_argument(format!(
            "LoRA adapter_config.json at `{}` is invalid: {error}",
            config_path.display()
        ))
    })?;
    if !config.is_object() {
        return Err(Status::invalid_argument(format!(
            "LoRA adapter_config.json at `{}` must contain a JSON object",
            config_path.display()
        )));
    }

    let has_weights = ["adapter_model.safetensors", "adapter_model.bin"]
        .into_iter()
        .any(|name| source_path.join(name).is_file());
    if !has_weights {
        return Err(Status::invalid_argument(format!(
            "LoRA directory `{}` is missing adapter_model.safetensors or adapter_model.bin",
            source_path.display()
        )));
    }
    Ok(())
}

fn activate_lora_status(error: ActivateLoraError) -> Status {
    match error {
        ActivateLoraError::NotFound { lora_name } => {
            Status::not_found(format!("LoRA adapter `{lora_name}` is not registered"))
        }
        ActivateLoraError::Engine(error) => Status::internal(error.to_report_string()),
        ActivateLoraError::NotLoaded { lora_name } => {
            Status::internal(format!("vLLM rejected LoRA adapter `{lora_name}`"))
        }
    }
}

fn load_exact_status(error: LoadExactLoraError) -> Status {
    match error {
        LoadExactLoraError::BaseModelName { lora_name } => Status::already_exists(format!(
            "LoRA adapter name `{lora_name}` conflicts with a base model name"
        )),
        LoadExactLoraError::Conflict { existing } => Status::already_exists(format!(
            "LoRA identity conflicts with existing adapter `{}` (id {}, path `{}`)",
            existing.lora_name, existing.lora_int_id, existing.lora_path
        )),
    }
}

fn unload_status(error: UnloadLoraError) -> Status {
    match error {
        UnloadLoraError::NotFound { lora_name } => {
            Status::not_found(format!("LoRA adapter `{lora_name}` is not registered"))
        }
        UnloadLoraError::IntIdMismatch { .. }
        | UnloadLoraError::Engine(_)
        | UnloadLoraError::NotRemoved { .. } => Status::internal(format!("{error:?}")),
    }
}

fn offset_endpoint_port(endpoint: &str, rank: u32) -> String {
    if rank == 0 || endpoint.is_empty() {
        return endpoint.to_string();
    }
    if endpoint.starts_with("inproc://") {
        return format!("{endpoint}_dp{rank}");
    }
    if endpoint.starts_with("tcp://")
        && let Some((prefix, port)) = endpoint.rsplit_once(':')
        && let Ok(port) = port.parse::<u32>()
    {
        return format!("{prefix}:{}", port.saturating_add(rank));
    }
    endpoint.to_string()
}

fn parse_zmq_endpoint(endpoint: &str, advertise_host: &str) -> Option<pb::KvEndpoint> {
    let endpoint = endpoint.strip_prefix("tcp://")?;
    let (host, port) = endpoint.rsplit_once(':')?;
    let host = host.trim_matches(|character| character == '[' || character == ']');
    let host = match host {
        "" | "*" | "0.0.0.0" | "::" => routable_host(advertise_host),
        host => host.to_string(),
    };
    Some(pb::KvEndpoint {
        host,
        port: port.parse().ok()?,
        protocol: "tcp".to_string(),
    })
}

fn connectable_zmq_uri(endpoint: &str, advertise_host: &str) -> Option<String> {
    let endpoint = parse_zmq_endpoint(endpoint, advertise_host)?;
    let host = if endpoint.host.contains(':') {
        format!("[{}]", endpoint.host)
    } else {
        endpoint.host
    };
    Some(format!("{}://{host}:{}", endpoint.protocol, endpoint.port))
}

fn routable_host(configured: &str) -> String {
    if !matches!(configured, "" | "*" | "0.0.0.0" | "::") {
        return configured.to_string();
    }
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|socket| {
            socket.connect("10.255.255.255:1")?;
            Ok(socket.local_addr()?.ip().to_string())
        })
        .unwrap_or_else(|_| "127.0.0.1".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_rank_event_port_matches_vllm_publisher() {
        assert_eq!(offset_endpoint_port("tcp://*:5557", 0), "tcp://*:5557");
        assert_eq!(offset_endpoint_port("tcp://*:5557", 2), "tcp://*:5559");
    }

    #[test]
    fn wildcard_event_endpoint_is_connectable() {
        let endpoint = parse_zmq_endpoint("tcp://*:5557", "10.1.2.3").unwrap();
        assert_eq!(endpoint.host, "10.1.2.3");
        assert_eq!(endpoint.port, 5557);
    }
}
