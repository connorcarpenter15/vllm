//! Private engine RPC service backed by the shared [`vllm_text::TextLlm`]
//! facade.
//!
//! This API is consumed by out-of-process vLLM frontends. It is a sibling of
//! [`crate::grpc::GenerateServiceImpl`] and
//! is backed by the same [`AppState`]. Topology, role, and limits are sourced
//! from the engine startup handshake
//! ([`vllm_engine_core_client::EngineCoreClient::ready_response`]) rather than
//! from CLI flags, so the sidecar discovers everything over the wire.

mod convert;
mod lora;

use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;
use uuid::Uuid;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextOutputStreamExt as _, TextRequest};

use crate::state::AppState;

/// Generated protobuf/gRPC types for the `vllm.engine.v1` package.
pub mod pb {
    tonic::include_proto!("vllm.engine.v1");
}

pub use pb::engine_server::EngineServer;

#[cfg(test)]
mod tests;

/// API version string advertised by [`EngineServiceImpl::get_engine_info`].
const ENGINE_RPC_API_VERSION: &str = "vllm.engine.v1";
const INFERENCE_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Engine RPC service implementation backed by the shared application state.
pub struct EngineServiceImpl {
    state: Arc<AppState>,
}

impl EngineServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    /// The engine's startup-handshake response, the source of all discovery
    /// metadata advertised over the engine RPC API.
    fn ready(&self) -> &EngineCoreReadyResponse {
        self.state.engine_core_client().ready_response()
    }

    /// Conservative per-engine KV-cache block capacity advertised to the frontend.
    fn per_rank_kv_blocks(&self) -> u64 {
        per_rank_kv_blocks(&self.state.engine_core_client().ready_responses())
    }
}

/// Each ready response reports one connected engine's own cache. Use the
/// minimum so heterogeneous ranks never over-advertise capacity. This remains
/// correct when the frontend owns only a subset of the global DP ranks.
fn per_rank_kv_blocks(ready: &[&EngineCoreReadyResponse]) -> u64 {
    ready.iter().map(|response| response.num_gpu_blocks).min().unwrap_or(0)
}

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl pb::engine_server::Engine for EngineServiceImpl {
    type GenerateStream = ResponseStream<pb::GenerateResponse>;

    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        lora::load_lora(&self.state, request).await
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        lora::unload_lora(&self.state, request).await
    }

    async fn list_loras(
        &self,
        request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        lora::list_loras(&self.state, request).await
    }

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let load_guard = self
            .state
            .try_admit_engine_work()
            .ok_or_else(|| Status::unavailable("engine is draining"))?;

        let proto_req = request.into_inner();
        let role = convert::role_from_kv_role(self.ready().kv_role.as_deref());
        let handoff_dp_rank = convert::validate_disaggregated_request(
            &proto_req,
            role,
            self.ready().kv_connector.as_deref(),
            self.ready().data_parallel_size,
            self.ready().data_parallel_rank,
        )?;
        let lora_name = proto_req.lora_name.clone();
        // Extract media before moving the request into text conversion; wire
        // order aligns with the placeholder markers carried in the prompt.
        let media_parts = convert::media_parts_from_request(&proto_req.media)?;
        let mut text_request =
            convert::to_text_request(proto_req, self.state.served_model_names())?;

        let mut lora_lease = None;
        if !lora_name.is_empty() {
            if !self.ready().supports_lora {
                return Err(Status::failed_precondition(
                    "engine was not started with LoRA enabled",
                ));
            }
            let mut resolution = self.state.resolve_model_with_loras(Some(&lora_name)).await;
            lora_lease = resolution.lease.take();
            // Resolve first so an in-flight mutation cannot become inconsistent
            // between this check and generation. The shared lease remains held
            // through the response stream.
            if !self.state.lora_state_is_consistent() {
                return Err(Status::failed_precondition(
                    "LoRA state differs across engine ranks; restart the engine",
                ));
            }
            text_request.lora_request = Some(resolution.lora_request.ok_or_else(|| {
                Status::not_found(format!("LoRA adapter `{lora_name}` is not loaded"))
            })?);
        }

        let request_id = text_request.request_id.clone();
        info!(%request_id, "engine_rpc generate");

        // Multimodal: the orchestrator forwards media plus token IDs that still
        // carry un-expanded placeholder markers (one per item). Fetch and
        // preprocess the media, expand the markers in place, and attach the
        // engine-facing features. This runs before `mark_prefill_request` so the
        // prefill engine encodes the media and produces the KV the decode peer
        // pulls.
        if !media_parts.is_empty() {
            let Prompt::TokenIds(mut token_ids) = text_request.prompt else {
                return Err(Status::invalid_argument(
                    "multimodal engine RPC requests must provide token_ids input; \
                     placeholder markers are expanded engine-side",
                ));
            };
            let mm_features = self
                .state
                .chat
                .prepare_media(media_parts, &mut token_ids)
                .await
                .map_err(|e| Status::internal(e.to_report_string()))?;
            text_request.prompt = Prompt::TokenIds(token_ids);
            text_request.mm_features = mm_features;
        }

        // Role and connector are uniform across engines; snapshot owned copies
        // so the response mapping can run inside the spawned task.
        let kv_connector = self.ready().kv_connector.clone();

        // Prefill role: NixlConnector only retains the KV blocks and reports the
        // handoff metadata (`remote_block_ids` / `remote_engine_id` /
        // `remote_host` / `remote_port`) when the request carries
        // `do_remote_decode`. The engine is authoritative about its own role, so
        // inject it here — the sidecar stays role-agnostic.
        if role == pb::EngineRole::Prefill {
            convert::mark_prefill_request(&mut text_request);
        }

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(|e| Status::internal(e.to_report_string()))?;
        let stream = crate::lora::hold_lora_lease(stream, lora_lease);

        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            let _load_guard = load_guard;
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                let responses = match event {
                    Ok(event) => convert::event_to_responses(
                        event,
                        &request_id,
                        role,
                        kv_connector.as_deref(),
                        handoff_dp_rank,
                    ),
                    Err(e) => {
                        // Surface a mid-stream failure as a structured terminal
                        // error event, then stop.
                        let resp = convert::error_response(&request_id, e.to_report_string());
                        let _ = tx.send(Ok(resp)).await;
                        break;
                    }
                };
                for response in responses {
                    if tx.send(Ok(response)).await.is_err() {
                        // Client disconnected; dropping the stream aborts the
                        // engine-core request.
                        return;
                    }
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn get_engine_info(
        &self,
        _request: Request<pb::GetEngineInfoRequest>,
    ) -> Result<Response<pb::EngineInfo>, Status> {
        let rr = self.ready();
        let role = convert::role_from_kv_role(rr.kv_role.as_deref());

        Ok(Response::new(pb::EngineInfo {
            engine_name: "vllm".to_string(),
            engine_version: rr.vllm_version.clone(),
            api_version: ENGINE_RPC_API_VERSION.to_string(),
            role: role as i32,
            instance_id: rr.kv_engine_id.clone().unwrap_or_default(),
            supported_models: self.state.served_model_names().to_vec(),
            parallelism: Some(self.parallelism_info()),
            kv_connector: Some(self.kv_connector_info()),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let client = self.state.engine_core_client();
        let rr = self.ready();
        let served = self.state.served_model_names();

        Ok(Response::new(pb::ModelInfo {
            model_id: self.state.chat.text().model_id().to_string(),
            served_model_name: self.state.primary_model_name().to_string(),
            served_model_aliases: served.iter().skip(1).cloned().collect(),
            max_context_length: client.max_model_len(),
            max_output_tokens: 0,
            kv_block_size: rr.block_size.min(u64::from(u32::MAX)) as u32,
            total_kv_blocks: self.per_rank_kv_blocks(),
            max_running_requests: rr.max_num_seqs,
            max_batched_tokens: rr.max_num_batched_tokens,
            tokenizer_modes: Vec::new(),
            max_loras: rr.max_loras,
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_lora: rr.supports_lora,
            supports_multimodal: self.state.chat.supports_multimodal(),
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
        }))
    }

    async fn health(
        &self,
        request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        let req = request.into_inner();
        let client = self.state.engine_core_client();

        let mut checks = Vec::new();
        let engine_state = if !client.is_healthy() {
            pb::HealthState::NotReady
        } else if self.state.engine_is_draining() {
            pb::HealthState::Draining
        } else {
            pb::HealthState::Ready
        };
        checks.push(health_check(
            "engine",
            engine_state,
            client.health_error().map(|e| e.to_string()),
        ));

        let mut overall = engine_state;
        if self.ready().supports_lora && !self.state.lora_state_is_consistent() {
            checks.push(health_check(
                "lora",
                pb::HealthState::NotReady,
                Some("adapter state differs across engine ranks; restart required".to_string()),
            ));
            overall = pb::HealthState::NotReady;
        }

        if req.include_inference_probe && overall == pb::HealthState::Ready {
            if let Some(_guard) = self.state.try_admit_engine_work() {
                let (probe_state, message) = self.run_inference_probe(&req.model).await;
                if probe_state != pb::HealthState::Ready {
                    overall = pb::HealthState::Degraded;
                }
                checks.push(health_check("inference_probe", probe_state, message));
            } else {
                overall = pb::HealthState::Draining;
            }
        }

        Ok(Response::new(pb::HealthResponse {
            state: overall as i32,
            checks,
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let req = request.into_inner();

        if req.abort_all {
            // The frontend does not retain the full set of in-flight IDs, so
            // bulk abort is not supported over this contract.
            return Ok(Response::new(pb::AbortResponse {
                status: pb::AbortStatus::Unsupported as i32,
                message: "abort_all is not supported".to_string(),
            }));
        }

        if req.request_id.is_empty() {
            return Err(Status::invalid_argument("request_id is required"));
        }

        // `abort` is a no-op for requests that are not in-flight, so this is
        // idempotent; we always report ABORTED on success.
        self.state
            .engine_core_client()
            .abort(std::slice::from_ref(&req.request_id))
            .await
            .map_err(|e| Status::internal(e.to_report_string()))?;

        Ok(Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: String::new(),
        }))
    }

    async fn drain(
        &self,
        _request: Request<pb::DrainRequest>,
    ) -> Result<Response<pb::DrainResponse>, Status> {
        self.state.begin_engine_drain();
        let in_flight = self.state.server_load().min(u64::from(u32::MAX)) as u32;
        let drain_state = if in_flight == 0 {
            pb::DrainState::Complete
        } else {
            pb::DrainState::InProgress
        };
        Ok(Response::new(pb::DrainResponse {
            state: drain_state as i32,
            in_flight_requests: in_flight,
            message: String::new(),
        }))
    }

    async fn get_kv_connector_info(
        &self,
        _request: Request<pb::GetKvConnectorInfoRequest>,
    ) -> Result<Response<pb::KvConnectorInfo>, Status> {
        Ok(Response::new(self.kv_connector_info()))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let client = self.state.engine_core_client();
        let responses = client.ready_responses();
        let ranked = responses
            .into_iter()
            .map(|response| (response.data_parallel_rank, response))
            .collect::<Vec<_>>();
        let sources = build_kv_event_sources(&ranked);
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
    }
}

impl EngineServiceImpl {
    fn parallelism_info(&self) -> pb::ParallelismInfo {
        let rr = self.ready();
        pb::ParallelismInfo {
            tensor_parallel_size: rr.tensor_parallel_size,
            pipeline_parallel_size: rr.pipeline_parallel_size,
            data_parallel_size: rr.data_parallel_size.min(u64::from(u32::MAX)) as u32,
            data_parallel_rank: rr.data_parallel_rank,
            data_parallel_start_rank: rr.data_parallel_rank,
        }
    }

    fn kv_connector_info(&self) -> pb::KvConnectorInfo {
        let rr = self.ready();
        let enabled = rr.kv_connector.is_some();
        pb::KvConnectorInfo {
            enabled,
            transfer_backend: rr.kv_connector.clone().unwrap_or_default(),
            local_endpoints: Vec::new(),
            supported_protocols: Vec::new(),
            schema_version: 1,
        }
    }

    /// Run a bounded single-token generation as a liveness probe.
    async fn run_inference_probe(&self, model: &str) -> (pb::HealthState, Option<String>) {
        if !model.is_empty() && !self.state.served_model_names().iter().any(|n| n == model) {
            return (
                pb::HealthState::Degraded,
                Some(format!("model `{model}` not found")),
            );
        }

        let probe = TextRequest {
            request_id: format!("engine_rpc-health-{}", Uuid::new_v4()),
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
            let stream = self.state.chat.text().generate(probe).await?;
            stream.collect_output().await
        };
        match tokio::time::timeout(INFERENCE_PROBE_TIMEOUT, probe).await {
            Ok(Ok(_)) => (pb::HealthState::Ready, None),
            Ok(Err(error)) => (pb::HealthState::Degraded, Some(error.to_report_string())),
            Err(_) => (
                pb::HealthState::Degraded,
                Some(format!(
                    "inference probe timed out after {}s",
                    INFERENCE_PROBE_TIMEOUT.as_secs()
                )),
            ),
        }
    }
}

fn health_check(name: &str, state: pb::HealthState, message: Option<String>) -> pb::HealthCheck {
    pb::HealthCheck {
        name: name.to_string(),
        state: state as i32,
        message: message.unwrap_or_default(),
    }
}

/// Offset a KV-event publisher endpoint's port by the data-parallel rank,
/// mirroring vLLM's `ZmqEventPublisher.offset_endpoint_port`
/// (`vllm/distributed/kv_events.py`). The engine reports the base endpoint in
/// its handshake but binds the publisher on `base_port + data_parallel_rank`,
/// so a subscriber must apply the same offset to connect to the right rank.
///
/// - rank 0: returned unchanged (no offset).
/// - `inproc://...`: `_dp{rank}` suffix.
/// - `tcp://host:port`: port becomes `port + rank`.
/// - anything else: returned unchanged.
fn offset_endpoint_port(endpoint: &str, data_parallel_rank: u32) -> String {
    if data_parallel_rank == 0 || endpoint.is_empty() {
        return endpoint.to_string();
    }
    if endpoint.contains("inproc") {
        return format!("{endpoint}_dp{data_parallel_rank}");
    }
    if endpoint.contains("tcp")
        && let Some((base_addr, port)) = endpoint.rsplit_once(':')
        && let Ok(base_port) = port.parse::<u32>()
    {
        return format!("{base_addr}:{}", base_port + data_parallel_rank);
    }
    endpoint.to_string()
}

/// Build the KV-event sources advertised for KV-aware routing from the engines'
/// startup handshakes.
///
/// Each connected DP engine runs its own KV-event publisher, so emit
/// one source per engine. The engine reports the *base* `kv_events_endpoint`;
/// vLLM's `ZmqEventPublisher` offsets the port by `data_parallel_rank` at bind
/// time (`vllm/distributed/kv_events.py::offset_endpoint_port`), replicated
/// here.
///
/// Publishers advertise the bind form (`tcp://0.0.0.0:PORT`) and let
/// [`kv_endpoint_from_zmq`] rewrite the wildcard host to a routable address.
fn build_kv_event_sources(
    ready_responses: &[(u32, &EngineCoreReadyResponse)],
) -> Vec<pb::KvEventSource> {
    ready_responses
        .iter()
        .filter(|(_, rr)| rr.kv_events_publisher.as_deref() == Some("zmq"))
        .filter_map(|(rank, rr)| {
            let base = rr.kv_events_endpoint.as_ref()?;
            let endpoint = offset_endpoint_port(base, *rank);
            let endpoint_addr = kv_endpoint_from_zmq(&endpoint)?;
            Some(pb::KvEventSource {
                transport: "zmq".to_string(),
                endpoint_addr: Some(endpoint_addr),
                topic: rr.kv_events_topic.clone().unwrap_or_default(),
                replay_endpoint: String::new(),
                data_parallel_rank: *rank,
                encoding: "msgpack".to_string(),
                schema_version: 1,
                buffer_steps: 0,
                hwm: 0,
                max_queue_size: 0,
            })
        })
        .collect()
}

/// Parse a ZMQ publisher endpoint (`tcp://host:port`) into a connectable
/// [`pb::KvEndpoint`], substituting a routable host for bind wildcards
/// (`*` / `0.0.0.0` / `::`). The engine binds its KV-event publisher on a
/// wildcard, which a remote KV router on another node cannot dial; this
/// advertises the node's routable address instead. Returns `None` if the
/// endpoint cannot be parsed.
fn kv_endpoint_from_zmq(endpoint: &str) -> Option<pb::KvEndpoint> {
    let rest = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let (host, port) = rest.rsplit_once(':')?;
    let port: u32 = port.parse().ok()?;
    let host = match host.trim_matches(|c| c == '[' || c == ']') {
        "*" | "0.0.0.0" | "::" | "" => advertise_host(),
        concrete => concrete.to_string(),
    };
    Some(pb::KvEndpoint {
        host,
        port,
        protocol: "tcp".to_string(),
    })
}

/// Best-effort routable host for advertising a locally-bound socket. Discovers
/// the node's primary outbound IP via a connected UDP socket (no packets are
/// sent — `connect` only consults the routing table); falls back to loopback
/// when no route exists (single-host deployments).
fn advertise_host() -> String {
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|sock| {
            sock.connect("10.255.255.255:1")?;
            Ok(sock.local_addr()?.ip().to_string())
        })
        .unwrap_or_else(|_| "127.0.0.1".to_string())
}
