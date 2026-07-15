//! gRPC Generate service backed by the shared [`vllm_text::TextLlm`] facade.

mod convert;

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tonic_health::server::HealthReporter;
use tracing::info;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_text::{DecodedTextEvent, Prompt, TextOutputStreamExt as _, TextRequest};

use self::convert::ResponseOpts;
use crate::state::AppState;

/// Generated protobuf/gRPC types for the `vllm` package.
pub mod pb {
    tonic::include_proto!("vllm");
}

pub use pb::control_server::ControlServer;
pub use pb::generate_server::GenerateServer;

#[cfg(test)]
mod tests;

/// gRPC Generate service implementation backed by the shared application state.
#[derive(Clone)]
pub struct GenerateServiceImpl {
    state: Arc<AppState>,
    admission: Arc<AdmissionState>,
}

impl GenerateServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            admission: Arc::new(AdmissionState::default()),
        }
    }

    pub fn control_service(&self, health_reporter: Option<HealthReporter>) -> ControlServiceImpl {
        ControlServiceImpl {
            state: self.state.clone(),
            admission: self.admission.clone(),
            health_reporter,
        }
    }
}

#[derive(Default)]
struct AdmissionState {
    draining: AtomicBool,
    in_flight: AtomicU64,
}

struct AdmissionGuard(Arc<AdmissionState>);

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

impl GenerateServiceImpl {
    fn is_draining(&self) -> bool {
        self.admission.draining.load(Ordering::SeqCst)
    }

    fn try_admit(&self) -> Option<AdmissionGuard> {
        if self.is_draining() {
            return None;
        }
        self.admission.in_flight.fetch_add(1, Ordering::SeqCst);
        if self.is_draining() {
            self.admission.in_flight.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(AdmissionGuard(self.admission.clone()))
    }

    async fn prepare_request(
        &self,
        proto_request: pb::GenerateRequest,
        stream: bool,
    ) -> Result<TextRequest, Status> {
        if !proto_request.lora_name.is_empty() {
            return Err(Status::unimplemented("LoRA request selection"));
        }
        let media = convert::media_parts_from_request(&proto_request.media)?;
        let mut text_request =
            convert::to_text_request(proto_request, stream, self.state.served_model_names())?;

        if !media.is_empty() {
            let Prompt::TokenIds(mut token_ids) = text_request.prompt else {
                return Err(Status::invalid_argument(
                    "multimodal gRPC requests must provide token_ids input",
                ));
            };
            let mm_features = self
                .state
                .chat
                .prepare_media(media, &mut token_ids)
                .await
                .map_err(|error| Status::internal(error.to_report_string()))?;
            text_request.prompt = Prompt::TokenIds(token_ids);
            text_request.mm_features = mm_features;
        }

        Ok(text_request)
    }
}

const GRPC_API_VERSION: &str = "vllm";

pub struct ControlServiceImpl {
    state: Arc<AppState>,
    admission: Arc<AdmissionState>,
    health_reporter: Option<HealthReporter>,
}

impl ControlServiceImpl {
    fn ready(&self) -> &EngineCoreReadyResponse {
        self.state.engine_core_client().ready_response()
    }

    fn per_rank_kv_blocks(&self) -> u64 {
        self.state
            .engine_core_client()
            .ready_responses()
            .into_iter()
            .map(|response| response.num_gpu_blocks)
            .min()
            .unwrap_or(0)
    }

    fn parallelism_info(&self) -> pb::ParallelismInfo {
        let ready = self.ready();
        pb::ParallelismInfo {
            tensor_parallel_size: ready.tensor_parallel_size,
            pipeline_parallel_size: ready.pipeline_parallel_size,
            data_parallel_size: ready.data_parallel_size.min(u64::from(u32::MAX)) as u32,
            data_parallel_rank: ready.data_parallel_rank,
            data_parallel_start_rank: ready.data_parallel_rank,
            decode_context_parallel_size: ready.decode_context_parallel_size,
        }
    }

    fn begin_drain(&self) {
        self.admission.draining.store(true, Ordering::SeqCst);
    }

    async fn report_not_serving(&self) {
        if let Some(reporter) = &self.health_reporter {
            crate::set_generate_not_serving(reporter).await;
        }
    }

    fn in_flight(&self) -> u64 {
        self.admission.in_flight.load(Ordering::SeqCst)
    }
}

#[tonic::async_trait]
impl pb::generate_server::Generate for GenerateServiceImpl {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    /// Unary generate: collect all output and return a single response.
    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let _guard = self
            .try_admit()
            .ok_or_else(|| Status::unavailable("gRPC service is draining"))?;
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let text_request = self.prepare_request(proto_req, false).await?;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (unary)");

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(text_error_to_status)?;

        let collected = stream.collect_output().await.map_err(text_error_to_status)?;

        // Build the single aggregated response.
        let prompt_info = convert::to_prompt_info(
            &collected.prompt_token_ids,
            collected.prompt_logprobs.as_ref(),
            &response_opts,
        );

        let finish_info = vllm_text::Finished {
            usage: collected.usage,
            finish_reason: collected.finish_reason,
            kv_transfer_params: collected.kv_transfer_params,
        };

        let outputs = convert::to_sequence_output(
            &collected.text,
            &collected.token_ids,
            collected.logprobs.as_ref(),
            Some(&finish_info),
            &response_opts,
        );

        Ok(Response::new(pb::GenerateResponse {
            prompt_info: Some(prompt_info),
            outputs: Some(outputs),
        }))
    }

    /// Streaming generate: yield incremental responses as tokens are produced.
    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let guard = self
            .try_admit()
            .ok_or_else(|| Status::unavailable("gRPC service is draining"))?;
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let text_request = self.prepare_request(proto_req, true).await?;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (stream)");

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(text_error_to_status)?;

        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            let _guard = guard;
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                let response = match event {
                    Err(e) => Err(text_error_to_status(e)),
                    Ok(DecodedTextEvent::Start {
                        prompt_token_ids,
                        prompt_logprobs,
                    }) => {
                        let prompt_info = convert::to_prompt_info(
                            &prompt_token_ids,
                            prompt_logprobs.as_ref(),
                            &response_opts,
                        );
                        Ok(pb::GenerateResponse {
                            prompt_info: Some(prompt_info),
                            outputs: None,
                        })
                    }
                    Ok(DecodedTextEvent::TextDelta {
                        delta,
                        token_ids,
                        logprobs,
                        finished,
                    }) => Ok(pb::GenerateResponse {
                        prompt_info: None,
                        outputs: Some(convert::to_sequence_output(
                            &delta,
                            &token_ids,
                            logprobs.as_ref(),
                            finished.as_ref(),
                            &response_opts,
                        )),
                    }),
                };

                if tx.send(response).await.is_err() {
                    // Client disconnected.
                    break;
                }
            }
        });

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for ControlServiceImpl {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        let ready = self.ready();
        Ok(Response::new(pb::ServerInfo {
            engine_version: ready.vllm_version.clone(),
            api_version: GRPC_API_VERSION.to_string(),
            instance_id: ready.instance_id.clone(),
            parallelism: Some(self.parallelism_info()),
            max_model_len: ready.max_model_len.min(u64::from(u32::MAX)) as u32,
            kv_block_size: ready.block_size.min(u64::from(u32::MAX)) as u32,
            total_kv_blocks: self.per_rank_kv_blocks(),
            max_running_requests: ready.max_num_seqs,
            max_batched_tokens: ready.max_num_batched_tokens,
            max_loras: 0,
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let served = self.state.served_model_names();
        Ok(Response::new(pb::ModelInfo {
            model_id: self.state.chat.text().model_id().to_string(),
            served_model_name: self.state.primary_model_name().to_string(),
            served_model_aliases: served.iter().skip(1).cloned().collect(),
            tokenizer_modes: Vec::new(),
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_lora: false,
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
    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request = request.into_inner();
        if request.request_ids.is_empty() {
            return Err(Status::invalid_argument("request_ids is required"));
        }
        self.state
            .engine_core_client()
            .abort(&request.request_ids)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        Ok(Response::new(pb::AbortResponse {}))
    }

    async fn drain(
        &self,
        _request: Request<pb::DrainRequest>,
    ) -> Result<Response<pb::DrainResponse>, Status> {
        self.begin_drain();
        self.report_not_serving().await;
        let in_flight = self.in_flight().min(u64::from(u32::MAX)) as u32;
        let state = if in_flight == 0 {
            pb::DrainState::Complete
        } else {
            pb::DrainState::InProgress
        };
        Ok(Response::new(pb::DrainResponse {
            state: state as i32,
            in_flight_requests: in_flight,
            message: String::new(),
        }))
    }

    async fn load_lora(
        &self,
        _request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        Err(Status::unimplemented("LoadLora"))
    }

    async fn unload_lora(
        &self,
        _request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        Err(Status::unimplemented("UnloadLora"))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        Err(Status::unimplemented("ListLoras"))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        Err(Status::unimplemented("GetKvEventSources"))
    }
}

fn text_error_to_status(error: vllm_text::Error) -> Status {
    let message = error.to_report_string();
    if error.is_request_validation_error() {
        Status::invalid_argument(message)
    } else {
        Status::internal(message)
    }
}
