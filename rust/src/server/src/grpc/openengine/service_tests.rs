// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::StreamExt as _;
use tonic::Request;
use vllm_chat::{
    ChatBackend, ChatLlm, ChatRenderer, ChatRequest, ChatTextBackend, DefaultChatOutputProcessor,
    DynChatOutputProcessor, DynChatRenderer, NewChatOutputProcessorOptions, RenderedPrompt,
};
use vllm_engine_core_client::mock_engine::default_ready_response;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
};
use vllm_engine_core_client::protocol::request::{EngineCoreRequest, EngineCoreRequestType};
use vllm_engine_core_client::protocol::stats::SchedulerStats;
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task_with_ready};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, TransportMode};
use vllm_llm::Llm;
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{Prompt, TextBackend};
use vllm_tokenizer::test_utils::TestTokenizer;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

use super::pb::control_server::Control as _;
use super::pb::inference_server::Inference as _;
use super::{HANDOFF_PROFILE, OpenEngineService, TRANSFER_BACKEND, pb, used_kv_blocks};
use crate::state::AppState;

type TestFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

struct MockEngineTask {
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Future for MockEngineTask {
    type Output = Result<(), tokio::task::JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        match self.join_handle.as_mut() {
            Some(join_handle) => Pin::new(join_handle).poll(cx),
            None => Poll::Ready(Ok(())),
        }
    }
}

impl Drop for MockEngineTask {
    fn drop(&mut self) {
        if let Some(join_handle) = &self.join_handle {
            join_handle.abort();
        }
    }
}

#[derive(Clone, Debug)]
struct FakeTextBackend;

impl TextBackend for FakeTextBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(TestTokenizer::new())
    }

    fn model_id(&self) -> &str {
        "canonical/test-model"
    }
}

impl ChatBackend for FakeTextBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(self.clone())
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> vllm_chat::Result<DynChatOutputProcessor> {
        Ok(Box::new(DefaultChatOutputProcessor::new(
            request,
            self.model_id(),
            self.tokenizer(),
            options.tool_call_parser,
            options.reasoning_parser,
        )?))
    }
}

impl ChatRenderer for FakeTextBackend {
    fn render(&self, _request: &ChatRequest) -> vllm_chat::Result<RenderedPrompt> {
        Ok(RenderedPrompt {
            prompt: Prompt::Text(String::new()),
            effective_template_kwargs: Default::default(),
        })
    }
}

fn request_outputs(request_id: &str) -> EngineCoreOutputs {
    RequestBatchOutputs {
        outputs: [
            (vec![b'h' as u32], None),
            (vec![b'i' as u32], None),
            (vec![b'!' as u32], Some(EngineCoreFinishReason::Stop)),
        ]
        .into_iter()
        .map(|(new_token_ids, finish_reason)| EngineCoreOutput {
            request_id: request_id.to_string(),
            new_token_ids,
            new_logprobs: None,
            new_prompt_logprobs_tensors: None,
            pooling_output: None,
            finish_reason,
            stop_reason: None,
            events: None,
            kv_transfer_params: None,
            ec_transfer_params: None,
            trace_headers: None,
            prefill_stats: None,
            routed_experts: None,
            num_nans_in_logits: 0,
        })
        .collect(),
        ..Default::default()
    }
    .into()
}

async fn setup_service(
    ready_response: EngineCoreReadyResponse,
    serve_generation: bool,
) -> (OpenEngineService, MockEngineTask) {
    let ipc = IpcNamespace::new().expect("create IPC namespace");
    let handshake_address = ipc.handshake_endpoint();
    let (shutdown_tx, join_handle) = spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        vec![0, 0],
        ready_response,
        move |dealer: &mut DealerSocket, push: &mut PushSocket| -> TestFuture<'_> {
            Box::pin(async move {
                let message = dealer.recv().await.expect("receive engine request").into_vec();
                if !serve_generation {
                    return;
                }
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&message[1]).expect("decode request");
                let outputs = request_outputs(&request.request_id);
                push.send(ZmqMessage::from(
                    rmp_serde::to_vec_named(&outputs).expect("encode outputs"),
                ))
                .await
                .expect("send outputs");
            })
        },
    );
    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name("canonical/test-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .expect("connect engine client");
    (
        service_from_client(client),
        MockEngineTask {
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        },
    )
}

fn service_from_client(client: EngineCoreClient) -> OpenEngineService {
    let chat = ChatLlm::from_shared_backend(
        Llm::new(client).with_request_id_randomization(false),
        Arc::new(FakeTextBackend) as Arc<dyn ChatTextBackend>,
    );
    let state = Arc::new(
        AppState::new(
            vec!["served-model".to_string(), "model-alias".to_string()],
            chat,
        )
        .with_tokenizer_mode("auto".to_string()),
    );
    OpenEngineService::new(state, "127.0.0.1".to_string()).expect("valid topology")
}

async fn send_scheduler_stats(
    push: &mut PushSocket,
    engine_index: u32,
    running: u64,
    waiting: u64,
    skipped_waiting: u64,
    kv_cache_usage: f64,
) {
    let outputs: EngineCoreOutputs = RequestBatchOutputs {
        engine_index,
        scheduler_stats: Some(Box::new(SchedulerStats {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            num_skipped_waiting_reqs: skipped_waiting,
            kv_cache_usage,
            ..Default::default()
        })),
        ..Default::default()
    }
    .into();
    push.send(ZmqMessage::from(
        rmp_serde::to_vec_named(&outputs).expect("encode scheduler stats"),
    ))
    .await
    .expect("send scheduler stats");
}

async fn get_scheduler_load(service: &OpenEngineService, include_per_rank: bool) -> pb::LoadInfo {
    wait_for_load(service, include_per_rank, |load| {
        load.queued_requests.is_some()
    })
    .await
}

async fn wait_for_load(
    service: &OpenEngineService,
    include_per_rank: bool,
    mut predicate: impl FnMut(&pb::LoadInfo) -> bool,
) -> pb::LoadInfo {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let load = service
                .get_load(Request::new(pb::GetLoadRequest { include_per_rank }))
                .await
                .expect("get load")
                .into_inner();
            if predicate(&load) {
                return load;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("expected load state")
}

async fn setup_abort_cleanup_service() -> (
    OpenEngineService,
    MockEngineTask,
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
) {
    let ipc = IpcNamespace::new().expect("create IPC namespace");
    let handshake_address = ipc.handshake_endpoint();
    let mut ready = default_ready_response();
    ready.kv_connector = Some("NixlConnector".to_string());
    ready.kv_role = Some("kv_producer".to_string());
    let (release_ack_tx, release_ack_rx) = tokio::sync::oneshot::channel();
    let (abort_received_tx, abort_received_rx) = tokio::sync::oneshot::channel();
    let (shutdown_tx, join_handle) = spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        vec![0, 0],
        ready,
        move |dealer: &mut DealerSocket, push: &mut PushSocket| -> TestFuture<'_> {
            Box::pin(async move {
                let add = dealer.recv().await.expect("receive add request").into_vec();
                assert_eq!(
                    add[0].as_ref(),
                    EngineCoreRequestType::Add.to_frame().as_ref()
                );
                send_scheduler_stats(push, 0, 0, 0, 0, 0.25).await;

                let abort = dealer.recv().await.expect("receive abort request").into_vec();
                assert_eq!(
                    abort[0].as_ref(),
                    EngineCoreRequestType::Abort.to_frame().as_ref()
                );
                let aborted_ids: Vec<String> =
                    rmp_serde::from_slice(&abort[1]).expect("decode aborted ids");
                assert_eq!(aborted_ids, vec!["req-1".to_string()]);
                let _ = abort_received_tx.send(());

                if release_ack_rx.await.is_ok() {
                    send_scheduler_stats(push, 0, 0, 0, 0, 0.25).await;
                }
            })
        },
    );
    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name("canonical/test-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .expect("connect engine client");
    (
        service_from_client(client),
        MockEngineTask {
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        },
        release_ack_tx,
        abort_received_rx,
    )
}

async fn abort_request_and_wait_for_scheduler_ack(
    service: &OpenEngineService,
    abort_received: tokio::sync::oneshot::Receiver<()>,
) -> pb::LoadInfo {
    let stream = service
        .generate(Request::new(base_request()))
        .await
        .expect("generate")
        .into_inner();
    let initial = get_scheduler_load(service, false).await;
    assert_eq!(initial.running_requests, Some(1));

    service
        .abort(Request::new(pb::AbortRequest {
            target: Some(pb::abort_request::Target::RequestId("req-1".to_string())),
        }))
        .await
        .expect("abort request");
    tokio::time::timeout(Duration::from_secs(2), abort_received)
        .await
        .expect("abort reached engine")
        .expect("abort notification sender");

    let responses: Vec<_> = tokio::time::timeout(
        Duration::from_secs(2),
        stream.map(|response| response.expect("abort response")).collect(),
    )
    .await
    .expect("aborted generation stream completed");
    assert!(responses.iter().any(|response| {
        matches!(
            response.event,
            Some(pb::generate_response::Event::Error(ref error))
                if error.code == pb::ErrorCode::Cancelled as i32
        )
    }));

    wait_for_load(service, false, |load| {
        load.running_requests == Some(1) && load.active_kv_sessions == Some(1)
    })
    .await
}

fn base_request() -> pb::GenerateRequest {
    pb::GenerateRequest {
        request_id: "req-1".to_string(),
        model: "served-model".to_string(),
        input: Some(pb::generate_request::Input::Prompt("hello".to_string())),
        stopping: Some(pb::StoppingOptions {
            max_tokens: Some(4),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn aggregate_generation_streams_tokens_and_terminal_usage() {
    let (service, engine_task) = setup_service(default_ready_response(), true).await;
    let responses: Vec<_> = service
        .generate(Request::new(base_request()))
        .await
        .expect("generate")
        .into_inner()
        .map(|response| response.expect("stream response"))
        .collect()
        .await;
    assert!(responses.iter().any(|response| {
        matches!(response.event, Some(pb::generate_response::Event::Token(_)))
    }));
    let terminal = responses
        .iter()
        .find(|response| {
            matches!(
                response.event,
                Some(pb::generate_response::Event::Finished(_))
            )
        })
        .expect("terminal response");
    let usage = terminal.usage.as_ref().expect("terminal usage");
    assert_eq!(usage.completion_tokens, 3);
    assert_eq!(usage.total_tokens, usage.prompt_tokens + 3);
    engine_task.await.expect("mock engine task");
}

#[tokio::test]
async fn aggregate_generation_accepts_advertised_canonical_model_id() {
    let (service, engine_task) = setup_service(default_ready_response(), true).await;
    let mut request = base_request();
    request.model = "canonical/test-model".to_string();
    let responses: Vec<_> = service
        .generate(Request::new(request))
        .await
        .expect("generate with canonical model id")
        .into_inner()
        .map(|response| response.expect("stream response"))
        .collect()
        .await;
    assert!(responses.iter().any(|response| {
        matches!(
            response.event,
            Some(pb::generate_response::Event::Finished(_))
        )
    }));
    engine_task.await.expect("mock engine task");
}

#[tokio::test]
async fn discovery_reports_prefill_profile_tokenizer_aliases_and_kv_source() {
    let mut ready = default_ready_response();
    ready.kv_connector = Some("NixlConnector".to_string());
    ready.kv_role = Some("kv_producer".to_string());
    ready.kv_events_publisher = Some("zmq".to_string());
    ready.kv_events_endpoint = Some("tcp://*:5557".to_string());
    ready.kv_events_replay_endpoint = Some("tcp://*:5657".to_string());
    ready.kv_events_topic = Some("kv-events".to_string());
    ready.kv_events_buffer_steps = 10_000;
    ready.kv_events_hwm = 100_000;
    ready.kv_events_max_queue_size = 100_000;
    let (service, _engine_task) = setup_service(ready, false).await;

    let server = service
        .get_server_info(Request::new(pb::GetServerInfoRequest {}))
        .await
        .expect("server info")
        .into_inner();
    assert_eq!(server.schema_revision, 3);
    assert_eq!(server.schema_release, super::SCHEMA_RELEASE);
    assert_eq!(server.engine_role, pb::EngineRole::Prefill as i32);
    assert_eq!(server.supported_models, vec!["canonical/test-model"]);
    let connector = server.kv_connector.expect("connector discovery");
    assert_eq!(connector.handoff_profile, HANDOFF_PROFILE);
    assert_eq!(connector.transfer_backend, TRANSFER_BACKEND);

    let model = service
        .get_model_info(Request::new(pb::GetModelInfoRequest::default()))
        .await
        .expect("model info")
        .into_inner();
    assert_eq!(model.model_id, "canonical/test-model");
    assert_eq!(model.served_model_name, "served-model");
    assert_eq!(model.served_model_aliases, vec!["model-alias"]);
    let tokenizer = model.tokenizer.expect("tokenizer");
    assert_eq!(tokenizer.source, "canonical/test-model");
    assert_eq!(tokenizer.mode, "auto");
    assert_eq!(model.tokenizer_modes, vec!["auto"]);

    let sources = service
        .get_kv_event_sources(Request::new(pb::GetKvEventSourcesRequest::default()))
        .await
        .expect("KV sources")
        .into_inner()
        .sources;
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].transport, "zmq");
    assert_eq!(sources[0].encoding, "msgpack");
    assert_eq!(
        sources[0].endpoint_addr.as_ref().expect("endpoint").port,
        5557
    );
    assert_eq!(sources[0].replay_endpoint, "tcp://127.0.0.1:5657");
    assert_eq!(sources[0].buffer_steps, Some(10_000));
}

#[tokio::test]
async fn get_load_reports_scheduler_backed_dp_rank_load_and_kv_usage() {
    let ipc = IpcNamespace::new().expect("create IPC namespace");
    let handshake_address = ipc.handshake_endpoint();
    let mut ready_0 = default_ready_response();
    ready_0.data_parallel_size = 2;
    ready_0.data_parallel_rank = 0;
    ready_0.num_gpu_blocks = 100;
    let mut ready_1 = ready_0.clone();
    ready_1.data_parallel_rank = 1;
    ready_1.num_gpu_blocks = 200;

    let (shutdown_0, join_0) = spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        vec![0, 0],
        ready_0,
        move |_dealer: &mut DealerSocket, push: &mut PushSocket| -> TestFuture<'_> {
            Box::pin(async move {
                send_scheduler_stats(push, 0, 2, 1, 1, 0.25).await;
            })
        },
    );
    let (shutdown_1, join_1) = spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        vec![1, 0],
        ready_1,
        move |_dealer: &mut DealerSocket, push: &mut PushSocket| -> TestFuture<'_> {
            Box::pin(async move {
                send_scheduler_stats(push, 1, 1, 2, 0, 0.5).await;
            })
        },
    );
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address,
            advertised_host: "127.0.0.1".to_string(),
            engine_count: 2,
            ready_timeout: Duration::from_secs(2),
            local_input_address: Some(ipc.input_endpoint()),
            local_output_address: Some(ipc.output_endpoint()),
        },
        coordinator_mode: None,
        model_name: "canonical/test-model".to_string(),
        client_index: 0,
    })
    .await
    .expect("connect DP engine client");
    let service = service_from_client(client);

    let load = get_scheduler_load(&service, true).await;
    assert_eq!(load.running_requests, Some(3));
    assert_eq!(load.queued_requests, Some(4));
    assert_eq!(load.used_kv_blocks, Some(125));
    assert_eq!(load.total_kv_blocks, Some(300));
    assert_eq!(load.ranks.len(), 2);
    assert_eq!(load.ranks[0].data_parallel_rank, Some(0));
    assert_eq!(load.ranks[0].running_requests, Some(2));
    assert_eq!(load.ranks[0].queued_requests, Some(2));
    assert_eq!(load.ranks[0].used_kv_blocks, Some(25));
    assert_eq!(load.ranks[1].data_parallel_rank, Some(1));
    assert_eq!(load.ranks[1].running_requests, Some(1));
    assert_eq!(load.ranks[1].queued_requests, Some(2));
    assert_eq!(load.ranks[1].used_kv_blocks, Some(100));

    MockEngineTask {
        shutdown_tx: Some(shutdown_0),
        join_handle: Some(join_0),
    }
    .await
    .expect("rank 0 engine task");
    MockEngineTask {
        shutdown_tx: Some(shutdown_1),
        join_handle: Some(join_1),
    }
    .await
    .expect("rank 1 engine task");
}

#[tokio::test]
async fn abort_load_stays_nonzero_until_a_new_zero_scheduler_snapshot() {
    let (service, engine_task, release_ack, abort_received) = setup_abort_cleanup_service().await;
    let pending = abort_request_and_wait_for_scheduler_ack(&service, abort_received).await;
    assert_eq!(pending.queued_requests, Some(0));

    release_ack.send(()).expect("release scheduler acknowledgement");
    let cleaned = wait_for_load(&service, false, |load| {
        load.running_requests == Some(0) && load.active_kv_sessions == Some(0)
    })
    .await;
    assert_eq!(cleaned.queued_requests, Some(0));
    service.shutdown_background_tasks().await;
    engine_task.await.expect("mock engine task");
}

#[tokio::test]
async fn shutdown_cancels_and_joins_stalled_abort_cleanup() {
    let (service, engine_task, release_ack, abort_received) = setup_abort_cleanup_service().await;
    abort_request_and_wait_for_scheduler_ack(&service, abort_received).await;

    tokio::time::timeout(Duration::from_secs(1), service.shutdown_background_tasks())
        .await
        .expect("cleanup task shutdown is bounded");
    let cleaned = service
        .get_load(Request::new(pb::GetLoadRequest::default()))
        .await
        .expect("get load after cleanup shutdown")
        .into_inner();
    assert_eq!(cleaned.running_requests, Some(0));
    assert_eq!(cleaned.active_kv_sessions, Some(0));

    drop(release_ack);
    engine_task.await.expect("mock engine task");
}

#[tokio::test]
async fn load_lora_rejects_invalid_peft_directories_before_registration() {
    let mut ready = default_ready_response();
    ready.supports_lora = true;
    let (service, _engine_task) = setup_service(ready, false).await;

    let non_object = tempfile::tempdir().expect("non-object adapter directory");
    std::fs::write(non_object.path().join("adapter_config.json"), b"[]")
        .expect("write adapter config");
    std::fs::write(
        non_object.path().join("adapter_model.safetensors"),
        b"weights",
    )
    .expect("write adapter weights");
    let error = service
        .load_lora(Request::new(pb::LoadLoraRequest {
            adapter: Some(pb::LoraAdapter {
                lora_id: 1,
                lora_name: "non-object".to_string(),
                source_path: non_object.path().to_string_lossy().into_owned(),
            }),
        }))
        .await
        .expect_err("non-object adapter config accepted");
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("must contain a JSON object"));

    let missing_weights = tempfile::tempdir().expect("missing-weights adapter directory");
    std::fs::write(missing_weights.path().join("adapter_config.json"), b"{}")
        .expect("write adapter config");
    let error = service
        .load_lora(Request::new(pb::LoadLoraRequest {
            adapter: Some(pb::LoraAdapter {
                lora_id: 2,
                lora_name: "missing-weights".to_string(),
                source_path: missing_weights.path().to_string_lossy().into_owned(),
            }),
        }))
        .await
        .expect_err("adapter without weights accepted");
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("is missing adapter_model.safetensors"));

    let valid = tempfile::tempdir().expect("valid adapter directory");
    std::fs::write(valid.path().join("adapter_config.json"), b"{}").expect("write adapter config");
    std::fs::write(valid.path().join("adapter_model.bin"), b"weights")
        .expect("write adapter weights");
    let loaded = service
        .load_lora(Request::new(pb::LoadLoraRequest {
            adapter: Some(pb::LoraAdapter {
                lora_id: 3,
                lora_name: "valid".to_string(),
                source_path: valid.path().to_string_lossy().into_owned(),
            }),
        }))
        .await
        .expect("valid adapter registration")
        .into_inner();
    assert!(!loaded.already_loaded);
    assert_eq!(
        loaded.adapter.expect("registered adapter").lora_name,
        "valid"
    );
}

#[tokio::test]
async fn drain_completes_and_rejects_new_work_process_wide() {
    let (service, _engine_task) = setup_service(default_ready_response(), false).await;
    let updates: Vec<_> = service
        .drain(Request::new(pb::DrainRequest {
            stop_accepting_new_requests: true,
            ..Default::default()
        }))
        .await
        .expect("drain")
        .into_inner()
        .map(|response| response.expect("drain response"))
        .collect()
        .await;
    assert!(matches!(
        updates.last().and_then(|update| update.event.as_ref()),
        Some(pb::drain_response::Event::State(state))
            if *state == pb::DrainState::Complete as i32
    ));
    let error = match service.generate(Request::new(base_request())).await {
        Ok(_) => panic!("draining server accepted new work"),
        Err(error) => error,
    };
    assert_eq!(error.code(), tonic::Code::Unavailable);
    let health = service
        .health(Request::new(pb::HealthRequest::default()))
        .await
        .expect("health")
        .into_inner();
    assert_eq!(health.state, pb::HealthState::Draining as i32);
}

#[test]
fn kv_usage_converts_to_bounded_block_counts() {
    assert_eq!(used_kv_blocks(1_000, Some(0.125)), Some(125));
    assert_eq!(used_kv_blocks(1_000, Some(2.0)), Some(1_000));
    assert_eq!(used_kv_blocks(1_000, Some(-0.1)), None);
    assert_eq!(used_kv_blocks(1_000, Some(f64::NAN)), None);
    assert_eq!(used_kv_blocks(1_000, None), None);
}
