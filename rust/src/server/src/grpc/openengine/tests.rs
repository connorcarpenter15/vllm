//! Wire-shape tests for the OpenEngine v1 gRPC service.
//!
//! These mirror [`crate::grpc::tests`] (the `vllm` Generate service tests):
//! they stand up the tonic `OpenEngine` service over the shared [`AppState`],
//! backed by the `mock-engine` test double, and assert the OpenEngine wire shape
//! for the happy-path stream, abort, drain, and the discovery RPCs. Real-engine
//! coverage lives in the cluster smoke matrix (plan Phase 5).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::StreamExt as _;
use serial_test::serial;
use tonic::transport::Server as TonicServer;
use vllm_chat::{
    ChatBackend, ChatLlm, ChatRenderer, ChatRequest, ChatTextBackend, DefaultChatOutputProcessor,
    DynChatOutputProcessor, DynChatRenderer, MediaContentPart, NewChatOutputProcessorOptions,
    RenderedPrompt,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
};
use vllm_engine_core_client::mock_engine::default_ready_response;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, EngineId};
use vllm_llm::Llm;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};
use vllm_text::{Prompt, TextBackend};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

use super::pb::open_engine_client::OpenEngineClient;
use super::{OpenEngineServer, OpenEngineServiceImpl, pb};
use crate::state::AppState;

// ========================================================================================
// Helpers (mirrors crate::grpc::tests)
// ========================================================================================

type TestFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

fn boxed_test_future<'a>(future: impl Future<Output = ()> + Send + 'a) -> TestFuture<'a> {
    Box::pin(future)
}

struct MockEngineTask {
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MockEngineTask {
    fn new(
        (shutdown_tx, join_handle): (
            tokio::sync::oneshot::Sender<()>,
            tokio::task::JoinHandle<()>,
        ),
    ) -> Self {
        Self {
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        }
    }
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

fn request_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: None,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason: None,
        events: None,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn engine_outputs_for_request(
    request_id: &str,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> EngineCoreOutputs {
    EngineCoreOutputs {
        engine_index: 0,
        outputs: output_specs
            .into_iter()
            .map(|(token_ids, finish_reason)| request_output(request_id, token_ids, finish_reason))
            .collect(),
        scheduler_stats: None,
        timestamp: 0.0,
        utility_output: None,
        finished_requests: None,
        wave_complete: None,
        start_wave: None,
    }
}

fn default_stream_output_specs() -> Vec<(Vec<u32>, Option<EngineCoreFinishReason>)> {
    vec![
        (vec![b'h' as u32], None),
        (vec![b'i' as u32], None),
        (vec![b'!' as u32], Some(EngineCoreFinishReason::Stop)),
    ]
}

async fn send_outputs(push: &mut PushSocket, outputs: EngineCoreOutputs) {
    push.send(ZmqMessage::from(
        rmp_serde::to_vec_named(&outputs).expect("encode outputs"),
    ))
    .await
    .expect("send outputs");
}

async fn recv_engine_message(dealer: &mut DealerSocket) -> Vec<bytes::Bytes> {
    dealer.recv().await.expect("recv engine message").into_vec()
}

fn test_llm(client: EngineCoreClient) -> Llm {
    Llm::new(client).with_request_id_randomization(false)
}

#[derive(Clone, Debug)]
struct FakeTextBackend;

#[derive(Debug)]
struct FakeTokenizer;

impl Tokenizer for FakeTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> vllm_text::tokenizer::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> vllm_text::tokenizer::Result<String> {
        Ok(
            String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                .into_owned(),
        )
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.bytes().next().map(u32::from)
    }
}

impl TextBackend for FakeTextBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(FakeTokenizer)
    }

    fn model_id(&self) -> &str {
        "test-model"
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
        })
    }
}

/// Stand up an OpenEngine gRPC server backed by a mock engine that serves a
/// single request with the given output specs. Returns the OpenEngine client,
/// the gRPC server task, and the mock engine task.
///
/// RPCs that do not drive generation (discovery, abort, drain) never send an
/// `Add` to the engine, so the mock closure simply stays parked on its `recv`
/// until the test drops the returned `MockEngineTask`.
async fn openengine_test_server(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (
    OpenEngineClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
) {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = engine_id.into();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        move |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, output_specs),
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name("test-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .expect("connect client");

    let chat = ChatLlm::from_shared_backend(
        test_llm(client),
        Arc::new(FakeTextBackend) as Arc<dyn ChatTextBackend>,
    );
    let state = Arc::new(AppState::new(vec!["test-model".to_string()], chat));
    let svc = OpenEngineServer::new(OpenEngineServiceImpl::new(state));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind grpc listener");
    let addr = listener.local_addr().expect("local addr");

    let server_task = tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        TonicServer::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .expect("grpc server");
    });

    let grpc_client = OpenEngineClient::connect(format!("http://{addr}"))
        .await
        .expect("connect grpc client");

    (grpc_client, server_task, engine_task)
}

fn base_request() -> pb::GenerateRequest {
    pb::GenerateRequest {
        request_id: "req-1".to_string(),
        model: "test-model".to_string(),
        input: Some(pb::generate_request::Input::Prompt("hello".to_string())),
        stream: true,
        ..Default::default()
    }
}

// ========================================================================================
// Generate
// ========================================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn generate_streams_tokens_then_finishes() {
    let (mut client, server_task, engine_task) =
        openengine_test_server(b"oe-generate", default_stream_output_specs()).await;

    let stream = client
        .generate(base_request())
        .await
        .expect("generate")
        .into_inner();

    let responses: Vec<pb::GenerateResponse> =
        stream.map(|r| r.expect("stream item")).collect().await;

    // Every response should carry the request id.
    assert!(responses.iter().all(|r| r.request_id == "req-1"));

    let text: String = responses
        .iter()
        .filter_map(|r| match &r.event {
            Some(pb::generate_response::Event::Token(t)) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    // The terminal stop token ('!') is suppressed from visible text.
    assert_eq!(text, "hi");

    let finished = responses
        .iter()
        .rev()
        .find_map(|r| match &r.event {
            Some(pb::generate_response::Event::Finished(f)) => Some((f, r.usage.as_ref())),
            _ => None,
        })
        .expect("finished event present");
    assert_eq!(finished.0.reason, pb::FinishReason::Stop as i32);

    let usage = finished.1.expect("usage on terminal response");
    assert_eq!(usage.prompt_tokens, 5); // "hello"
    assert_eq!(usage.completion_tokens, 3);
    assert_eq!(usage.total_tokens, 8);

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn generate_with_token_ids_input() {
    let (mut client, server_task, engine_task) =
        openengine_test_server(b"oe-token-ids", default_stream_output_specs()).await;

    let stream = client
        .generate(pb::GenerateRequest {
            request_id: "req-tok".to_string(),
            model: "test-model".to_string(),
            input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
            stream: true,
            ..Default::default()
        })
        .await
        .expect("generate")
        .into_inner();

    let responses: Vec<pb::GenerateResponse> =
        stream.map(|r| r.expect("stream item")).collect().await;
    let text: String = responses
        .iter()
        .filter_map(|r| match &r.event {
            Some(pb::generate_response::Event::Token(t)) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "hi");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn generate_missing_input_is_invalid_argument() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-no-input", default_stream_output_specs()).await;

    // Server-streaming RPCs surface a handler error on the initial call.
    let status = client
        .generate(pb::GenerateRequest {
            request_id: "req-no-input".to_string(),
            model: "test-model".to_string(),
            input: None,
            ..Default::default()
        })
        .await
        .expect_err("should fail without input");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn generate_rejects_wrong_model() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-wrong-model", default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "req-wrong".to_string(),
            model: "other-model".to_string(),
            input: Some(pb::generate_request::Input::Prompt("hi".to_string())),
            ..Default::default()
        })
        .await
        .expect_err("should fail with wrong model");

    assert_eq!(status.code(), tonic::Code::NotFound);

    server_task.abort();
}

// ========================================================================================
// Discovery RPCs
// ========================================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn get_engine_info_reports_aggregated_role_and_topology() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-engine-info", default_stream_output_specs()).await;

    let info = client
        .get_engine_info(pb::GetEngineInfoRequest {})
        .await
        .expect("get_engine_info")
        .into_inner();

    assert_eq!(info.engine_name, "vllm");
    assert_eq!(info.engine_version, "test-vllm-version");
    assert_eq!(info.api_version, "openengine.v1");
    assert_eq!(info.role, pb::EngineRole::Aggregated as i32);
    assert_eq!(info.supported_models, vec!["test-model".to_string()]);

    let parallelism = info.parallelism.expect("parallelism present");
    assert_eq!(parallelism.tensor_parallel_size, 1);
    assert_eq!(parallelism.pipeline_parallel_size, 1);
    assert_eq!(parallelism.data_parallel_size, 1);

    let kv = info.kv_connector.expect("kv_connector present");
    assert!(!kv.enabled);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn get_model_info_reports_caps_from_handshake() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-model-info", default_stream_output_specs()).await;

    let info = client
        .get_model_info(pb::GetModelInfoRequest {})
        .await
        .expect("get_model_info")
        .into_inner();

    assert_eq!(info.model_id, "test-model");
    assert_eq!(info.served_model_name, "test-model");
    assert!(info.served_model_aliases.is_empty());
    assert_eq!(info.kv_block_size, 16);
    assert_eq!(info.max_running_requests, 256);
    assert_eq!(info.max_batched_tokens, 8192);
    assert!(info.supports_text_input);
    assert!(info.supports_token_ids_input);
    assert!(!info.supports_multimodal);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn get_load_reports_idle() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-load", default_stream_output_specs()).await;

    let load = client
        .get_load(pb::GetLoadRequest::default())
        .await
        .expect("get_load")
        .into_inner();

    assert_eq!(load.running_requests, 0);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn get_kv_connector_info_disabled_without_connector() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-kv-conn", default_stream_output_specs()).await;

    let info = client
        .get_kv_connector_info(pb::GetKvConnectorInfoRequest {})
        .await
        .expect("get_kv_connector_info")
        .into_inner();

    assert!(!info.enabled);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn get_kv_event_sources_empty_without_publisher() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-kv-events", default_stream_output_specs()).await;

    let resp = client
        .get_kv_event_sources(pb::GetKvEventSourcesRequest::default())
        .await
        .expect("get_kv_event_sources")
        .into_inner();

    assert!(resp.sources.is_empty());

    server_task.abort();
}

// ----------------------------------------------------------------------------------------
// KV event source selection (pure, no server harness) — consolidator vs per-rank
// ----------------------------------------------------------------------------------------

fn ready_with_kv_events(
    dp_rank: u32,
    publisher: Option<&str>,
    endpoint: Option<&str>,
    consolidated: Option<&str>,
) -> EngineCoreReadyResponse {
    EngineCoreReadyResponse {
        data_parallel_rank: dp_rank,
        kv_events_publisher: publisher.map(str::to_string),
        kv_events_endpoint: endpoint.map(str::to_string),
        kv_events_topic: Some("kv".to_string()),
        kv_events_consolidated_endpoint: consolidated.map(str::to_string),
        ..default_ready_response()
    }
}

#[test]
fn build_kv_event_sources_prefers_consolidator_single_rank0_source() {
    // Two DP engines each expose a raw ZMQ publisher, but the engine also
    // advertises the KVBM consolidator endpoint. The consolidator merges both
    // ranks into one deduped stream, so exactly one rank-0 source is emitted
    // (the consolidator endpoint), not the per-rank raw publishers.
    let r0 = ready_with_kv_events(
        0,
        Some("zmq"),
        Some("tcp://*:5557"),
        Some("tcp://0.0.0.0:57001"),
    );
    let r1 = ready_with_kv_events(1, Some("zmq"), Some("tcp://*:5557"), None);
    let sources = super::build_kv_event_sources(&[&r0, &r1]);

    assert_eq!(sources.len(), 1, "consolidator collapses to one source");
    let src = &sources[0];
    assert_eq!(src.transport, "zmq");
    assert_eq!(src.data_parallel_rank, 0);
    assert_eq!(src.endpoint, "tcp://0.0.0.0:57001");
    // Consolidator republishes on an empty topic; advertising the raw vLLM topic
    // here would make the router's SUB filter reject every consolidated message.
    assert_eq!(src.topic, "", "consolidator source must advertise an empty topic");
    let addr = src.endpoint_addr.as_ref().expect("routable endpoint_addr");
    assert_eq!(addr.port, 57001);
    assert_ne!(addr.host, "0.0.0.0", "bind wildcard rewritten to a routable host");
}

#[test]
fn build_kv_event_sources_falls_back_to_per_rank_publishers() {
    // No consolidator: one source per DP engine, with vLLM's per-rank port
    // offset (rank 0 unchanged, rank 1 = base + 1).
    let r0 = ready_with_kv_events(0, Some("zmq"), Some("tcp://*:5557"), None);
    let r1 = ready_with_kv_events(1, Some("zmq"), Some("tcp://*:5557"), None);
    let sources = super::build_kv_event_sources(&[&r0, &r1]);

    assert_eq!(sources.len(), 2);
    let ports: Vec<u32> = sources
        .iter()
        .map(|s| s.endpoint_addr.as_ref().expect("endpoint_addr").port)
        .collect();
    assert_eq!(ports, vec![5557, 5558]);
}

#[test]
fn build_kv_event_sources_empty_without_publisher_or_consolidator() {
    let r0 = ready_with_kv_events(0, None, None, None);
    let sources = super::build_kv_event_sources(&[&r0]);
    assert!(sources.is_empty());
}

// ========================================================================================
// Health
// ========================================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn health_reports_ready_without_probe() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-health", default_stream_output_specs()).await;

    let resp = client
        .health(pb::HealthRequest {
            include_inference_probe: false,
            ..Default::default()
        })
        .await
        .expect("health")
        .into_inner();

    assert_eq!(resp.state, pb::HealthState::Ready as i32);
    assert!(resp.checks.iter().any(|c| c.name == "engine"));

    server_task.abort();
}

// ========================================================================================
// Abort
// ========================================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_unknown_request_is_idempotent() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-abort", default_stream_output_specs()).await;

    let resp = client
        .abort(pb::AbortRequest {
            request_id: "not-in-flight".to_string(),
            ..Default::default()
        })
        .await
        .expect("abort")
        .into_inner();

    assert_eq!(resp.status, pb::AbortStatus::Aborted as i32);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_all_is_unsupported() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-abort-all", default_stream_output_specs()).await;

    let resp = client
        .abort(pb::AbortRequest {
            abort_all: true,
            ..Default::default()
        })
        .await
        .expect("abort")
        .into_inner();

    assert_eq!(resp.status, pb::AbortStatus::Unsupported as i32);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_empty_request_id_is_invalid_argument() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-abort-empty", default_stream_output_specs()).await;

    let status = client
        .abort(pb::AbortRequest::default())
        .await
        .expect_err("empty request_id should be rejected");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    server_task.abort();
}

// ========================================================================================
// Drain
// ========================================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn drain_completes_when_idle() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-drain", default_stream_output_specs()).await;

    let stream = client
        .drain(pb::DrainRequest::default())
        .await
        .expect("drain")
        .into_inner();

    let responses: Vec<pb::DrainResponse> =
        stream.map(|r| r.expect("drain item")).collect().await;

    let first = responses.first().expect("at least one drain response");
    assert_eq!(first.state, pb::DrainState::Started as i32);

    let last = responses.last().expect("at least one drain response");
    assert_eq!(last.state, pb::DrainState::Complete as i32);
    assert_eq!(last.in_flight_requests, 0);

    server_task.abort();
}

// ========================================================================================
// Unimplemented subscriptions
// ========================================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn subscribe_kv_events_is_unimplemented() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-sub-kv", default_stream_output_specs()).await;

    let status = client
        .subscribe_kv_events(pb::SubscribeKvEventsRequest::default())
        .await
        .expect_err("subscribe_kv_events should be unimplemented");

    assert_eq!(status.code(), tonic::Code::Unimplemented);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn subscribe_runtime_events_is_unimplemented() {
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-sub-rt", default_stream_output_specs()).await;

    let status = client
        .subscribe_runtime_events(pb::SubscribeRuntimeEventsRequest::default())
        .await
        .expect_err("subscribe_runtime_events should be unimplemented");

    assert_eq!(status.code(), tonic::Code::Unimplemented);

    server_task.abort();
}

// ========================================================================================
// DP helpers (data-parallel orchestration: per-rank KV blocks + per-rank KV
// event endpoints)
// ========================================================================================

#[test]
fn per_rank_kv_blocks_divides_aggregate_by_dp_size() {
    // DP=1 (or 0, treated as 1): aggregate is already per-rank.
    assert_eq!(super::per_rank_kv_blocks(1000, 1), 1000);
    assert_eq!(super::per_rank_kv_blocks(1000, 0), 1000);
    // DP>1: floor-divide the aggregate across ranks (matches per_rank_kv_blocks
    // in dynamo/components/src/dynamo/vllm/capacity.py).
    assert_eq!(super::per_rank_kv_blocks(1000, 4), 250);
    assert_eq!(super::per_rank_kv_blocks(1001, 4), 250); // floor
    // Zero aggregate (e.g. Ray DP backend sentinel) stays zero.
    assert_eq!(super::per_rank_kv_blocks(0, 8), 0);
    // Fewer blocks than ranks clamps to 1 rather than 0.
    assert_eq!(super::per_rank_kv_blocks(3, 8), 1);
}

#[test]
fn offset_endpoint_port_matches_vllm_convention() {
    // Rank 0 is never offset.
    assert_eq!(super::offset_endpoint_port("tcp://*:5557", 0), "tcp://*:5557");
    // tcp: port += data_parallel_rank.
    assert_eq!(super::offset_endpoint_port("tcp://*:5557", 1), "tcp://*:5558");
    assert_eq!(
        super::offset_endpoint_port("tcp://127.0.0.1:5557", 7),
        "tcp://127.0.0.1:5564"
    );
    // inproc: `_dp{rank}` suffix.
    assert_eq!(super::offset_endpoint_port("inproc://cache", 3), "inproc://cache_dp3");
    // Empty endpoint is returned unchanged.
    assert_eq!(super::offset_endpoint_port("", 5), "");
}

// ========================================================================================
// Multimodal wire shape
// ========================================================================================

fn image_item(source: pb::media_item::Source) -> pb::MediaItem {
    pb::MediaItem {
        modality: pb::Modality::Image as i32,
        source: Some(source),
        ..Default::default()
    }
}

#[test]
fn media_parts_maps_url_data_uri_and_raw_bytes() {
    let media = vec![
        pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::Url("http://h/a.png".to_string())),
            uuid: "uid-1".to_string(),
            ..Default::default()
        },
        // MODALITY_UNSPECIFIED is forward-compat and treated as image.
        pb::MediaItem {
            modality: pb::Modality::Unspecified as i32,
            source: Some(pb::media_item::Source::DataUri(
                "data:image/png;base64,AAAA".to_string(),
            )),
            ..Default::default()
        },
        pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::RawBytes(vec![1, 2, 3])),
            mime_type: "image/png".to_string(),
            ..Default::default()
        },
    ];

    let parts = super::convert::media_parts_from_request(&media).expect("convert media");
    assert_eq!(parts.len(), 3);

    match &parts[0] {
        MediaContentPart::ImageUrl { url, detail, uuid } => {
            assert_eq!(url, "http://h/a.png");
            assert!(detail.is_none());
            assert_eq!(uuid.as_deref(), Some("uid-1"));
        }
        _ => panic!("part 0 should be an ImageUrl from a url source"),
    }
    match &parts[1] {
        // A data: URI also lands on ImageUrl; the media connector decodes it.
        MediaContentPart::ImageUrl { url, uuid, .. } => {
            assert_eq!(url, "data:image/png;base64,AAAA");
            assert!(uuid.is_none(), "absent proto uuid maps to None");
        }
        _ => panic!("part 1 should be an ImageUrl from a data_uri source"),
    }
    match &parts[2] {
        MediaContentPart::ImageData {
            data, mime_type, ..
        } => {
            assert_eq!(data, &[1, 2, 3]);
            assert_eq!(mime_type.as_deref(), Some("image/png"));
        }
        _ => panic!("part 2 should be ImageData from a raw_bytes source"),
    }
}

#[test]
fn media_parts_empty_input_yields_no_parts() {
    assert!(super::convert::media_parts_from_request(&[]).unwrap().is_empty());
}

#[test]
fn media_parts_rejects_item_without_source() {
    let media = vec![pb::MediaItem {
        modality: pb::Modality::Image as i32,
        source: None,
        ..Default::default()
    }];
    let status = super::convert::media_parts_from_request(&media)
        .expect_err("a media item with no source must be rejected");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
}

#[test]
fn media_parts_rejects_video_modality_in_v1() {
    // v1 is image-only; video/audio preprocessing is a later phase.
    let media = vec![pb::MediaItem {
        modality: pb::Modality::Video as i32,
        source: Some(pb::media_item::Source::Url("http://h/clip.mp4".to_string())),
        ..Default::default()
    }];
    let status = super::convert::media_parts_from_request(&media)
        .expect_err("video modality must be rejected in v1");
    assert_eq!(status.code(), tonic::Code::Unimplemented);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn generate_with_media_requires_token_ids_input() {
    // Placeholder markers are expanded engine-side, so a multimodal request must
    // carry token_ids — a text prompt is rejected before the engine is touched.
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-media-text", default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "req-media-text".to_string(),
            model: "test-model".to_string(),
            input: Some(pb::generate_request::Input::Prompt("hi".to_string())),
            media: vec![image_item(pb::media_item::Source::Url(
                "http://h/a.png".to_string(),
            ))],
            stream: true,
            ..Default::default()
        })
        .await
        .expect_err("media with a text prompt should be rejected");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn generate_with_media_on_text_only_backend_fails_closed() {
    // The mock `FakeTextBackend` has no `MultimodalModelInfo`, so a media
    // request must error rather than silently drop the image and degrade to a
    // text completion.
    let (mut client, server_task, _engine_task) =
        openengine_test_server(b"oe-media-textonly", default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "req-media-textonly".to_string(),
            model: "test-model".to_string(),
            input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
            media: vec![image_item(pb::media_item::Source::Url(
                "http://h/a.png".to_string(),
            ))],
            stream: true,
            ..Default::default()
        })
        .await
        .expect_err("media on a text-only backend should fail closed");

    assert_eq!(status.code(), tonic::Code::Internal);

    server_task.abort();
}
