//! Wire-shape tests for the engine RPC v1 gRPC service.
//!
//! These mirror [`crate::grpc::tests`] (the `vllm` Generate service tests):
//! they stand up the tonic engine RPC service over the shared [`AppState`],
//! backed by the `mock-engine` test double, and assert the engine RPC wire shape
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
    DynChatOutputProcessor, DynChatRenderer, NewChatOutputProcessorOptions, ParserSelection,
    RenderedPrompt,
};
use vllm_engine_core_client::mock_engine::{MockEngineConfig, default_ready_response};
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
    UtilityCallOutput,
};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
use vllm_engine_core_client::test_utils::{
    IpcNamespace, spawn_mock_engine_task, spawn_mock_engine_task_with_config,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, EngineId};
use vllm_llm::Llm;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};
use vllm_text::{Prompt, TextBackend};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

use super::pb::engine_client::EngineClient;
use super::{EngineServer, EngineServiceImpl, pb};
use crate::state::AppState;

mod discovery;
mod generate;
mod lifecycle;
mod lora;
mod media;
mod topology;

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
    RequestBatchOutputs {
        outputs: output_specs
            .into_iter()
            .map(|(token_ids, finish_reason)| request_output(request_id, token_ids, finish_reason))
            .collect(),
        ..Default::default()
    }
    .into()
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

    fn id_to_token(&self, id: u32) -> Option<String> {
        char::from_u32(id).map(|token| token.to_string())
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
            effective_template_kwargs: Default::default(),
        })
    }
}

/// Stand up an engine RPC gRPC server backed by a mock engine that serves a
/// single request with the given output specs. Returns the engine RPC client,
/// the gRPC server task, and the mock engine task.
///
/// RPCs that do not drive generation (discovery, abort, drain) never send an
/// `Add` to the engine, so the mock closure simply stays parked on its `recv`
/// until the test drops the returned `MockEngineTask`.
async fn engine_rpc_test_server(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (
    EngineClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
) {
    engine_rpc_test_server_with_parsers(
        engine_id,
        output_specs,
        ParserSelection::Auto,
        ParserSelection::Auto,
    )
    .await
}

async fn engine_rpc_test_server_with_parsers(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    tool_call_parser: ParserSelection,
    reasoning_parser: ParserSelection,
) -> (
    EngineClient<tonic::transport::Channel>,
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
    )
    .with_tool_call_parser(tool_call_parser)
    .with_reasoning_parser(reasoning_parser);
    let state = Arc::new(AppState::new(vec!["test-model".to_string()], chat));
    let svc = EngineServer::new(EngineServiceImpl::new(state));

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

    let grpc_client = EngineClient::connect(format!("http://{addr}"))
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

async fn lora_scripted_test_server<F>(
    supports_lora: bool,
    run: F,
) -> (
    EngineClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
)
where
    F: for<'a> FnOnce(&'a mut DealerSocket, &'a mut PushSocket) -> TestFuture<'a> + Send + 'static,
{
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let mut ready = default_ready_response();
    ready.supports_lora = supports_lora;
    ready.max_loras = if supports_lora { 4 } else { 0 };
    let config = MockEngineConfig {
        local: true,
        headless: true,
        ready_response: ready,
        ..Default::default()
    };

    let engine_task = MockEngineTask::new(spawn_mock_engine_task_with_config(
        handshake_address.clone(),
        vec![0x00, 0x00],
        config,
        run,
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
    let engine_rpc = EngineServer::new(EngineServiceImpl::new(state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(async move {
        TonicServer::builder()
            .add_service(engine_rpc)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    (EngineClient::new(channel), server_task, engine_task)
}

async fn lora_test_server(
    supports_lora: bool,
    unload_result: bool,
) -> (
    EngineClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
) {
    lora_scripted_test_server(supports_lora, move |dealer, push| {
        boxed_test_future(async move {
            let load = recv_engine_message(dealer).await;
            assert_eq!(load[0].as_ref(), &[0x03]);
            let load: rmpv::Value = rmp_serde::from_slice(&load[1]).unwrap();
            let load = load.as_array().expect("load utility tuple");
            assert_eq!(load[2], rmpv::Value::from("add_lora"));
            send_utility_bool(push, load[1].as_u64().unwrap(), true).await;

            let add = recv_engine_message(dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
            let lora = request.lora_request.expect("LoRA request");
            assert_eq!(lora.lora_name, "adapter-a");
            assert_eq!(lora.lora_int_id, 17);
            send_outputs(
                push,
                engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
            )
            .await;

            let unload = recv_engine_message(dealer).await;
            assert_eq!(unload[0].as_ref(), &[0x03]);
            let unload: rmpv::Value = rmp_serde::from_slice(&unload[1]).unwrap();
            let unload = unload.as_array().expect("unload utility tuple");
            assert_eq!(unload[2], rmpv::Value::from("remove_lora"));
            send_utility_bool(push, unload[1].as_u64().unwrap(), unload_result).await;
        })
    })
    .await
}

async fn send_utility_bool(push: &mut PushSocket, call_id: u64, result: bool) {
    send_outputs(
        push,
        UtilityCallOutput {
            engine_index: 0,
            timestamp: 0.0,
            output: UtilityOutput {
                call_id: call_id.into(),
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(rmpv::Value::from(
                    result,
                ))),
            },
        }
        .into(),
    )
    .await;
}
