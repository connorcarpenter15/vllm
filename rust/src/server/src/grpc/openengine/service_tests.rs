// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

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
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task_with_ready};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{Prompt, TextBackend};
use vllm_tokenizer::test_utils::TestTokenizer;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

use super::pb::control_server::Control as _;
use super::pb::inference_server::Inference as _;
use super::{HANDOFF_PROFILE, OpenEngineService, TRANSFER_BACKEND, pb};
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
    (
        OpenEngineService::new(state, "127.0.0.1".to_string()).expect("valid topology"),
        MockEngineTask {
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        },
    )
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
    assert_eq!(server.engine_role, pb::EngineRole::Prefill as i32);
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
    assert_eq!(
        model.tokenizer.expect("tokenizer").source,
        "canonical/test-model"
    );

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
