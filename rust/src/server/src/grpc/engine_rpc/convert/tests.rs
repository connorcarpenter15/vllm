use vllm_llm::TokenUsage;
use vllm_text::{DecodedTextEvent, FinishReason, Finished};

use super::super::pb;
use super::kv::kv_transfer_params_to_kv_session;
use super::*;
use crate::grpc::struct_json::{json_to_prost_struct, prost_struct_to_json};

fn base_request() -> pb::GenerateRequest {
    pb::GenerateRequest {
        request_id: "req".to_string(),
        model: "test-model".to_string(),
        input: Some(pb::generate_request::Input::Prompt("hi".to_string())),
        ..Default::default()
    }
}

fn finished(reason: FinishReason) -> Finished {
    Finished {
        usage: TokenUsage {
            prompt_token_count: 3,
            output_token_count: 2,
            cached_token_count: 0,
        },
        finish_reason: reason,
        kv_transfer_params: None,
    }
}

#[test]
fn request_conversion_preserves_sampling_stop_and_routing_fields() {
    let request = pb::GenerateRequest {
        sampling: Some(pb::SamplingParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(50),
            seed: Some(-7),
            min_tokens: Some(2),
            ..Default::default()
        }),
        stop: vec![
            pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopText("END".to_string())),
            },
            pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopTokenId(7)),
            },
        ],
        priority: Some(-3),
        data_parallel_rank: Some(2),
        ..base_request()
    };
    let text = to_text_request(request, &["test-model".to_string()]).unwrap();
    assert_eq!(text.sampling_params.temperature, Some(0.7));
    assert_eq!(text.sampling_params.seed, Some(-7));
    assert_eq!(text.sampling_params.stop_token_ids, Some(vec![7]));
    assert_eq!(
        text.decode_options.stop_strings,
        Some(vec!["END".to_string()])
    );
    assert_eq!(text.decode_options.min_tokens, 2);
    assert_eq!(text.priority, -3);
    assert_eq!(text.data_parallel_rank, Some(2));
}

#[test]
fn request_conversion_rejects_missing_input_and_wrong_model() {
    let missing = pb::GenerateRequest {
        input: None,
        ..base_request()
    };
    assert_eq!(
        to_text_request(missing, &["test-model".to_string()]).unwrap_err().code(),
        tonic::Code::InvalidArgument
    );
    let wrong = pb::GenerateRequest {
        model: "other".to_string(),
        ..base_request()
    };
    assert_eq!(
        to_text_request(wrong, &["test-model".to_string()]).unwrap_err().code(),
        tonic::Code::NotFound
    );
}

#[test]
fn kv_session_round_trips_into_request_transfer_params() {
    let params = serde_json::json!({
        "remote_engine_id": "engine-7",
        "remote_port": 20097,
        "remote_block_ids": [4, 5, 6],
    });
    let request = pb::GenerateRequest {
        kv_session: Some(kv_transfer_params_to_kv_session(
            &params,
            "session",
            Some("NixlConnector"),
            2,
        )),
        ..base_request()
    };
    let text = to_text_request(request, &["test-model".to_string()]).unwrap();
    assert_eq!(
        text.sampling_params.vllm_xargs.unwrap().get("kv_transfer_params"),
        Some(&params)
    );
}

#[test]
fn role_validation_requires_exact_handoff_shape() {
    let mut request = base_request();
    assert!(
        validate_disaggregated_request(
            &request,
            pb::EngineRole::Decode,
            Some("NixlConnector"),
            1,
            0,
        )
        .is_err()
    );
    request.kv_session = Some(pb::KvSessionRef {
        session_id: "session".to_string(),
        transfer_backend: "NixlConnector".to_string(),
        dp_rank: 0,
        attributes_struct: json_to_prost_struct(&serde_json::json!({"remote_port": 8000})),
    });
    assert_eq!(
        validate_disaggregated_request(
            &request,
            pb::EngineRole::Decode,
            Some("NixlConnector"),
            1,
            0,
        )
        .unwrap(),
        0
    );
    assert!(
        validate_disaggregated_request(
            &request,
            pb::EngineRole::Prefill,
            Some("NixlConnector"),
            1,
            0,
        )
        .is_err()
    );
}

#[test]
fn prefill_terminal_emits_typed_handoff() {
    let mut terminal = finished(FinishReason::Length);
    terminal.kv_transfer_params = Some(serde_json::json!({
        "remote_engine_id": "engine-7",
        "remote_block_ids": [1, 2, 3],
    }));
    let responses = event_to_responses(
        DecodedTextEvent::TextDelta {
            delta: String::new(),
            token_ids: vec![5],
            logprobs: None,
            finished: Some(terminal),
        },
        "req",
        pb::EngineRole::Prefill,
        Some("NixlConnector"),
        3,
    );
    let session = responses
        .iter()
        .find_map(|response| match &response.event {
            Some(pb::generate_response::Event::PrefillReady(ready)) => ready.kv_session.as_ref(),
            _ => None,
        })
        .unwrap();
    assert_eq!(session.dp_rank, 3);
    assert_eq!(
        prost_struct_to_json(session.attributes_struct.as_ref().unwrap())["remote_block_ids"],
        serde_json::json!([1, 2, 3])
    );
}

#[test]
fn failed_prefill_never_emits_ready() {
    let mut terminal = finished(FinishReason::Error);
    terminal.kv_transfer_params = Some(serde_json::json!({"remote_port": 8000}));
    let responses = event_to_responses(
        DecodedTextEvent::TextDelta {
            delta: String::new(),
            token_ids: Vec::new(),
            logprobs: None,
            finished: Some(terminal),
        },
        "req",
        pb::EngineRole::Prefill,
        Some("NixlConnector"),
        0,
    );
    assert!(matches!(
        responses[0].event,
        Some(pb::generate_response::Event::Error(_))
    ));
}

#[test]
fn prefill_marker_preserves_existing_params() {
    let request = pb::GenerateRequest {
        kv_session: Some(pb::KvSessionRef {
            attributes_struct: json_to_prost_struct(
                &serde_json::json!({"remote_engine_id": "engine-7"}),
            ),
            ..Default::default()
        }),
        ..base_request()
    };
    let mut text = to_text_request(request, &["test-model".to_string()]).unwrap();
    mark_prefill_request(&mut text);
    let params = &text.sampling_params.vllm_xargs.unwrap()["kv_transfer_params"];
    assert_eq!(params["do_remote_decode"], true);
    assert_eq!(params["remote_engine_id"], "engine-7");
}
