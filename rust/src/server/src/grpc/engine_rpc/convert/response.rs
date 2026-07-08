use vllm_text::{DecodedTextEvent, FinishReason, Finished};

use super::super::pb;
use super::kv::kv_transfer_params_to_kv_session;

pub fn event_to_responses(
    event: DecodedTextEvent,
    request_id: &str,
    role: pb::EngineRole,
    kv_connector: Option<&str>,
    handoff_dp_rank: u32,
) -> Vec<pb::GenerateResponse> {
    match event {
        DecodedTextEvent::Start { .. } => Vec::new(),
        DecodedTextEvent::TextDelta {
            delta,
            token_ids,
            logprobs: _,
            finished,
        } => {
            let mut responses = Vec::new();
            if !token_ids.is_empty() || !delta.is_empty() {
                responses.push(pb::GenerateResponse {
                    request_id: request_id.to_string(),
                    event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                        token_ids,
                        text: delta,
                    })),
                    usage: None,
                });
            }
            if let Some(finished) = finished {
                responses.push(terminal_response(
                    &finished,
                    request_id,
                    role,
                    kv_connector,
                    handoff_dp_rank,
                ));
            }
            responses
        }
    }
}

fn terminal_response(
    finished: &Finished,
    request_id: &str,
    role: pb::EngineRole,
    kv_connector: Option<&str>,
    handoff_dp_rank: u32,
) -> pb::GenerateResponse {
    let usage = Some(to_usage(finished));
    if role == pb::EngineRole::Prefill {
        if matches!(
            finished.finish_reason,
            FinishReason::Abort | FinishReason::Error
        ) {
            return kv_transfer_error_response(
                request_id,
                "prefill did not complete successfully",
                usage,
            );
        }
        let Some(params) = finished.kv_transfer_params.as_ref() else {
            return kv_transfer_error_response(
                request_id,
                "prefill completed without KV transfer metadata",
                usage,
            );
        };
        let kv_session =
            kv_transfer_params_to_kv_session(params, request_id, kv_connector, handoff_dp_rank);
        if kv_session
            .attributes_struct
            .as_ref()
            .is_none_or(|attributes| attributes.fields.is_empty())
        {
            return kv_transfer_error_response(
                request_id,
                "prefill returned invalid KV transfer metadata",
                usage,
            );
        }
        return pb::GenerateResponse {
            request_id: request_id.to_string(),
            event: Some(pb::generate_response::Event::PrefillReady(
                pb::PrefillReady {
                    kv_session: Some(kv_session),
                },
            )),
            usage,
        };
    }

    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                reason: finish_reason_to_proto(&finished.finish_reason) as i32,
                message: String::new(),
            },
        )),
        usage,
    }
}

fn kv_transfer_error_response(
    request_id: &str,
    message: &str,
    usage: Option<pb::Usage>,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Error(pb::EngineError {
            code: pb::ErrorCode::KvTransferFailed as i32,
            message: message.to_string(),
            retry_hint: String::new(),
        })),
        usage,
    }
}

pub fn error_response(request_id: &str, message: String) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Error(pb::EngineError {
            code: pb::ErrorCode::Internal as i32,
            message,
            retry_hint: String::new(),
        })),
        usage: None,
    }
}

fn to_usage(finished: &Finished) -> pb::Usage {
    let prompt = finished.usage.prompt_token_count as u32;
    let completion = finished.usage.output_token_count as u32;
    pb::Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

fn finish_reason_to_proto(reason: &FinishReason) -> pb::FinishReason {
    match reason {
        FinishReason::Stop(_) | FinishReason::Repetition(_) => pb::FinishReason::Stop,
        FinishReason::Length => pb::FinishReason::Length,
        FinishReason::Abort => pb::FinishReason::Cancelled,
        FinishReason::Error => pb::FinishReason::Error,
    }
}
