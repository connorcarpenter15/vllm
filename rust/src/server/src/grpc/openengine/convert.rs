//! Conversion between OpenEngine v1 protobuf types and the internal
//! `vllm-text` request/response types.
//!
//! This mirrors [`crate::grpc::convert`] (the `vllm` Generate service) but
//! targets the vendor-neutral OpenEngine contract consumed by the Dynamo
//! sidecar.

use std::collections::HashMap;

use tonic::Status;
use uuid::Uuid;
use vllm_text::{
    DecodedTextEvent, FinishReason, Finished, Prompt, SamplingParams, TextDecodeOptions,
    TextRequest,
};

use super::pb;

// ========================================================================================
// Request conversion
// ========================================================================================

/// Convert an OpenEngine `GenerateRequest` into the internal `TextRequest`.
///
/// If `req.model` is non-empty, it must match one of `served_model_names`;
/// otherwise the request is rejected with `NotFound`. An empty string is
/// treated as "unset" (proto3 default) and accepted.
pub fn to_text_request(
    req: pb::GenerateRequest,
    served_model_names: &[String],
) -> Result<TextRequest, Status> {
    if !req.model.is_empty() && !served_model_names.iter().any(|n| n == &req.model) {
        return Err(Status::not_found(format!("model `{}` not found", req.model)));
    }

    let prompt = match req.input {
        Some(pb::generate_request::Input::Prompt(text)) => Prompt::Text(text),
        Some(pb::generate_request::Input::TokenIds(ids)) => Prompt::TokenIds(ids.ids),
        None => return Err(Status::invalid_argument("input (prompt or token_ids) is required")),
    };

    let request_id = if req.request_id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        req.request_id
    };

    let mut sampling_params = build_sampling_params(req.sampling.as_ref());

    // Split stop conditions into stop strings (decode-side matching) and stop
    // token IDs (sampler-side handling).
    let mut stop_strings: Vec<String> = Vec::new();
    let mut stop_token_ids: Vec<u32> = Vec::new();
    for cond in &req.stop {
        match cond.condition.as_ref() {
            Some(pb::stop_condition::Condition::StopText(text)) => {
                stop_strings.push(text.clone());
            }
            Some(pb::stop_condition::Condition::StopTokenId(id)) => stop_token_ids.push(*id),
            None => {}
        }
    }
    if !stop_token_ids.is_empty() {
        sampling_params.stop_token_ids = Some(stop_token_ids);
    }

    // Decode-role disaggregation: lift the KV session attributes into
    // `kv_transfer_params` so the engine-core request carries them through to
    // the connector. Phase 3 refines the exact session contents; the encoding
    // here round-trips with [`kv_transfer_params_to_kv_session`].
    if let Some(kv_session) = req.kv_session.as_ref()
        && !kv_session.attributes.is_empty()
    {
        let kv_json = kv_session_attributes_to_json(&kv_session.attributes);
        let map = sampling_params.vllm_xargs.get_or_insert_with(Default::default);
        map.insert("kv_transfer_params".to_string(), kv_json);
    }

    let decode_options = TextDecodeOptions {
        skip_special_tokens: true,
        include_stop_str_in_output: false,
        stop_strings: (!stop_strings.is_empty()).then_some(stop_strings),
        min_tokens: 0,
    };

    Ok(TextRequest {
        request_id,
        prompt,
        mm_features: None,
        sampling_params,
        decode_options,
        // The OpenEngine Generate RPC is always server-streaming; the `stream`
        // flag controls whether intermediate token deltas are emitted or only
        // the terminal output is returned.
        intermediate: req.stream,
        priority: 0,
        cache_salt: None,
        add_special_tokens: true,
        data_parallel_rank: None,
    })
}

fn build_sampling_params(sampling: Option<&pb::SamplingParams>) -> SamplingParams {
    // Default to greedy (0.0) for the programmatic gRPC API when the caller does
    // not provide sampling params, matching the `vllm` Generate service.
    let Some(s) = sampling else {
        return SamplingParams {
            temperature: Some(0.0),
            ..SamplingParams::default()
        };
    };

    SamplingParams {
        temperature: Some(s.temperature as f32),
        // The protobuf scalar default (`0`) is treated as "unset" for the
        // remaining fields and left to the lowering stage to resolve.
        top_p: (s.top_p != 0.0).then_some(s.top_p as f32),
        top_k: (s.top_k != 0).then_some(s.top_k as u32),
        // Seed `0` is indistinguishable from "unset" in the OpenEngine proto
        // (the field is a plain `uint64`), so it is treated as no override.
        seed: (s.seed != 0).then_some(s.seed as i64),
        max_tokens: (s.max_tokens != 0).then_some(s.max_tokens),
        frequency_penalty: (s.frequency_penalty != 0.0).then_some(s.frequency_penalty as f32),
        presence_penalty: (s.presence_penalty != 0.0).then_some(s.presence_penalty as f32),
        ..SamplingParams::default()
    }
}

// ========================================================================================
// Response conversion
// ========================================================================================

/// Map a decoded text event to the OpenEngine `GenerateResponse` messages it
/// produces, given the request ID and engine role.
///
/// A single `TextDelta` can yield up to two messages: a `token` event for any
/// newly produced text/tokens, followed by a terminal `finished` (or, for the
/// prefill role, `prefill_ready`) event.
pub fn event_to_responses(
    event: DecodedTextEvent,
    request_id: &str,
    role: pb::EngineRole,
    kv_connector: Option<&str>,
) -> Vec<pb::GenerateResponse> {
    match event {
        // OpenEngine has no dedicated "start" event; prompt metadata is not
        // surfaced over this contract.
        DecodedTextEvent::Start { .. } => Vec::new(),
        DecodedTextEvent::TextDelta {
            delta,
            token_ids,
            logprobs: _,
            finished,
        } => {
            let mut responses = Vec::new();

            // Emit a token event for any visible content in this delta.
            if !token_ids.is_empty() || !delta.is_empty() {
                responses.push(pb::GenerateResponse {
                    request_id: request_id.to_string(),
                    event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                        token_ids,
                        text: delta,
                        logprobs: Vec::new(),
                    })),
                    usage: None,
                });
            }

            if let Some(finished) = finished {
                responses.push(terminal_response(&finished, request_id, role, kv_connector));
            }

            responses
        }
    }
}

/// Build the terminal `GenerateResponse` for a finished request.
fn terminal_response(
    finished: &Finished,
    request_id: &str,
    role: pb::EngineRole,
    kv_connector: Option<&str>,
) -> pb::GenerateResponse {
    let usage = Some(to_usage(finished));

    // Prefill-role engines hand the request off to the decode side: emit a
    // `prefill_ready` carrying the KV session derived from the connector's
    // transfer params instead of a normal completion.
    if role == pb::EngineRole::Prefill {
        let kv_session = finished.kv_transfer_params.as_ref().map(|params| {
            kv_transfer_params_to_kv_session(params, request_id, kv_connector)
        });
        return pb::GenerateResponse {
            request_id: request_id.to_string(),
            event: Some(pb::generate_response::Event::PrefillReady(pb::PrefillReady {
                kv_session,
            })),
            usage,
        };
    }

    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Finished(pb::GenerationFinished {
            reason: finish_reason_to_proto(&finished.finish_reason) as i32,
            message: String::new(),
        })),
        usage,
    }
}

/// Build an `error` `GenerateResponse` for a mid-stream failure.
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
    let prompt = finished.prompt_token_count as u32;
    let completion = finished.output_token_count as u32;
    pb::Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

/// Map the internal finish reason to the OpenEngine enum.
///
/// `Repetition` (a model-driven natural stop) maps to `STOP`; OpenEngine has no
/// dedicated repetition reason.
fn finish_reason_to_proto(reason: &FinishReason) -> pb::FinishReason {
    match reason {
        FinishReason::Stop(_) | FinishReason::Repetition => pb::FinishReason::Stop,
        FinishReason::Length => pb::FinishReason::Length,
        FinishReason::Abort => pb::FinishReason::Cancelled,
        FinishReason::Error => pb::FinishReason::Error,
    }
}

// ========================================================================================
// Engine role mapping
// ========================================================================================

/// Derive the OpenEngine role from the engine's authoritative KV transfer role.
///
/// `kv_producer` → prefill, `kv_consumer` → decode, everything else (including
/// `kv_both` and no connector) → aggregated.
pub fn role_from_kv_role(kv_role: Option<&str>) -> pb::EngineRole {
    match kv_role {
        Some("kv_producer") => pb::EngineRole::Prefill,
        Some("kv_consumer") => pb::EngineRole::Decode,
        _ => pb::EngineRole::Aggregated,
    }
}

/// Mark a request as the prefill (producer) side of a disaggregated exchange.
///
/// NixlConnector treats a request as the prefill node only when its
/// `kv_transfer_params` carries `do_remote_decode: true`. Setting it makes the
/// engine retain the KV blocks after prefill and emit the handoff metadata
/// (`remote_block_ids` / `remote_engine_id` / `remote_host` / `remote_port` /
/// `do_remote_prefill: true`) in the terminal `kv_transfer_params`, which
/// [`terminal_response`] then packs into `PrefillReady.kv_session` for the
/// decode peer. Existing `kv_transfer_params` keys are preserved.
pub fn mark_prefill_request(request: &mut TextRequest) {
    let xargs = request
        .sampling_params
        .vllm_xargs
        .get_or_insert_with(Default::default);
    let entry = xargs
        .entry("kv_transfer_params".to_string())
        .or_insert_with(|| serde_json::Value::Object(Default::default()));
    if let Some(obj) = entry.as_object_mut() {
        obj.insert("do_remote_decode".to_string(), serde_json::Value::Bool(true));
    }
}

// ========================================================================================
// KV session <-> kv_transfer_params encoding
// ========================================================================================

/// Decode KV session attributes (string-valued map) into a JSON object suitable
/// for `kv_transfer_params`.
///
/// Each value is parsed as JSON when possible (so the prefill side can encode
/// arbitrary types as JSON strings) and falls back to a plain string otherwise.
fn kv_session_attributes_to_json(attrs: &HashMap<String, String>) -> serde_json::Value {
    let map = attrs
        .iter()
        .map(|(k, v)| {
            let value = serde_json::from_str(v).unwrap_or_else(|_| serde_json::Value::String(v.clone()));
            (k.clone(), value)
        })
        .collect();
    serde_json::Value::Object(map)
}

/// Encode `kv_transfer_params` (a JSON object) into a KV session reference.
///
/// Object fields become string-valued attributes (scalar JSON values are
/// stringified plainly; compound values are JSON-encoded) so they round-trip
/// through [`kv_session_attributes_to_json`] on the decode side.
fn kv_transfer_params_to_kv_session(
    params: &serde_json::Value,
    request_id: &str,
    kv_connector: Option<&str>,
) -> pb::KvSessionRef {
    let attributes = match params {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, v)| {
                let s = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (k.clone(), s)
            })
            .collect(),
        _ => HashMap::new(),
    };

    pb::KvSessionRef {
        session_id: request_id.to_string(),
        transfer_backend: kv_connector.unwrap_or_default().to_string(),
        endpoints: Vec::new(),
        dp_rank: 0,
        attributes,
    }
}

#[cfg(test)]
mod tests {
    use vllm_text::{FinishReason, Finished};

    use super::super::pb;
    use super::*;

    fn base_request() -> pb::GenerateRequest {
        pb::GenerateRequest {
            request_id: "req".to_string(),
            model: "test-model".to_string(),
            input: Some(pb::generate_request::Input::Prompt("hi".to_string())),
            ..Default::default()
        }
    }

    #[test]
    fn unset_sampling_defaults_to_greedy() {
        let text = to_text_request(base_request(), &["test-model".to_string()]).expect("convert");
        assert_eq!(text.sampling_params.temperature, Some(0.0));
    }

    #[test]
    fn sampling_params_propagate() {
        let req = pb::GenerateRequest {
            sampling: Some(pb::SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 50,
                max_tokens: 16,
                seed: 42,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, &["test-model".to_string()]).expect("convert");
        assert_eq!(text.sampling_params.temperature, Some(0.7));
        assert_eq!(text.sampling_params.top_p, Some(0.9));
        assert_eq!(text.sampling_params.top_k, Some(50));
        assert_eq!(text.sampling_params.max_tokens, Some(16));
        assert_eq!(text.sampling_params.seed, Some(42));
    }

    #[test]
    fn zero_seed_is_treated_as_unset() {
        let req = pb::GenerateRequest {
            sampling: Some(pb::SamplingParams {
                seed: 0,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, &["test-model".to_string()]).expect("convert");
        assert_eq!(text.sampling_params.seed, None);
    }

    #[test]
    fn stop_conditions_split_into_strings_and_token_ids() {
        let req = pb::GenerateRequest {
            stop: vec![
                pb::StopCondition {
                    condition: Some(pb::stop_condition::Condition::StopText("END".to_string())),
                },
                pb::StopCondition {
                    condition: Some(pb::stop_condition::Condition::StopTokenId(7)),
                },
            ],
            ..base_request()
        };
        let text = to_text_request(req, &["test-model".to_string()]).expect("convert");
        assert_eq!(
            text.decode_options.stop_strings,
            Some(vec!["END".to_string()])
        );
        assert_eq!(text.sampling_params.stop_token_ids, Some(vec![7]));
    }

    #[test]
    fn missing_input_is_rejected() {
        let req = pb::GenerateRequest {
            input: None,
            ..base_request()
        };
        let status = to_text_request(req, &["test-model".to_string()]).expect_err("should reject");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn wrong_model_is_rejected() {
        let req = pb::GenerateRequest {
            model: "other".to_string(),
            ..base_request()
        };
        let status = to_text_request(req, &["test-model".to_string()]).expect_err("should reject");
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    fn finished(reason: FinishReason) -> Finished {
        Finished {
            prompt_token_count: 3,
            output_token_count: 2,
            finish_reason: reason,
            kv_transfer_params: None,
        }
    }

    #[test]
    fn aggregated_terminal_is_finished_with_usage() {
        let responses = event_to_responses(
            DecodedTextEvent::TextDelta {
                delta: "hi".to_string(),
                token_ids: vec![1, 2],
                logprobs: None,
                finished: Some(finished(FinishReason::Length)),
            },
            "req",
            pb::EngineRole::Aggregated,
            None,
        );
        // token event, then finished event.
        assert_eq!(responses.len(), 2);
        let token = match &responses[0].event {
            Some(pb::generate_response::Event::Token(t)) => t,
            _ => panic!("first event should be a token"),
        };
        assert_eq!(token.text, "hi");
        let finished = match &responses[1].event {
            Some(pb::generate_response::Event::Finished(f)) => f,
            _ => panic!("second event should be finished"),
        };
        assert_eq!(finished.reason, pb::FinishReason::Length as i32);
        let usage = responses[1].usage.as_ref().expect("usage present");
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 5);
    }

    #[test]
    fn prefill_terminal_emits_prefill_ready() {
        let mut fin = finished(FinishReason::Length);
        fin.kv_transfer_params = Some(serde_json::json!({
            "remote_engine_id": "engine-7",
            "remote_block_ids": [1, 2, 3]
        }));
        let responses = event_to_responses(
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: vec![5],
                logprobs: None,
                finished: Some(fin),
            },
            "req",
            pb::EngineRole::Prefill,
            Some("NixlConnector"),
        );
        let prefill = responses
            .iter()
            .find_map(|r| match &r.event {
                Some(pb::generate_response::Event::PrefillReady(p)) => Some(p),
                _ => None,
            })
            .expect("prefill_ready present");
        let session = prefill.kv_session.as_ref().expect("kv_session present");
        assert_eq!(session.session_id, "req");
        assert_eq!(session.transfer_backend, "NixlConnector");
        assert_eq!(
            session.attributes.get("remote_engine_id"),
            Some(&"engine-7".to_string())
        );
        // Compound values round-trip as JSON strings.
        assert_eq!(
            session.attributes.get("remote_block_ids"),
            Some(&"[1,2,3]".to_string())
        );
    }

    #[test]
    fn kv_session_attributes_round_trip_into_kv_transfer_params() {
        let mut attributes = HashMap::new();
        attributes.insert("remote_engine_id".to_string(), "engine-7".to_string());
        attributes.insert("remote_block_ids".to_string(), "[1,2,3]".to_string());
        let req = pb::GenerateRequest {
            kv_session: Some(pb::KvSessionRef {
                session_id: "sess".to_string(),
                attributes,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, &["test-model".to_string()]).expect("convert");
        let xargs = text.sampling_params.vllm_xargs.expect("xargs present");
        let params = xargs.get("kv_transfer_params").expect("kv_transfer_params present");
        assert_eq!(params["remote_engine_id"], "engine-7");
        assert_eq!(params["remote_block_ids"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn mark_prefill_sets_do_remote_decode() {
        let mut text = to_text_request(base_request(), &["test-model".to_string()]).expect("convert");
        mark_prefill_request(&mut text);
        let xargs = text.sampling_params.vllm_xargs.expect("xargs present");
        let params = xargs.get("kv_transfer_params").expect("kv_transfer_params present");
        assert_eq!(params["do_remote_decode"], serde_json::json!(true));
    }

    #[test]
    fn mark_prefill_preserves_existing_kv_transfer_params() {
        let mut attributes = HashMap::new();
        attributes.insert("remote_engine_id".to_string(), "engine-7".to_string());
        let req = pb::GenerateRequest {
            kv_session: Some(pb::KvSessionRef {
                attributes,
                ..Default::default()
            }),
            ..base_request()
        };
        let mut text = to_text_request(req, &["test-model".to_string()]).expect("convert");
        mark_prefill_request(&mut text);
        let xargs = text.sampling_params.vllm_xargs.expect("xargs present");
        let params = xargs.get("kv_transfer_params").expect("kv_transfer_params present");
        assert_eq!(params["do_remote_decode"], serde_json::json!(true));
        assert_eq!(params["remote_engine_id"], "engine-7");
    }

    #[test]
    fn role_mapping() {
        assert_eq!(role_from_kv_role(Some("kv_producer")), pb::EngineRole::Prefill);
        assert_eq!(role_from_kv_role(Some("kv_consumer")), pb::EngineRole::Decode);
        assert_eq!(role_from_kv_role(Some("kv_both")), pb::EngineRole::Aggregated);
        assert_eq!(role_from_kv_role(None), pb::EngineRole::Aggregated);
    }
}
