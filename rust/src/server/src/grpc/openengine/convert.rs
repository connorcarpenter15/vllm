//! Conversion between OpenEngine v1 protobuf types and the internal
//! `vllm-text` request/response types.
//!
//! This mirrors [`crate::grpc::convert`] (the `vllm` Generate service) but
//! targets the vendor-neutral OpenEngine contract consumed by the Dynamo
//! sidecar.

use tonic::Status;
use uuid::Uuid;
use vllm_chat::MediaContentPart;
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

    // Decode-role disaggregation: lift the KV session handoff into
    // `kv_transfer_params` so the engine-core request carries it through to the
    // connector. Round-trips with [`kv_transfer_params_to_kv_session`].
    if let Some(kv_session) = req.kv_session.as_ref() {
        if let Some(s) = kv_session.attributes_struct.as_ref().filter(|s| !s.fields.is_empty()) {
            let map = sampling_params.vllm_xargs.get_or_insert_with(Default::default);
            map.insert("kv_transfer_params".to_string(), prost_struct_to_json(s));
        }
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
        // Honour the KV router's forced internal DP rank when set, pinning this
        // request to the engine rank that holds its prefix (KV-aware routing).
        // Unset → the engine load-balances across its DP ranks as before.
        data_parallel_rank: req.data_parallel_rank,
        lora_request: None,
    })
}

/// Build chat-layer media parts from the proto `media` field, in wire order
/// (which aligns with the placeholder markers carried in the prompt tokens).
///
/// v1 supports the image modality only: `MODALITY_IMAGE` and the
/// forward-compatible `MODALITY_UNSPECIFIED` map to image parts; `VIDEO` /
/// `AUDIO` are rejected with `Unimplemented` until the engine-side
/// preprocessing for them lands. `url` and `data_uri` sources both become
/// `ImageUrl` (the media connector fetches/decodes either); `raw_bytes`
/// becomes `ImageData`. A `MediaItem` with no `source` set is rejected.
pub fn media_parts_from_request(
    media: &[pb::MediaItem],
) -> Result<Vec<MediaContentPart>, Status> {
    let mut parts = Vec::with_capacity(media.len());
    for item in media {
        let modality = pb::Modality::try_from(item.modality).unwrap_or(pb::Modality::Unspecified);
        match modality {
            pb::Modality::Image | pb::Modality::Unspecified => {}
            other => {
                return Err(Status::unimplemented(format!(
                    "media modality {other:?} is not supported by the vLLM OpenEngine service \
                     (image only in v1)"
                )));
            }
        }
        let uuid = (!item.uuid.is_empty()).then(|| item.uuid.clone());
        let part = match item.source.as_ref() {
            Some(pb::media_item::Source::Url(url)) => MediaContentPart::ImageUrl {
                url: url.clone(),
                detail: None,
                uuid,
            },
            Some(pb::media_item::Source::DataUri(uri)) => MediaContentPart::ImageUrl {
                url: uri.clone(),
                detail: None,
                uuid,
            },
            Some(pb::media_item::Source::RawBytes(bytes)) => MediaContentPart::ImageData {
                data: bytes.clone(),
                mime_type: (!item.mime_type.is_empty()).then(|| item.mime_type.clone()),
                uuid,
                detail: None,
            },
            None => {
                return Err(Status::invalid_argument(
                    "media item has no source (expected url, data_uri, or raw_bytes)",
                ));
            }
        };
        parts.push(part);
    }
    Ok(parts)
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
        ignore_eos: s.ignore_eos,
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
                        top_logprobs: Vec::new(),
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

/// Encode `kv_transfer_params` (a JSON object) into a KV session reference.
///
/// The params are carried in `attributes_struct` (a `google.protobuf.Struct`)
/// so numbers, booleans, and arrays survive the wire with their JSON type
/// intact — no string re-parsing on the decode side (the connector reads
/// `remote_port` / `tp_size` etc. as their native types). Round-trips through
/// the `attributes_struct` branch of [`to_text_request`].
fn kv_transfer_params_to_kv_session(
    params: &serde_json::Value,
    request_id: &str,
    kv_connector: Option<&str>,
) -> pb::KvSessionRef {
    pb::KvSessionRef {
        session_id: request_id.to_string(),
        transfer_backend: kv_connector.unwrap_or_default().to_string(),
        endpoints: Vec::new(),
        dp_rank: 0,
        attributes_struct: json_to_prost_struct(params),
    }
}

/// Convert a JSON object into a `google.protobuf.Struct`. Non-object inputs
/// (the connector always hands back an object) yield `None`.
pub(crate) fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_prost_value(v))).collect(),
        }),
        _ => None,
    }
}

fn json_to_prost_value(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(prost_types::NullValue::NullValue as i32),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(prost_types::Struct {
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_prost_value(v))).collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

/// Convert a `google.protobuf.Struct` back into a JSON object.
pub(crate) fn prost_struct_to_json(s: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        s.fields.iter().map(|(k, v)| (k.clone(), prost_value_to_json(v))).collect(),
    )
}

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match &value.kind {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::NumberValue(n)) => number_to_json(*n),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(l)) => {
            serde_json::Value::Array(l.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => prost_struct_to_json(s),
    }
}

/// `google.protobuf.Struct` numbers are IEEE-754 doubles. Recover integral
/// values as JSON integers (the connector encodes ints like `remote_port` /
/// `tp_size`, which downstream code reads as ints, not floats).
fn number_to_json(n: f64) -> serde_json::Value {
    if n.is_finite() && n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
        serde_json::Value::Number((n as i64).into())
    } else {
        serde_json::Number::from_f64(n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
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
                ignore_eos: true,
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
        assert!(text.sampling_params.ignore_eos);
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

    #[test]
    fn forwards_router_forced_dp_rank() {
        let req = pb::GenerateRequest {
            data_parallel_rank: Some(3),
            ..base_request()
        };
        let text = to_text_request(req, &["test-model".to_string()]).expect("convert");
        assert_eq!(text.data_parallel_rank, Some(3));
    }

    #[test]
    fn unset_dp_rank_defaults_to_none() {
        let text = to_text_request(base_request(), &["test-model".to_string()]).expect("convert");
        assert_eq!(text.data_parallel_rank, None);
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
        // Typed attributes preserve their JSON types (no stringification).
        let attrs = prost_struct_to_json(
            session
                .attributes_struct
                .as_ref()
                .expect("attributes_struct present"),
        );
        assert_eq!(attrs["remote_engine_id"], serde_json::json!("engine-7"));
        assert_eq!(attrs["remote_block_ids"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn kv_session_attributes_struct_round_trips_into_kv_transfer_params() {
        // Mirrors the wire path: prefill builds a Struct, decode reads it back.
        let params = serde_json::json!({
            "remote_engine_id": "engine-7",
            "remote_port": 20097,
            "tp_size": 1,
            "do_remote_prefill": true,
            "remote_block_ids": [4, 5, 6],
        });
        let session = kv_transfer_params_to_kv_session(&params, "sess", Some("NixlConnector"));
        let req = pb::GenerateRequest {
            kv_session: Some(session),
            ..base_request()
        };
        let text = to_text_request(req, &["test-model".to_string()]).expect("convert");
        let xargs = text.sampling_params.vllm_xargs.expect("xargs present");
        let kv = xargs.get("kv_transfer_params").expect("kv_transfer_params present");
        // Numbers stay numbers (ints recovered), bools stay bools, arrays stay arrays.
        assert_eq!(kv["remote_engine_id"], serde_json::json!("engine-7"));
        assert_eq!(kv["remote_port"], serde_json::json!(20097));
        assert_eq!(kv["tp_size"], serde_json::json!(1));
        assert_eq!(kv["do_remote_prefill"], serde_json::json!(true));
        assert_eq!(kv["remote_block_ids"], serde_json::json!([4, 5, 6]));
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
        let attributes_struct = json_to_prost_struct(&serde_json::json!({
            "remote_engine_id": "engine-7",
        }));
        let req = pb::GenerateRequest {
            kv_session: Some(pb::KvSessionRef {
                attributes_struct,
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
