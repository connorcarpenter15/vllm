// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use openengine_proto::openengine::v1 as pb;
use tonic::{Request, Status};
use vllm_chat::MediaContentPart;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputBackend, StructuredOutputsParams,
};
use vllm_text::{
    DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent, FinishReason, Finished, Prompt,
    SamplingParams, TextDecodeOptions, TextRequest,
};

use super::struct_json::{
    json_to_prost_struct, prost_handoff_struct_to_json, prost_struct_to_json,
};

pub(super) const HANDOFF_PROFILE: &str = "vllm.kv_transfer_params.v1";
pub(super) const TRANSFER_BACKEND: &str = "nixl";
const TARGET_DP_RANK: &str = "openengine-target-dp-rank";
const PRIORITY: &str = "openengine-priority";

pub(super) fn role_from_kv_role(kv_role: Option<&str>) -> Option<pb::EngineRole> {
    match kv_role {
        Some("kv_producer") => Some(pb::EngineRole::Prefill),
        Some("kv_consumer") => Some(pb::EngineRole::Decode),
        None => Some(pb::EngineRole::Aggregated),
        Some(_) => None,
    }
}

pub(super) fn metadata_options<T>(request: &Request<T>) -> Result<(Option<u32>, i32), Status> {
    let rank = request
        .metadata()
        .get(TARGET_DP_RANK)
        .map(|value| {
            value
                .to_str()
                .map_err(|_| Status::invalid_argument("openengine-target-dp-rank is not ASCII"))?
                .parse::<u32>()
                .map_err(|_| {
                    Status::invalid_argument("openengine-target-dp-rank must be a base-10 uint32")
                })
        })
        .transpose()?;
    let portable_priority = request
        .metadata()
        .get(PRIORITY)
        .map(|value| {
            value
                .to_str()
                .map_err(|_| Status::invalid_argument("openengine-priority is not ASCII"))?
                .parse::<i32>()
                .map_err(|_| {
                    Status::invalid_argument("openengine-priority must be a base-10 int32")
                })
        })
        .transpose()?
        .unwrap_or_default();
    let priority = portable_priority.checked_neg().ok_or_else(|| {
        Status::invalid_argument("openengine-priority i32::MIN cannot map to vLLM priority")
    })?;
    Ok((rank, priority))
}

pub(super) async fn to_text_request(
    mut request: pb::GenerateRequest,
    role: pb::EngineRole,
    target_dp_rank: Option<u32>,
    priority: i32,
    data_parallel_size: u64,
    state: &crate::state::AppState,
) -> Result<TextRequest, Status> {
    validate_model(
        &request.model,
        state.chat.model_id(),
        state.served_model_names(),
    )?;
    validate_role_and_handoff(&request, role, target_dp_rank, data_parallel_size)?;

    if request.media_options.as_ref().is_some_and(|options| !options.fields.is_empty()) {
        return Err(Status::invalid_argument(
            "per-request media_options are not supported by the vLLM Rust frontend",
        ));
    }

    let mut prompt = match request.input.take() {
        Some(pb::generate_request::Input::Prompt(prompt)) => Prompt::Text(prompt),
        Some(pb::generate_request::Input::TokenIds(token_ids)) => Prompt::TokenIds(token_ids.ids),
        None => return Err(Status::invalid_argument("prompt or token_ids is required")),
    };

    let mm_features = if request.media.is_empty() {
        None
    } else {
        tokenize_multimodal_prompt(&mut prompt, state.chat.text().tokenizer().as_ref())?;
        let Prompt::TokenIds(token_ids) = &mut prompt else {
            unreachable!("multimodal text prompts are tokenized above");
        };
        let parts = media_parts(&request.media)?;
        state
            .chat
            .prepare_media(parts, token_ids)
            .await
            .map_err(|error| Status::invalid_argument(error.to_string()))?
    };

    let mut sampling_params = sampling_params(&request)?;
    let kv = request.kv.as_ref();
    if kv.and_then(|kv| kv.bypass_prefix_cache) == Some(true) {
        sampling_params.skip_reading_prefix_cache = Some(true);
    }
    if let Some(extra) = request.extra.as_ref().filter(|extra| !extra.fields.is_empty()) {
        let serde_json::Value::Object(extra) = prost_struct_to_json(extra) else {
            unreachable!("protobuf Struct always converts to an object")
        };
        sampling_params.vllm_xargs.get_or_insert_with(Default::default).extend(extra);
    }
    match role {
        pb::EngineRole::Prefill => {
            sampling_params.vllm_xargs.get_or_insert_with(Default::default).insert(
                "kv_transfer_params".to_string(),
                serde_json::json!({"do_remote_decode": true}),
            );
        }
        pb::EngineRole::Decode => {
            let attributes = request
                .kv
                .as_ref()
                .and_then(|kv| kv.session.as_ref())
                .and_then(|session| session.attributes_struct.as_ref())
                .expect("validated decode request has transfer attributes");
            sampling_params.vllm_xargs.get_or_insert_with(Default::default).insert(
                "kv_transfer_params".to_string(),
                prost_handoff_struct_to_json(attributes),
            );
        }
        _ => {}
    }

    let stopping = request.stopping.as_ref();
    let stop_strings = stopping
        .map(|options| {
            options
                .conditions
                .iter()
                .filter_map(|condition| match condition.condition.as_ref() {
                    Some(pb::stop_condition::Condition::StopText(text)) => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty());

    Ok(TextRequest {
        request_id: if request.request_id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            request.request_id
        },
        prompt,
        mm_features,
        sampling_params,
        decode_options: TextDecodeOptions {
            skip_special_tokens: true,
            include_stop_str_in_output: stopping
                .and_then(|options| options.include_stop_in_output)
                .unwrap_or(false),
            stop_strings,
            min_tokens: stopping.and_then(|options| options.min_tokens).unwrap_or_default(),
        },
        intermediate: true,
        priority,
        cache_salt: kv.and_then(|kv| kv.cache_salt.clone()),
        add_special_tokens: true,
        data_parallel_rank: target_dp_rank,
        reasoning_parser_kwargs: None,
        lora_request: None,
        arrival_time: None,
    })
}

fn tokenize_multimodal_prompt(
    prompt: &mut Prompt,
    tokenizer: &dyn vllm_text::tokenizer::Tokenizer,
) -> Result<(), Status> {
    if let Prompt::Text(text) = prompt {
        let token_ids = tokenizer.encode(text, true).map_err(|error| {
            Status::invalid_argument(format!("failed to tokenize multimodal prompt: {error}"))
        })?;
        *prompt = Prompt::TokenIds(token_ids);
    }
    Ok(())
}

fn validate_model(model: &str, canonical: &str, served: &[String]) -> Result<(), Status> {
    if model.is_empty() || model == canonical || served.iter().any(|served| served == model) {
        Ok(())
    } else {
        Err(Status::not_found(format!("model `{model}` not found")))
    }
}

fn validate_role_and_handoff(
    request: &pb::GenerateRequest,
    role: pb::EngineRole,
    target_dp_rank: Option<u32>,
    data_parallel_size: u64,
) -> Result<(), Status> {
    if target_dp_rank.is_some_and(|rank| u64::from(rank) >= data_parallel_size) {
        return Err(Status::invalid_argument(format!(
            "target data-parallel rank is outside 0..{data_parallel_size}"
        )));
    }
    let session = request.kv.as_ref().and_then(|kv| kv.session.as_ref());
    match role {
        pb::EngineRole::Decode => {
            let session = session
                .ok_or_else(|| Status::failed_precondition("decode requests require kv.session"))?;
            if session.session_id.trim().is_empty() {
                return Err(Status::invalid_argument(
                    "kv.session.session_id is required",
                ));
            }
            if session.handoff_profile != HANDOFF_PROFILE {
                return Err(Status::invalid_argument(format!(
                    "unsupported handoff profile `{}`; expected `{HANDOFF_PROFILE}`",
                    session.handoff_profile
                )));
            }
            if session.transfer_backend != TRANSFER_BACKEND {
                return Err(Status::invalid_argument(format!(
                    "unsupported transfer backend `{}`; expected `{TRANSFER_BACKEND}`",
                    session.transfer_backend
                )));
            }
            if session
                .attributes_struct
                .as_ref()
                .is_none_or(|attributes| attributes.fields.is_empty())
            {
                return Err(Status::invalid_argument(
                    "kv.session.attributes_struct must contain NIXL transfer parameters",
                ));
            }
        }
        pb::EngineRole::Prefill => {
            if session.is_some() {
                return Err(Status::invalid_argument(
                    "prefill requests must not contain a KV session",
                ));
            }
            if data_parallel_size > 1 && target_dp_rank.is_none() {
                return Err(Status::invalid_argument(
                    "multi-rank prefill requests require openengine-target-dp-rank",
                ));
            }
        }
        pb::EngineRole::Aggregated | pb::EngineRole::Unspecified => {
            if session.is_some() {
                return Err(Status::invalid_argument(
                    "aggregated requests must not contain a KV session",
                ));
            }
        }
    }
    Ok(())
}

fn sampling_params(request: &pb::GenerateRequest) -> Result<SamplingParams, Status> {
    let sampling = request.sampling.as_ref();
    if sampling.and_then(|sampling| sampling.num_sequences).unwrap_or(1) != 1 {
        return Err(Status::invalid_argument(
            "vLLM OpenEngine currently supports exactly one output sequence",
        ));
    }
    let top_k = match sampling.and_then(|sampling| sampling.top_k) {
        None => None,
        Some(-1 | 0) => Some(0),
        Some(value) if value > 0 => Some(value as u32),
        Some(value) => {
            return Err(Status::invalid_argument(format!(
                "top_k must be -1, 0, or positive; got {value}"
            )));
        }
    };
    let seed = sampling
        .and_then(|sampling| sampling.seed)
        .map(|seed| {
            i64::try_from(seed)
                .map_err(|_| Status::invalid_argument("seed exceeds vLLM's signed 64-bit range"))
        })
        .transpose()?;
    let stopping = request.stopping.as_ref();
    let mut stop_token_ids = Vec::new();
    if let Some(stopping) = stopping {
        for condition in &stopping.conditions {
            match condition.condition.as_ref() {
                Some(pb::stop_condition::Condition::StopText(_)) => {}
                Some(pb::stop_condition::Condition::StopTokenId(token_id)) => {
                    stop_token_ids.push(*token_id)
                }
                None => {
                    return Err(Status::invalid_argument(
                        "every stop condition must set stop_text or stop_token_id",
                    ));
                }
            }
        }
    }
    let response = request.response.as_ref();
    if response.and_then(|response| response.prompt_logprob_start).unwrap_or_default() != 0 {
        return Err(Status::invalid_argument(
            "prompt_logprob_start is not supported by vLLM",
        ));
    }
    let (prompt_logprobs, _) = candidate_selection(
        response.and_then(|response| response.prompt_candidates.as_ref()),
        response.and_then(|response| response.return_prompt_logprobs) == Some(true),
        false,
    )?;
    let (logprobs, logprob_token_ids) = candidate_selection(
        response.and_then(|response| response.output_candidates.as_ref()),
        response.and_then(|response| response.return_output_logprobs) == Some(true),
        true,
    )?;

    Ok(SamplingParams {
        temperature: sampling.and_then(|sampling| sampling.temperature).map(|value| value as f32),
        top_p: sampling.and_then(|sampling| sampling.top_p).map(|value| value as f32),
        top_k,
        seed,
        max_tokens: stopping.and_then(|stopping| stopping.max_tokens),
        min_tokens: stopping.and_then(|stopping| stopping.min_tokens),
        logprobs,
        prompt_logprobs,
        min_p: sampling.and_then(|sampling| sampling.min_p).map(|value| value as f32),
        frequency_penalty: sampling
            .and_then(|sampling| sampling.frequency_penalty)
            .map(|value| value as f32),
        presence_penalty: sampling
            .and_then(|sampling| sampling.presence_penalty)
            .map(|value| value as f32),
        repetition_penalty: sampling
            .and_then(|sampling| sampling.repetition_penalty)
            .map(|value| value as f32),
        stop_token_ids: (!stop_token_ids.is_empty()).then_some(stop_token_ids),
        ignore_eos: stopping.and_then(|stopping| stopping.ignore_eos).unwrap_or(false),
        logprob_token_ids,
        structured_outputs: request.guided.as_ref().map(guided_decoding).transpose()?,
        ..SamplingParams::default()
    })
}

fn candidate_selection(
    selection: Option<&pb::CandidateTokenSelection>,
    enabled: bool,
    token_ids_supported: bool,
) -> Result<(Option<i32>, Option<Vec<u32>>), Status> {
    if !enabled {
        return Ok((None, None));
    }
    use pb::candidate_token_selection::Selection;
    match selection.and_then(|selection| selection.selection.as_ref()) {
        None => Ok((Some(1), None)),
        Some(Selection::TopN(top_n)) => Ok((
            Some(i32::try_from(*top_n).map_err(|_| {
                Status::invalid_argument("logprob top_n exceeds signed 32-bit range")
            })?),
            None,
        )),
        Some(Selection::All(_)) => Ok((Some(-1), None)),
        Some(Selection::TokenIds(token_ids)) if token_ids_supported => {
            Ok((Some(1), Some(token_ids.ids.clone())))
        }
        Some(Selection::TokenIds(_)) => Err(Status::invalid_argument(
            "prompt logprobs do not support token-id candidate selection",
        )),
    }
}

fn guided_decoding(guided: &pb::GuidedDecoding) -> Result<StructuredOutputsParams, Status> {
    use pb::guided_decoding::Guide;
    let mut params = match guided.guide.as_ref() {
        Some(Guide::JsonSchema(schema)) => {
            StructuredOutputsParams::json(serde_json::from_str(schema).map_err(|error| {
                Status::invalid_argument(format!("invalid JSON schema: {error}"))
            })?)
        }
        Some(Guide::Regex(regex)) => StructuredOutputsParams::regex(regex.clone()),
        Some(Guide::EbnfGrammar(grammar)) => StructuredOutputsParams::grammar(grammar.clone()),
        Some(Guide::StructuralTag(tag)) => StructuredOutputsParams::structural_tag(tag.clone()),
        Some(Guide::Choice(choice)) => StructuredOutputsParams::choice(choice.choices.clone()),
        Some(Guide::JsonObject(_)) => StructuredOutputsParams::json_object(),
        None => {
            return Err(Status::invalid_argument(
                "guided decoding constraint is required",
            ));
        }
    };
    params.backend = match guided.backend.as_str() {
        "" | "guidance" | "llguidance" => StructuredOutputBackend::Guidance,
        "xgrammar" => StructuredOutputBackend::Xgrammar,
        "outlines" => StructuredOutputBackend::Outlines,
        "lm-format-enforcer" => StructuredOutputBackend::LmFormatEnforcer,
        backend => {
            return Err(Status::invalid_argument(format!(
                "unsupported guided decoding backend `{backend}`"
            )));
        }
    };
    Ok(params)
}

fn media_parts(media: &[pb::MediaItem]) -> Result<Vec<MediaContentPart>, Status> {
    media
        .iter()
        .map(|item| {
            if pb::Modality::try_from(item.modality).ok() != Some(pb::Modality::Image) {
                return Err(Status::invalid_argument(
                    "vLLM OpenEngine currently supports image media only",
                ));
            }
            let uuid = (!item.uuid.is_empty()).then(|| item.uuid.clone());
            match item.source.as_ref() {
                Some(pb::media_item::Source::Url(url))
                | Some(pb::media_item::Source::DataUri(url)) => Ok(MediaContentPart::ImageUrl {
                    url: url.clone(),
                    detail: None,
                    uuid,
                }),
                Some(pb::media_item::Source::RawBytes(data)) => Ok(MediaContentPart::ImageData {
                    data: data.clone(),
                    mime_type: (!item.mime_type.is_empty()).then(|| item.mime_type.clone()),
                    uuid,
                    detail: None,
                }),
                None => Err(Status::invalid_argument("media item source is required")),
            }
        })
        .collect()
}

pub(super) fn event_to_responses(
    event: DecodedTextEvent,
    request_id: &str,
    role: pb::EngineRole,
    handoff_dp_rank: u32,
) -> Vec<pb::GenerateResponse> {
    match event {
        DecodedTextEvent::Start {
            prompt_token_ids,
            prompt_logprobs,
        } => vec![pb::GenerateResponse {
            request_id: request_id.to_string(),
            event: Some(pb::generate_response::Event::Prompt(pb::PromptOutput {
                tokens: prompt_tokens(&prompt_token_ids, prompt_logprobs.as_ref()),
            })),
            usage: None,
        }],
        DecodedTextEvent::TextDelta {
            delta,
            token_ids,
            logprobs,
            finished,
        } => {
            let mut responses = Vec::with_capacity(2);
            // A prefill request generates one token only to finalize the KV
            // handoff. That internal token must not escape on the prefill-only
            // OpenEngine stream; the decode request produces the user-visible
            // output after importing the transferred prompt KV.
            if role != pb::EngineRole::Prefill && (!delta.is_empty() || !token_ids.is_empty()) {
                responses.push(pb::GenerateResponse {
                    request_id: request_id.to_string(),
                    event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                        output_index: Some(0),
                        tokens: output_tokens(&token_ids, logprobs.as_ref()),
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
                    handoff_dp_rank,
                ));
            }
            responses
        }
    }
}

fn prompt_tokens(ids: &[u32], logprobs: Option<&DecodedPromptLogprobs>) -> Vec<pb::TokenInfo> {
    ids.iter()
        .enumerate()
        .map(|(index, id)| {
            if index == 0 {
                return pb::TokenInfo {
                    token_id: *id,
                    token: logprobs
                        .map(|logprobs| logprobs.first_token.clone())
                        .unwrap_or_default(),
                    ..Default::default()
                };
            }
            token_info(
                *id,
                logprobs.and_then(|logprobs| logprobs.scored_positions.get(index - 1)),
            )
        })
        .collect()
}

fn output_tokens(ids: &[u32], logprobs: Option<&DecodedLogprobs>) -> Vec<pb::TokenInfo> {
    ids.iter()
        .enumerate()
        .map(|(index, id)| {
            token_info(
                *id,
                logprobs.and_then(|logprobs| logprobs.positions.get(index)),
            )
        })
        .collect()
}

fn token_info(id: u32, position: Option<&vllm_text::DecodedPositionLogprobs>) -> pb::TokenInfo {
    let selected = position.and_then(|position| {
        position
            .entries
            .iter()
            .find(|entry| entry.token_id == id)
            .or(position.entries.first())
    });
    pb::TokenInfo {
        token_id: id,
        token: selected.map(|entry| entry.token.clone()).unwrap_or_default(),
        logprob: selected.map(|entry| f64::from(entry.logprob)),
        rank: selected.map(|entry| entry.rank),
        candidates: position
            .map(|position| {
                position
                    .entries
                    .iter()
                    .map(|entry| pb::LogProb {
                        token_id: entry.token_id,
                        logprob: f64::from(entry.logprob),
                        token: entry.token.clone(),
                        rank: Some(entry.rank),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn terminal_response(
    finished: &Finished,
    request_id: &str,
    role: pb::EngineRole,
    handoff_dp_rank: u32,
) -> pb::GenerateResponse {
    let usage = Some(usage(finished));
    if finished.finish_reason == FinishReason::Error {
        return engine_error(
            request_id,
            pb::ErrorCode::Internal,
            "generation failed in vLLM engine core".to_string(),
            usage,
        );
    }
    if role == pb::EngineRole::Prefill && finished.finish_reason == FinishReason::Abort {
        return engine_error(
            request_id,
            pb::ErrorCode::Cancelled,
            "prefill request was cancelled before handoff".to_string(),
            usage,
        );
    }
    if role == pb::EngineRole::Prefill {
        let Some(attributes) = finished.kv_transfer_params.as_ref().and_then(json_to_prost_struct)
        else {
            return engine_error(
                request_id,
                pb::ErrorCode::KvTransferFailed,
                "prefill completed without valid NIXL handoff metadata".to_string(),
                usage,
            );
        };
        return pb::GenerateResponse {
            request_id: request_id.to_string(),
            event: Some(pb::generate_response::Event::PrefillReady(
                pb::PrefillReady {
                    kv_session: Some(pb::KvSessionRef {
                        session_id: request_id.to_string(),
                        transfer_backend: TRANSFER_BACKEND.to_string(),
                        endpoints: Vec::new(),
                        dp_rank: handoff_dp_rank,
                        attributes_struct: Some(attributes),
                        handoff_profile: HANDOFF_PROFILE.to_string(),
                        bootstrap: None,
                    }),
                },
            )),
            usage,
        };
    }

    let (reason, stop_match) = finish_reason(&finished.finish_reason);
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: reason as i32,
                message: String::new(),
                stop_match,
            },
        )),
        usage,
    }
}

fn finish_reason(reason: &FinishReason) -> (pb::FinishReason, Option<pb::StopMatch>) {
    match reason {
        FinishReason::Stop(Some(StopReason::TokenId(token_id)))
        | FinishReason::Repetition(Some(StopReason::TokenId(token_id))) => (
            pb::FinishReason::Stop,
            Some(pb::StopMatch {
                r#match: Some(pb::stop_match::Match::StopTokenId(*token_id)),
            }),
        ),
        FinishReason::Stop(Some(StopReason::Text(text)))
        | FinishReason::Repetition(Some(StopReason::Text(text))) => (
            pb::FinishReason::Stop,
            Some(pb::StopMatch {
                r#match: Some(pb::stop_match::Match::StopText(text.clone())),
            }),
        ),
        FinishReason::Stop(None) | FinishReason::Repetition(None) => (pb::FinishReason::Stop, None),
        FinishReason::Length => (pb::FinishReason::Length, None),
        FinishReason::Abort => (pb::FinishReason::Cancelled, None),
        FinishReason::Error => unreachable!("engine failures are emitted as EngineError"),
    }
}

fn usage(finished: &Finished) -> pb::Usage {
    let prompt = finished.usage.prompt_token_count as u32;
    let completion = finished.usage.output_token_count as u32;
    pb::Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt.saturating_add(completion),
        cached_prompt_tokens: Some(finished.usage.cached_token_count as u32),
        reasoning_tokens: None,
    }
}

pub(super) fn engine_error(
    request_id: &str,
    code: pb::ErrorCode,
    message: String,
    usage: Option<pb::Usage>,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Error(pb::EngineError {
            code: code as i32,
            message,
            retryable: false,
        })),
        usage,
    }
}

#[cfg(test)]
mod tests {
    use prost_types::value::Kind;
    use tonic::metadata::MetadataValue;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    fn decode_request(session: Option<pb::KvSessionRef>) -> pb::GenerateRequest {
        pb::GenerateRequest {
            kv: Some(pb::KvOptions {
                session,
                bypass_prefix_cache: None,
                cache_salt: None,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn direct_decode_requires_profiled_handoff() {
        let error =
            validate_role_and_handoff(&decode_request(None), pb::EngineRole::Decode, None, 1)
                .unwrap_err();
        assert_eq!(error.code(), tonic::Code::FailedPrecondition);

        let request = decode_request(Some(pb::KvSessionRef {
            session_id: "session".to_string(),
            transfer_backend: TRANSFER_BACKEND.to_string(),
            attributes_struct: Some(prost_types::Struct {
                fields: [(
                    "remote_port".to_string(),
                    prost_types::Value {
                        kind: Some(Kind::NumberValue(1234.0)),
                    },
                )]
                .into(),
            }),
            handoff_profile: "wrong.profile".to_string(),
            ..Default::default()
        }));
        let error =
            validate_role_and_handoff(&request, pb::EngineRole::Decode, None, 1).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn portable_priority_is_inverted_for_vllm() {
        let mut request = Request::new(());
        request.metadata_mut().insert(
            PRIORITY,
            MetadataValue::try_from("7").expect("valid metadata"),
        );
        request.metadata_mut().insert(
            TARGET_DP_RANK,
            MetadataValue::try_from("2").expect("valid metadata"),
        );
        assert_eq!(metadata_options(&request).unwrap(), (Some(2), -7));
    }

    #[test]
    fn multimodal_text_prompt_uses_shared_frontend_tokenizer() {
        let mut prompt = Prompt::Text("image prompt".to_string());
        tokenize_multimodal_prompt(&mut prompt, &TestTokenizer::new()).unwrap();
        let Prompt::TokenIds(token_ids) = prompt else {
            panic!("multimodal prompt was not tokenized");
        };
        assert!(!token_ids.is_empty());
    }

    #[test]
    fn unsupported_kv_role_is_not_advertised_as_aggregate() {
        assert_eq!(role_from_kv_role(None), Some(pb::EngineRole::Aggregated));
        assert_eq!(
            role_from_kv_role(Some("kv_producer")),
            Some(pb::EngineRole::Prefill)
        );
        assert_eq!(
            role_from_kv_role(Some("kv_consumer")),
            Some(pb::EngineRole::Decode)
        );
        assert_eq!(role_from_kv_role(Some("kv_both")), None);
    }

    #[test]
    fn prefill_handoff_preserves_profile_and_large_integer() {
        let finished = Finished {
            usage: vllm_llm::TokenUsage {
                prompt_token_count: 4,
                output_token_count: 1,
                cached_token_count: 0,
            },
            finish_reason: FinishReason::Length,
            kv_transfer_params: Some(serde_json::json!({
                "remote_request_id": 9_007_199_254_740_993_u64,
                "remote_port": 1234,
            })),
            ec_transfer_params: None,
        };
        let responses = event_to_responses(
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: Vec::new(),
                logprobs: None,
                finished: Some(finished),
            },
            "request",
            pb::EngineRole::Prefill,
            3,
        );
        let pb::generate_response::Event::PrefillReady(ready) =
            responses[0].event.as_ref().unwrap()
        else {
            panic!("expected prefill ready")
        };
        let session = ready.kv_session.as_ref().unwrap();
        assert_eq!(session.handoff_profile, HANDOFF_PROFILE);
        assert_eq!(session.transfer_backend, TRANSFER_BACKEND);
        assert_eq!(session.dp_rank, 3);
        let value = session
            .attributes_struct
            .as_ref()
            .unwrap()
            .fields
            .get("remote_request_id")
            .unwrap();
        assert_eq!(
            value.kind,
            Some(Kind::StringValue("9007199254740993".to_string()))
        );
    }

    #[test]
    fn prefill_handoff_suppresses_internal_generated_token() {
        let finished = Finished {
            usage: vllm_llm::TokenUsage {
                prompt_token_count: 4,
                output_token_count: 1,
                cached_token_count: 0,
            },
            finish_reason: FinishReason::Length,
            kv_transfer_params: Some(serde_json::json!({
                "remote_engine_id": "prefill",
            })),
            ec_transfer_params: None,
        };
        let responses = event_to_responses(
            DecodedTextEvent::TextDelta {
                delta: " internal".to_string(),
                token_ids: vec![42],
                logprobs: None,
                finished: Some(finished),
            },
            "request",
            pb::EngineRole::Prefill,
            0,
        );

        assert_eq!(responses.len(), 1);
        assert!(matches!(
            responses[0].event,
            Some(pb::generate_response::Event::PrefillReady(_))
        ));
    }
}
