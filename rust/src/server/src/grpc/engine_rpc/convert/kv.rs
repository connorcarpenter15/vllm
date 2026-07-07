use tonic::Status;
use vllm_text::TextRequest;

use super::super::pb;
use crate::grpc::struct_json::json_to_prost_struct;

pub fn validate_disaggregated_request(
    request: &pb::GenerateRequest,
    role: pb::EngineRole,
    kv_connector: Option<&str>,
    data_parallel_size: u64,
    engine_data_parallel_rank: u32,
) -> Result<u32, Status> {
    if let Some(rank) = request.data_parallel_rank
        && u64::from(rank) >= data_parallel_size
    {
        return Err(Status::invalid_argument(format!(
            "data_parallel_rank {rank} is outside the configured range 0..{data_parallel_size}"
        )));
    }

    match role {
        pb::EngineRole::Decode => {
            let session = request.kv_session.as_ref().ok_or_else(|| {
                Status::invalid_argument("kv_session is required for decode requests")
            })?;
            if session.session_id.trim().is_empty() {
                return Err(Status::invalid_argument(
                    "kv_session.session_id is required",
                ));
            }
            if session.transfer_backend.trim().is_empty() {
                return Err(Status::invalid_argument(
                    "kv_session.transfer_backend is required",
                ));
            }
            if let Some(expected) = kv_connector
                && session.transfer_backend != expected
            {
                return Err(Status::invalid_argument(format!(
                    "kv_session transfer backend `{}` does not match engine connector `{expected}`",
                    session.transfer_backend
                )));
            }
            if session
                .attributes_struct
                .as_ref()
                .is_none_or(|attributes| attributes.fields.is_empty())
            {
                return Err(Status::invalid_argument(
                    "kv_session.attributes_struct must contain transfer metadata",
                ));
            }
        }
        pb::EngineRole::Prefill => {
            if request.kv_session.is_some() {
                return Err(Status::invalid_argument(
                    "kv_session is only valid for decode requests",
                ));
            }
            if data_parallel_size > 1 && request.data_parallel_rank.is_none() {
                return Err(Status::invalid_argument(
                    "data_parallel_rank is required for multi-rank prefill requests",
                ));
            }
        }
        pb::EngineRole::Aggregated | pb::EngineRole::Unspecified => {
            if request.kv_session.is_some() {
                return Err(Status::invalid_argument(
                    "kv_session is only valid for decode requests",
                ));
            }
        }
    }

    Ok(request.data_parallel_rank.unwrap_or(engine_data_parallel_rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_explicit_rank_outside_data_parallel_size() {
        let request = pb::GenerateRequest {
            data_parallel_rank: Some(4),
            ..Default::default()
        };

        let error =
            validate_disaggregated_request(&request, pb::EngineRole::Aggregated, None, 4, 0)
                .unwrap_err();

        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn accepts_last_explicit_rank_in_data_parallel_size() {
        let request = pb::GenerateRequest {
            data_parallel_rank: Some(3),
            ..Default::default()
        };

        assert_eq!(
            validate_disaggregated_request(&request, pb::EngineRole::Aggregated, None, 4, 0,)
                .unwrap(),
            3
        );
    }
}

pub fn role_from_kv_role(kv_role: Option<&str>) -> pb::EngineRole {
    match kv_role {
        Some("kv_producer") => pb::EngineRole::Prefill,
        Some("kv_consumer") => pb::EngineRole::Decode,
        _ => pb::EngineRole::Aggregated,
    }
}

pub fn mark_prefill_request(request: &mut TextRequest) {
    let params = request
        .sampling_params
        .vllm_xargs
        .get_or_insert_with(Default::default)
        .entry("kv_transfer_params".to_string())
        .or_insert_with(|| serde_json::Value::Object(Default::default()));
    if let Some(params) = params.as_object_mut() {
        params.insert(
            "do_remote_decode".to_string(),
            serde_json::Value::Bool(true),
        );
    }
}

pub(super) fn kv_transfer_params_to_kv_session(
    params: &serde_json::Value,
    request_id: &str,
    kv_connector: Option<&str>,
    data_parallel_rank: u32,
) -> pb::KvSessionRef {
    pb::KvSessionRef {
        session_id: request_id.to_string(),
        transfer_backend: kv_connector.unwrap_or_default().to_string(),
        dp_rank: data_parallel_rank,
        attributes_struct: json_to_prost_struct(params),
    }
}
