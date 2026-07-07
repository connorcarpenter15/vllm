use std::path::Path;

use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::protocol::lora::LoraRequest;

use super::pb;
use crate::lora::{LoadExactLoraError, UnloadLoraError};
use crate::state::AppState;

fn ensure_enabled(state: &AppState) -> Result<(), Status> {
    state
        .engine_core_client()
        .ready_response()
        .supports_lora
        .then_some(())
        .ok_or_else(|| Status::failed_precondition("engine was not started with LoRA enabled"))
}

fn ensure_consistent(state: &AppState) -> Result<(), Status> {
    state.lora_state_is_consistent().then_some(()).ok_or_else(|| {
        Status::failed_precondition("LoRA state differs across engine ranks; restart the engine")
    })
}

pub(super) async fn load_lora(
    state: &std::sync::Arc<AppState>,
    request: Request<pb::LoadLoraRequest>,
) -> Result<Response<pb::LoadLoraResponse>, Status> {
    ensure_enabled(state)?;
    ensure_consistent(state)?;
    let adapter = normalize_adapter(
        request
            .into_inner()
            .adapter
            .ok_or_else(|| Status::invalid_argument("adapter is required"))?,
    )
    .await?;
    let _guard = state
        .try_admit_engine_work()
        .ok_or_else(|| Status::unavailable("engine is draining"))?;

    let (adapter, already_loaded) =
        state.load_lora_exact(adapter).await.map_err(|error| match error {
            LoadExactLoraError::Inconsistent => Status::failed_precondition(
                "LoRA state differs across engine ranks; restart the engine",
            ),
            LoadExactLoraError::BaseModelName { lora_name } => Status::already_exists(format!(
                "LoRA adapter `{lora_name}` conflicts with a served base model"
            )),
            LoadExactLoraError::Conflict { existing } => conflict(&existing),
            LoadExactLoraError::Engine(error) => Status::internal(error.to_report_string()),
            LoadExactLoraError::NotLoaded { lora_name } => Status::internal(format!(
                "one or more engine ranks rejected LoRA adapter `{lora_name}`"
            )),
        })?;
    Ok(Response::new(pb::LoadLoraResponse {
        adapter: Some(to_proto(&adapter)),
        already_loaded,
    }))
}

pub(super) async fn unload_lora(
    state: &std::sync::Arc<AppState>,
    request: Request<pb::UnloadLoraRequest>,
) -> Result<Response<pb::UnloadLoraResponse>, Status> {
    ensure_enabled(state)?;
    ensure_consistent(state)?;
    let name = request.into_inner().lora_name;
    if name.trim().is_empty() {
        return Err(Status::invalid_argument("lora_name is required"));
    }

    let adapter = state
        .served_lora_requests()
        .await
        .into_iter()
        .find(|adapter| adapter.lora_name == name)
        .ok_or_else(|| Status::not_found(format!("LoRA adapter `{name}` is not loaded")))?;
    let _guard = state
        .try_admit_engine_work()
        .ok_or_else(|| Status::unavailable("engine is draining"))?;
    let adapter =
        state
            .unload_lora(&name, Some(adapter.lora_int_id))
            .await
            .map_err(|error| match error {
                UnloadLoraError::Inconsistent => Status::failed_precondition(
                    "LoRA state differs across engine ranks; restart the engine",
                ),
                UnloadLoraError::NotFound { lora_name } => {
                    Status::not_found(format!("LoRA adapter `{lora_name}` is not loaded"))
                }
                UnloadLoraError::IntIdMismatch { .. } => {
                    Status::internal("LoRA registry changed during unload")
                }
                UnloadLoraError::Engine(error) => Status::internal(error.to_report_string()),
            })?;
    Ok(Response::new(pb::UnloadLoraResponse {
        adapter: Some(to_proto(&adapter)),
    }))
}

pub(super) async fn list_loras(
    state: &std::sync::Arc<AppState>,
    _request: Request<pb::ListLorasRequest>,
) -> Result<Response<pb::ListLorasResponse>, Status> {
    ensure_enabled(state)?;
    ensure_consistent(state)?;
    let mut adapters = state.served_lora_requests().await;
    adapters.sort_by(|left, right| left.lora_name.cmp(&right.lora_name));
    let adapters = adapters.iter().map(to_proto).collect();
    Ok(Response::new(pb::ListLorasResponse { adapters }))
}

async fn normalize_adapter(adapter: pb::LoraAdapter) -> Result<LoraRequest, Status> {
    if adapter.lora_id <= 0 {
        return Err(Status::invalid_argument("lora_id must be positive"));
    }
    if adapter.lora_name.trim().is_empty() {
        return Err(Status::invalid_argument("lora_name is required"));
    }
    let path = Path::new(&adapter.source_path);
    if !path.is_absolute() {
        return Err(Status::invalid_argument("source_path must be absolute"));
    }
    let canonical = tokio::fs::canonicalize(path)
        .await
        .map_err(|error| Status::invalid_argument(format!("invalid source_path: {error}")))?;
    let metadata = tokio::fs::metadata(&canonical)
        .await
        .map_err(|error| Status::invalid_argument(format!("invalid source_path: {error}")))?;
    if !metadata.is_dir() {
        return Err(Status::invalid_argument("source_path must be a directory"));
    }

    let request = LoraRequest {
        lora_name: adapter.lora_name,
        lora_int_id: u64::try_from(adapter.lora_id)
            .map_err(|_| Status::invalid_argument("lora_id must be positive"))?,
        lora_path: canonical.to_string_lossy().into_owned(),
        base_model_name: None,
        tensorizer_config_dict: None,
        load_inplace: false,
        is_3d_lora_weight: false,
    };
    Ok(request)
}

fn to_proto(adapter: &LoraRequest) -> pb::LoraAdapter {
    pb::LoraAdapter {
        lora_id: adapter.lora_int_id.min(i64::MAX as u64) as i64,
        lora_name: adapter.lora_name.clone(),
        source_path: adapter.lora_path.clone(),
    }
}

fn conflict(existing: &LoraRequest) -> Status {
    Status::already_exists(format!(
        "conflicts with loaded LoRA `{}` (id {})",
        existing.lora_name, existing.lora_int_id
    ))
}
