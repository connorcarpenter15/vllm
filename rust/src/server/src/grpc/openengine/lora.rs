use std::path::Path;
use std::sync::Arc;

use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use tracing::warn;
use vllm_engine_core_client::protocol::LoraRequest;

use super::pb;
use crate::state::AppState;

pub struct LoraManagerServiceImpl {
    state: Arc<AppState>,
}

impl LoraManagerServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    fn ensure_enabled(&self) -> Result<(), Status> {
        self.state
            .engine_core_client()
            .ready_response()
            .supports_lora
            .then_some(())
            .ok_or_else(|| Status::failed_precondition("engine was not started with LoRA enabled"))
    }
}

#[tonic::async_trait]
impl pb::lora_manager_server::LoraManager for LoraManagerServiceImpl {
    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        self.ensure_enabled()?;
        let adapter = normalize_adapter(
            request
                .into_inner()
                .adapter
                .ok_or_else(|| Status::invalid_argument("adapter is required"))?,
        )
        .await?;

        let registry = self.state.loras();
        let _guard = registry.lifecycle.lock().await;

        if let Some(existing) = registry.get(&adapter.lora_name).await {
            if existing == adapter {
                return Ok(Response::new(pb::LoadLoraResponse {
                    adapter: Some(to_proto(&existing)),
                    already_loaded: true,
                }));
            }
            return Err(conflict(&existing));
        }
        if let Some(existing) = registry.conflicting_adapter(&adapter).await {
            return Err(conflict(&existing));
        }

        let results = self
            .state
            .engine_core_client()
            .call_utility::<bool, _>("add_lora", (adapter.clone(),))
            .await;
        match results {
            Ok(results) if results.iter().all(|loaded| *loaded) => {}
            Ok(_) => {
                rollback_remove(&self.state, adapter.lora_int_id).await;
                return Err(Status::internal(
                    "one or more engine ranks rejected the LoRA",
                ));
            }
            Err(error) => {
                rollback_remove(&self.state, adapter.lora_int_id).await;
                return Err(Status::internal(error.to_report_string()));
            }
        }

        registry.insert(adapter.clone()).await;
        Ok(Response::new(pb::LoadLoraResponse {
            adapter: Some(to_proto(&adapter)),
            already_loaded: false,
        }))
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        self.ensure_enabled()?;
        let name = request.into_inner().lora_name;
        if name.trim().is_empty() {
            return Err(Status::invalid_argument("lora_name is required"));
        }

        let registry = self.state.loras();
        let _guard = registry.lifecycle.lock().await;
        let adapter = registry
            .get(&name)
            .await
            .ok_or_else(|| Status::not_found(format!("LoRA adapter `{name}` is not loaded")))?;

        let results = self
            .state
            .engine_core_client()
            .call_utility::<bool, _>("remove_lora", (adapter.lora_int_id,))
            .await;
        match results {
            Ok(results) if results.iter().all(|removed| *removed) => {}
            Ok(_) => {
                rollback_add(&self.state, &adapter).await;
                return Err(Status::internal(
                    "one or more engine ranks rejected the LoRA unload",
                ));
            }
            Err(error) => {
                rollback_add(&self.state, &adapter).await;
                return Err(Status::internal(error.to_report_string()));
            }
        }

        registry.remove(&name).await;
        Ok(Response::new(pb::UnloadLoraResponse {
            adapter: Some(to_proto(&adapter)),
        }))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        self.ensure_enabled()?;
        let adapters = self.state.loras().list().await.iter().map(to_proto).collect();
        Ok(Response::new(pb::ListLorasResponse { adapters }))
    }
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
        lora_int_id: adapter.lora_id,
        lora_path: canonical.to_string_lossy().into_owned(),
        base_model_name: None,
        tensorizer_config_dict: None,
        load_inplace: false,
        is_3d_lora_weight: false,
    };
    request.validate().map_err(Status::invalid_argument)?;
    Ok(request)
}

fn to_proto(adapter: &LoraRequest) -> pb::LoraAdapter {
    pb::LoraAdapter {
        lora_id: adapter.lora_int_id,
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

async fn rollback_remove(state: &AppState, id: i64) {
    if let Err(error) =
        state.engine_core_client().call_utility::<bool, _>("remove_lora", (id,)).await
    {
        warn!(error = %error.to_report_string(), id, "failed to roll back LoRA load");
    }
}

async fn rollback_add(state: &AppState, adapter: &LoraRequest) {
    if let Err(error) = state
        .engine_core_client()
        .call_utility::<bool, _>("add_lora", (adapter.clone(),))
        .await
    {
        warn!(
            error = %error.to_report_string(),
            lora_name = %adapter.lora_name,
            "failed to roll back LoRA unload"
        );
    }
}
