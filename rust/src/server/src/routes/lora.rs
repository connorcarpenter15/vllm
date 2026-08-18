// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use serde::Deserialize;
use thiserror_ext::AsReport;
use validator::Validate;

use crate::error::ApiError;
use crate::lora::{LoadLoraError, UnloadLoraError};
use crate::lora_path::{
    LoraPathError, LoraPathMode, RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV,
    runtime_lora_allowed_path_prefixes,
    validate_lora_path_access as validate_shared_lora_path_access,
};
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct LoadLoraAdapterRequest {
    lora_name: String,
    lora_path: String,
    #[serde(default)]
    load_inplace: bool,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

impl Normalizable for LoadLoraAdapterRequest {}

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct UnloadLoraAdapterRequest {
    lora_name: String,
    #[serde(default)]
    lora_int_id: Option<u64>,
}

impl Normalizable for UnloadLoraAdapterRequest {}

async fn validate_lora_path_access(
    lora_path: &str,
    allowed_prefixes: Option<&[PathBuf]>,
) -> Result<Option<String>, ApiError> {
    validate_shared_lora_path_access(lora_path, allowed_prefixes, LoraPathMode::LocalOrRemote)
        .await
        .map_err(|error| match error {
            LoraPathError::MissingAllowedPrefixes => ApiError::invalid_request(
                format!(
                    "Local LoRA adapter paths require {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} to be configured."
                ),
                Some("lora_path"),
            ),
            LoraPathError::MustBeAbsolute => ApiError::invalid_request(
                format!(
                    "Local LoRA adapter paths must be absolute and under one of the prefixes configured by {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV}."
                ),
                Some("lora_path"),
            ),
            LoraPathError::InvalidPath { .. } | LoraPathError::NotDirectory => {
                ApiError::invalid_request(
                    "Local LoRA adapter path must exist and be accessible.".to_string(),
                    Some("lora_path"),
                )
            }
            LoraPathError::InvalidPrefix { .. } => ApiError::server_error(format!(
                "configured {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} path prefix must exist and be accessible"
            )),
            LoraPathError::OutsideAllowedPrefixes => ApiError::invalid_request(
                "Local LoRA adapter path is outside the configured allowed prefixes.".to_string(),
                Some("lora_path"),
            ),
        })
}

/// Dynamically load one LoRA adapter and expose it as an OpenAI model id.
pub async fn load_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<LoadLoraAdapterRequest>,
) -> Result<String, ApiError> {
    if request.lora_name.is_empty() || request.lora_path.is_empty() {
        return Err(ApiError::invalid_request(
            "Both 'lora_name' and 'lora_path' must be provided.".to_string(),
            None,
        ));
    }
    let allowed_prefixes = runtime_lora_allowed_path_prefixes();
    let lora_path = validate_lora_path_access(&request.lora_path, allowed_prefixes.as_deref())
        .await?
        .unwrap_or(request.lora_path);

    let lora_name = request.lora_name;
    state
        .load_lora(
            lora_name.clone(),
            lora_path,
            request.load_inplace,
            request.is_3d_lora_weight,
        )
        .await
        .map_err(|error| match error {
            LoadLoraError::AlreadyLoaded { lora_name } => ApiError::invalid_request(
                format!(
                    "The lora adapter '{lora_name}' has already been loaded. If you want to load the adapter in place, set 'load_inplace' to true."
                ),
                Some("lora_name"),
            ),
            LoadLoraError::BaseModelName { lora_name } => ApiError::invalid_request(
                format!("The lora adapter name '{lora_name}' conflicts with a served base model."),
                Some("lora_name"),
            ),
            LoadLoraError::Engine(error) => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': {}",
                error.to_report_string()
            )),
            LoadLoraError::NotLoaded { lora_name } => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': engine rejected the adapter"
            )),
        })?;

    Ok(format!(
        "Success: LoRA adapter '{lora_name}' added successfully."
    ))
}

/// Remove one LoRA adapter from the engine and frontend registry.
pub async fn unload_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<UnloadLoraAdapterRequest>,
) -> Result<String, ApiError> {
    if request.lora_name.is_empty() {
        return Err(ApiError::invalid_request(
            "'lora_name' needs to be provided to unload a LoRA adapter.".to_string(),
            Some("lora_name"),
        ));
    }

    let lora_request = state
        .unload_lora(&request.lora_name, request.lora_int_id)
        .await
        .map_err(|error| match error {
            UnloadLoraError::NotFound { lora_name } => ApiError::model_not_found(lora_name),
            UnloadLoraError::IntIdMismatch {
                lora_name,
                expected,
                actual,
            } => ApiError::invalid_request(
                format!(
                    "The requested lora_int_id {actual} does not match loaded adapter '{lora_name}' with id {expected}."
                ),
                Some("lora_int_id"),
            ),
            UnloadLoraError::Engine(error) => ApiError::server_error(format!(
                "failed to unload LoRA adapter '{}': {}",
                request.lora_name,
                error.to_report_string()
            )),
            UnloadLoraError::NotRemoved {
                lora_name,
                lora_int_id,
            } => ApiError::server_error(format!(
                "failed to unload LoRA adapter '{lora_name}' with id {lora_int_id}"
            )),
        })?;

    Ok(format!(
        "Success: LoRA adapter '{}' removed successfully.",
        lora_request.lora_name
    ))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::validate_lora_path_access;

    fn temp_lora_dir(test_name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vllm-lora-{test_name}-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp lora dir");
        path
    }

    #[tokio::test]
    async fn lora_path_allows_hf_repo_ids_without_prefixes() {
        assert_eq!(
            validate_lora_path_access("org/adapter-a", None)
                .await
                .expect("hf repo id should be allowed"),
            None
        );
    }

    #[tokio::test]
    async fn lora_path_rejects_local_paths_without_prefixes() {
        assert!(validate_lora_path_access("/tmp/adapter-a", None).await.is_err());
        assert!(validate_lora_path_access("./adapter-a", None).await.is_err());
        assert!(validate_lora_path_access("~/adapter-a", None).await.is_err());
        assert!(validate_lora_path_access("subdir/../../../etc/sensitive", None).await.is_err());
    }

    #[tokio::test]
    async fn lora_path_rejects_existing_bare_relative_paths_without_prefixes() {
        let root =
            PathBuf::from("target").join(format!("vllm-lora-relative-{}", std::process::id()));
        let adapter = root.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create relative adapter dir");

        assert!(
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), None)
                .await
                .is_err()
        );

        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn lora_path_allows_absolute_paths_under_configured_prefixes() {
        let root = temp_lora_dir("allowed-prefix");
        let allowed = root.join("allowed");
        let adapter = allowed.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create adapter dir");

        let prefixes = [allowed];
        let resolved =
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .await
                .expect("path under configured prefix should be allowed");
        assert_eq!(
            resolved.as_deref(),
            Some(
                adapter
                    .canonicalize()
                    .expect("canonical adapter")
                    .to_str()
                    .expect("utf-8 temp path")
            )
        );

        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn lora_path_rejects_parent_escape_from_configured_prefixes() {
        let root = temp_lora_dir("parent-escape");
        let allowed = root.join("allowed");
        let private_adapter = root.join("private").join("adapter-a");
        fs::create_dir_all(&allowed).expect("create allowed dir");
        fs::create_dir_all(&private_adapter).expect("create private adapter dir");

        let escaped = allowed.join("../private/adapter-a");
        let prefixes = [allowed];
        assert!(
            validate_lora_path_access(escaped.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .await
                .is_err()
        );

        fs::remove_dir_all(root).ok();
    }
}
