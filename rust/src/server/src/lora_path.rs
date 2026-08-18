// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::{Component, Path, PathBuf};

pub(crate) const RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV: &str =
    "VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoraPathMode {
    LocalOrRemote,
    LocalDirectory,
}

#[derive(Debug)]
pub(crate) enum LoraPathError {
    MissingAllowedPrefixes,
    MustBeAbsolute,
    InvalidPath { source: std::io::Error },
    InvalidPrefix { source: std::io::Error },
    NotDirectory,
    OutsideAllowedPrefixes,
}

pub(crate) fn runtime_lora_allowed_path_prefixes() -> Option<Vec<PathBuf>> {
    let prefixes = std::env::var_os(RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV)?;
    let prefixes: Vec<_> = std::env::split_paths(&prefixes)
        .filter(|path| !path.as_os_str().is_empty())
        .collect();
    (!prefixes.is_empty()).then_some(prefixes)
}

pub(crate) async fn validate_lora_path_access(
    lora_path: &str,
    allowed_prefixes: Option<&[PathBuf]>,
    mode: LoraPathMode,
) -> Result<Option<String>, LoraPathError> {
    let path = Path::new(lora_path);
    if mode == LoraPathMode::LocalOrRemote
        && !looks_like_local_lora_path(lora_path)
        && !tokio::fs::try_exists(path).await.unwrap_or(false)
    {
        return Ok(None);
    }

    if mode == LoraPathMode::LocalDirectory && !path.is_absolute() {
        return Err(LoraPathError::MustBeAbsolute);
    }

    let canonical_path = if mode == LoraPathMode::LocalDirectory {
        let canonical_path = tokio::fs::canonicalize(path)
            .await
            .map_err(|source| LoraPathError::InvalidPath { source })?;
        let metadata = tokio::fs::metadata(&canonical_path)
            .await
            .map_err(|source| LoraPathError::InvalidPath { source })?;
        if !metadata.is_dir() {
            return Err(LoraPathError::NotDirectory);
        }
        Some(canonical_path)
    } else {
        None
    };

    let Some(allowed_prefixes) = allowed_prefixes else {
        return Err(LoraPathError::MissingAllowedPrefixes);
    };
    if !path.is_absolute() {
        return Err(LoraPathError::MustBeAbsolute);
    }

    let canonical_path = match canonical_path {
        Some(canonical_path) => canonical_path,
        None => tokio::fs::canonicalize(path)
            .await
            .map_err(|source| LoraPathError::InvalidPath { source })?,
    };

    let mut canonical_prefixes = Vec::with_capacity(allowed_prefixes.len());
    for prefix in allowed_prefixes {
        canonical_prefixes.push(
            tokio::fs::canonicalize(prefix)
                .await
                .map_err(|source| LoraPathError::InvalidPrefix { source })?,
        );
    }
    if canonical_prefixes.iter().any(|prefix| canonical_path.starts_with(prefix)) {
        return Ok(Some(canonical_path.to_string_lossy().into_owned()));
    }
    Err(LoraPathError::OutsideAllowedPrefixes)
}

fn looks_like_local_lora_path(lora_path: &str) -> bool {
    let path = Path::new(lora_path);
    path.is_absolute()
        || lora_path.starts_with('~')
        || lora_path.starts_with('.')
        || path.components().any(|component| matches!(component, Component::ParentDir))
}
