use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexMap;
use tokio::sync::{Mutex, RwLock};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::lora::LoraRequest;

/// Snapshot of the currently served model names plus the requested LoRA, if
/// the model name resolves to a dynamic adapter.
#[derive(Debug, Clone)]
pub(crate) struct LoraModelResolution {
    pub model_names: Vec<String>,
    pub lora_request: Option<LoraRequest>,
}

/// Runtime registry for dynamically loaded LoRA adapters.
pub(crate) struct LoraManager {
    /// Dynamically loaded LoRA adapters keyed by public model name, in load order.
    requests: RwLock<IndexMap<String, LoraRequest>>,
    /// Monotonic adapter id allocator. LoRA ids are one-indexed.
    id_counter: AtomicU64,
    /// Serialize dynamic LoRA registry updates around engine utility calls.
    update_lock: Mutex<()>,
}

#[derive(Debug)]
pub(crate) enum LoadLoraError {
    AlreadyLoaded { lora_name: String },
    BaseModelName { lora_name: String },
    Engine(vllm_engine_core_client::Error),
    NotLoaded { lora_name: String },
}

#[derive(Debug)]
pub(crate) enum UnloadLoraError {
    NotFound {
        lora_name: String,
    },
    IntIdMismatch {
        lora_name: String,
        expected: u64,
        actual: u64,
    },
    Engine(vllm_engine_core_client::Error),
    NotRemoved {
        lora_name: String,
        lora_int_id: u64,
    },
}

#[derive(Debug)]
pub(crate) enum LoadExactLoraError {
    BaseModelName { lora_name: String },
    Conflict { existing: LoraRequest },
    Engine(vllm_engine_core_client::Error),
    NotLoaded { lora_name: String },
}

impl LoraManager {
    pub fn new() -> Self {
        Self {
            requests: RwLock::new(IndexMap::new()),
            id_counter: AtomicU64::new(0),
            update_lock: Mutex::new(()),
        }
    }

    /// Snapshot loaded LoRA adapters in load order.
    pub async fn served_lora_requests(&self) -> Vec<LoraRequest> {
        self.requests.read().await.values().cloned().collect()
    }

    /// Resolve the requested model against one consistent LoRA registry
    /// snapshot.
    pub async fn resolve_model(
        &self,
        base_model_names: &[String],
        model_name: Option<&str>,
    ) -> LoraModelResolution {
        let requests = self.requests.read().await;
        let mut model_names = base_model_names.to_vec();
        model_names.extend(requests.keys().cloned());
        let lora_request = model_name.and_then(|name| requests.get(name).cloned());

        LoraModelResolution {
            model_names,
            lora_request,
        }
    }

    /// Load one dynamic LoRA adapter and register it as a public model name.
    pub async fn load_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        base_model_names: &[String],
        lora_name: String,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Result<LoraRequest, LoadLoraError> {
        let _guard = self.update_lock.lock().await;
        if base_model_names.iter().any(|name| name == &lora_name) {
            return Err(LoadLoraError::BaseModelName { lora_name });
        }
        if !load_inplace && self.requests.read().await.contains_key(&lora_name) {
            return Err(LoadLoraError::AlreadyLoaded { lora_name });
        }

        let lora_int_id = self
            .requests
            .read()
            .await
            .get(&lora_name)
            .map(|request| request.lora_int_id)
            .unwrap_or_else(|| self.id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        let lora_request = LoraRequest::new(
            lora_name.clone(),
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        );

        let loaded = engine_core_client
            .add_lora(&lora_request)
            .await
            .map_err(LoadLoraError::Engine)?;
        if !loaded {
            return Err(LoadLoraError::NotLoaded { lora_name });
        }
        self.requests.write().await.insert(lora_name, lora_request.clone());
        Ok(lora_request)
    }

    /// Load an adapter with a caller-supplied ID.
    pub async fn load_lora_exact(
        &self,
        engine_core_client: &EngineCoreClient,
        base_model_names: &[String],
        lora_request: LoraRequest,
    ) -> Result<(LoraRequest, bool), LoadExactLoraError> {
        let _guard = self.update_lock.lock().await;
        if base_model_names.iter().any(|name| name == &lora_request.lora_name) {
            return Err(LoadExactLoraError::BaseModelName {
                lora_name: lora_request.lora_name,
            });
        }

        let requests = self.requests.read().await;
        if let Some(existing) = requests.values().find(|existing| {
            existing.lora_name == lora_request.lora_name
                || existing.lora_int_id == lora_request.lora_int_id
                || existing.lora_path == lora_request.lora_path
        }) {
            if existing == &lora_request {
                return Ok((existing.clone(), true));
            }
            return Err(LoadExactLoraError::Conflict {
                existing: existing.clone(),
            });
        }
        drop(requests);

        let results =
            match engine_core_client.call_utility::<bool, _>("add_lora", (&lora_request,)).await {
                Ok(results) => results,
                Err(error) => {
                    let _ = engine_core_client
                        .call_utility::<bool, _>("remove_lora", (lora_request.lora_int_id,))
                        .await;
                    return Err(LoadExactLoraError::Engine(error));
                }
            };
        if !results.iter().all(|loaded| *loaded) {
            let _ = engine_core_client
                .call_utility::<bool, _>("remove_lora", (lora_request.lora_int_id,))
                .await;
            return Err(LoadExactLoraError::NotLoaded {
                lora_name: lora_request.lora_name,
            });
        }

        self.id_counter.fetch_max(lora_request.lora_int_id, Ordering::Relaxed);
        self.requests
            .write()
            .await
            .insert(lora_request.lora_name.clone(), lora_request.clone());
        Ok((lora_request, false))
    }

    /// Remove one dynamic LoRA adapter from the engine and public model
    /// registry.
    pub async fn unload_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        lora_name: &str,
        requested_lora_int_id: Option<u64>,
    ) -> Result<LoraRequest, UnloadLoraError> {
        let _guard = self.update_lock.lock().await;
        let lora_request = self.requests.read().await.get(lora_name).cloned().ok_or_else(|| {
            UnloadLoraError::NotFound {
                lora_name: lora_name.to_string(),
            }
        })?;

        if let Some(actual) = requested_lora_int_id
            && actual != lora_request.lora_int_id
        {
            return Err(UnloadLoraError::IntIdMismatch {
                lora_name: lora_name.to_string(),
                expected: lora_request.lora_int_id,
                actual,
            });
        }

        let removed = match engine_core_client
            .call_utility::<bool, _>("remove_lora", (lora_request.lora_int_id,))
            .await
        {
            Ok(results) => results.iter().all(|removed| *removed),
            Err(error) => {
                let _ =
                    engine_core_client.call_utility::<bool, _>("add_lora", (&lora_request,)).await;
                return Err(UnloadLoraError::Engine(error));
            }
        };
        if !removed {
            let _ = engine_core_client.call_utility::<bool, _>("add_lora", (&lora_request,)).await;
            return Err(UnloadLoraError::NotRemoved {
                lora_name: lora_request.lora_name,
                lora_int_id: lora_request.lora_int_id,
            });
        }

        Ok(self.requests.write().await.shift_remove(lora_name).unwrap_or(lora_request))
    }
}
