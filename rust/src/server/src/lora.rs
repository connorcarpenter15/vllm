// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use futures::Stream;
use indexmap::IndexMap;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, RwLock};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::lora::LoraRequest;

pub(crate) type LoraLease = Option<OwnedRwLockReadGuard<()>>;

/// Snapshot of the currently served names and an optional selected adapter.
#[derive(Debug)]
pub(crate) struct LoraModelResolution {
    pub model_names: Vec<String>,
    pub lora_request: Option<LoraRequest>,
    pub lease: LoraLease,
}

#[derive(Clone)]
struct RegisteredLora {
    request: LoraRequest,
    lease: Arc<RwLock<()>>,
    /// Serializes the first engine activation for lazily registered adapters.
    activation: Arc<Mutex<bool>>,
}

/// Runtime registry for eager HTTP adapters and lazy OpenEngine adapters.
pub(crate) struct LoraManager {
    registry: RwLock<IndexMap<String, RegisteredLora>>,
    id_counter: AtomicU64,
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
pub(crate) enum LoadExactLoraError {
    BaseModelName { lora_name: String },
    Conflict { existing: LoraRequest },
}

#[derive(Debug)]
pub(crate) enum ActivateLoraError {
    NotFound { lora_name: String },
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

impl LoraManager {
    pub fn new() -> Self {
        Self {
            registry: RwLock::new(IndexMap::new()),
            id_counter: AtomicU64::new(0),
            update_lock: Mutex::new(()),
        }
    }

    pub async fn served_lora_requests(&self) -> Vec<LoraRequest> {
        self.registry.read().await.values().map(|entry| entry.request.clone()).collect()
    }

    /// Resolve a model and hold a generation lease when it selects an adapter.
    pub async fn resolve_model(
        &self,
        base_model_names: &[String],
        model_name: Option<&str>,
    ) -> LoraModelResolution {
        loop {
            let candidate = {
                let registry = self.registry.read().await;
                model_name.and_then(|name| registry.get(name).cloned())
            };
            let lease = match &candidate {
                Some(entry) => Some(entry.lease.clone().read_owned().await),
                None => None,
            };
            let registry = self.registry.read().await;
            let current = model_name.and_then(|name| registry.get(name));
            if !same_entry(current, candidate.as_ref()) {
                drop(registry);
                drop(lease);
                tokio::task::yield_now().await;
                continue;
            }
            let mut model_names = base_model_names.to_vec();
            model_names.extend(registry.keys().cloned());
            return LoraModelResolution {
                model_names,
                lora_request: current.map(|entry| entry.request.clone()),
                lease,
            };
        }
    }

    /// Preserve the existing HTTP admin contract: eagerly activate adapters.
    pub async fn load_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        base_model_names: &[String],
        lora_name: String,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Result<LoraRequest, LoadLoraError> {
        let _update = self.update_lock.lock().await;
        if base_model_names.iter().any(|name| name == &lora_name) {
            return Err(LoadLoraError::BaseModelName { lora_name });
        }
        let previous = self.registry.read().await.get(&lora_name).cloned();
        if previous.is_some() && !load_inplace {
            return Err(LoadLoraError::AlreadyLoaded { lora_name });
        }
        let _lease = match &previous {
            Some(entry) => Some(entry.lease.clone().write_owned().await),
            None => None,
        };
        let lora_int_id = previous
            .as_ref()
            .map(|entry| entry.request.lora_int_id)
            .unwrap_or_else(|| self.id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        let request = LoraRequest::new(
            lora_name.clone(),
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        );
        let loaded = engine_core_client.add_lora(&request).await.map_err(LoadLoraError::Engine)?;
        if !loaded {
            return Err(LoadLoraError::NotLoaded { lora_name });
        }
        let entry = RegisteredLora {
            request: request.clone(),
            lease: previous.map(|entry| entry.lease).unwrap_or_else(|| Arc::new(RwLock::new(()))),
            activation: Arc::new(Mutex::new(true)),
        };
        self.registry.write().await.insert(lora_name, entry);
        Ok(request)
    }

    /// Register an OpenEngine adapter with its caller-assigned stable ID.
    /// Engine workers load it only when a generation first selects it.
    pub async fn register_lora_exact(
        &self,
        base_model_names: &[String],
        request: LoraRequest,
    ) -> Result<(LoraRequest, bool), LoadExactLoraError> {
        let _update = self.update_lock.lock().await;
        if base_model_names.iter().any(|name| name == &request.lora_name) {
            return Err(LoadExactLoraError::BaseModelName {
                lora_name: request.lora_name,
            });
        }
        let registry = self.registry.read().await;
        if let Some(existing) = registry.values().find(|entry| {
            entry.request.lora_name == request.lora_name
                || entry.request.lora_int_id == request.lora_int_id
                || entry.request.lora_path == request.lora_path
        }) {
            if same_wire_identity(&existing.request, &request) {
                return Ok((existing.request.clone(), true));
            }
            return Err(LoadExactLoraError::Conflict {
                existing: existing.request.clone(),
            });
        }
        drop(registry);
        self.id_counter.fetch_max(request.lora_int_id, Ordering::Relaxed);
        self.registry.write().await.insert(
            request.lora_name.clone(),
            RegisteredLora {
                request: request.clone(),
                lease: Arc::new(RwLock::new(())),
                activation: Arc::new(Mutex::new(false)),
            },
        );
        Ok((request, false))
    }

    /// Select a registered adapter, lazily activate it, and hold an in-flight
    /// lease so logical unload cannot invalidate an admitted request.
    pub async fn activate_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        lora_name: &str,
    ) -> Result<(LoraRequest, LoraLease), ActivateLoraError> {
        loop {
            let entry = self.registry.read().await.get(lora_name).cloned().ok_or_else(|| {
                ActivateLoraError::NotFound {
                    lora_name: lora_name.to_string(),
                }
            })?;
            let lease = entry.lease.clone().read_owned().await;
            if !same_entry(self.registry.read().await.get(lora_name), Some(&entry)) {
                drop(lease);
                tokio::task::yield_now().await;
                continue;
            }
            let mut activated = entry.activation.lock().await;
            if !*activated {
                let loaded = engine_core_client
                    .add_lora(&entry.request)
                    .await
                    .map_err(ActivateLoraError::Engine)?;
                if !loaded {
                    return Err(ActivateLoraError::NotLoaded {
                        lora_name: lora_name.to_string(),
                    });
                }
                *activated = true;
            }
            drop(activated);
            return Ok((entry.request, Some(lease)));
        }
    }

    /// Logical OpenEngine unload. Removing the registry entry immediately
    /// rejects new selection; existing leases keep admitted requests alive.
    pub async fn logical_unload(&self, lora_name: &str) -> Result<LoraRequest, UnloadLoraError> {
        let _update = self.update_lock.lock().await;
        self.registry
            .write()
            .await
            .shift_remove(lora_name)
            .map(|entry| entry.request)
            .ok_or_else(|| UnloadLoraError::NotFound {
                lora_name: lora_name.to_string(),
            })
    }

    /// Preserve the existing HTTP unload contract, including worker eviction.
    pub async fn unload_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        lora_name: &str,
        requested_lora_int_id: Option<u64>,
    ) -> Result<LoraRequest, UnloadLoraError> {
        let _update = self.update_lock.lock().await;
        let entry = self.registry.read().await.get(lora_name).cloned().ok_or_else(|| {
            UnloadLoraError::NotFound {
                lora_name: lora_name.to_string(),
            }
        })?;
        if let Some(actual) = requested_lora_int_id
            && actual != entry.request.lora_int_id
        {
            return Err(UnloadLoraError::IntIdMismatch {
                lora_name: lora_name.to_string(),
                expected: entry.request.lora_int_id,
                actual,
            });
        }
        let _lease = entry.lease.clone().write_owned().await;
        if *entry.activation.lock().await {
            let removed = engine_core_client
                .remove_lora(entry.request.lora_int_id)
                .await
                .map_err(UnloadLoraError::Engine)?;
            if !removed {
                return Err(UnloadLoraError::NotRemoved {
                    lora_name: entry.request.lora_name,
                    lora_int_id: entry.request.lora_int_id,
                });
            }
        }
        Ok(self
            .registry
            .write()
            .await
            .shift_remove(lora_name)
            .map(|entry| entry.request)
            .unwrap_or(entry.request))
    }
}

fn same_wire_identity(left: &LoraRequest, right: &LoraRequest) -> bool {
    left.lora_name == right.lora_name
        && left.lora_int_id == right.lora_int_id
        && left.lora_path == right.lora_path
}

fn same_entry(left: Option<&RegisteredLora>, right: Option<&RegisteredLora>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Arc::ptr_eq(&left.lease, &right.lease),
        (None, None) => true,
        _ => false,
    }
}

/// Hold an adapter generation lease until a streaming response ends or drops.
pub(crate) struct LoraLeaseStream<S> {
    stream: Pin<Box<S>>,
    _lease: LoraLease,
}

impl<S> Unpin for LoraLeaseStream<S> {}

impl<S: Stream> Stream for LoraLeaseStream<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

pub(crate) fn hold_lora_lease<S>(stream: S, lease: LoraLease) -> LoraLeaseStream<S>
where
    S: Stream,
{
    LoraLeaseStream {
        stream: Box::pin(stream),
        _lease: lease,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_identity_ignores_internal_load_flags() {
        let public = LoraRequest::new("adapter".into(), 17, "/adapter".into(), false, false);
        let mut internal = public.clone();
        internal.load_inplace = true;
        internal.is_3d_lora_weight = true;
        assert!(same_wire_identity(&public, &internal));
    }

    #[tokio::test]
    async fn exact_registration_is_idempotent_and_rejects_identity_conflicts() {
        let manager = LoraManager::new();
        let base = vec!["base".to_string()];
        let request = LoraRequest::new("adapter".into(), 17, "/adapter".into(), false, false);
        let (_, already_loaded) =
            manager.register_lora_exact(&base, request.clone()).await.unwrap();
        assert!(!already_loaded);
        let (_, already_loaded) =
            manager.register_lora_exact(&base, request.clone()).await.unwrap();
        assert!(already_loaded);

        let conflict = LoraRequest::new("other".into(), 17, "/other".into(), false, false);
        assert!(matches!(
            manager.register_lora_exact(&base, conflict).await,
            Err(LoadExactLoraError::Conflict { .. })
        ));
    }

    #[tokio::test]
    async fn logical_unload_rejects_new_selection_without_waiting_for_admitted_lease() {
        let manager = LoraManager::new();
        let base = vec!["base".to_string()];
        manager
            .register_lora_exact(
                &base,
                LoraRequest::new("adapter".into(), 17, "/adapter".into(), false, false),
            )
            .await
            .unwrap();
        let admitted = manager.resolve_model(&base, Some("adapter")).await;
        assert!(admitted.lora_request.is_some());

        manager.logical_unload("adapter").await.unwrap();
        let new_request = manager.resolve_model(&base, Some("adapter")).await;
        assert!(new_request.lora_request.is_none());
        drop(admitted);
    }
}
