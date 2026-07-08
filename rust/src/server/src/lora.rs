use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll};

use futures::Stream;
use futures::future::join_all;
use indexmap::IndexMap;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, RwLock};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::Error as EngineCoreError;
use vllm_engine_core_client::protocol::lora::LoraRequest;

#[cfg(not(test))]
const LORA_MUTATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
#[cfg(test)]
const LORA_MUTATION_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);

struct MutationGuard<'a> {
    consistent: &'a AtomicBool,
    final_state_proven: bool,
}

impl<'a> MutationGuard<'a> {
    fn new(consistent: &'a AtomicBool) -> Self {
        Self {
            consistent,
            final_state_proven: false,
        }
    }

    fn prove_final_state(&mut self) {
        self.final_state_proven = true;
    }
}

impl Drop for MutationGuard<'_> {
    fn drop(&mut self) {
        if !self.final_state_proven {
            self.consistent.store(false, Ordering::Release);
        }
    }
}

/// Snapshot of the currently served model names plus the requested LoRA, if
/// the model name resolves to a dynamic adapter.
pub(crate) type LoraLease = Option<OwnedRwLockReadGuard<()>>;

#[derive(Debug)]
pub(crate) struct LoraModelResolution {
    pub model_names: Vec<String>,
    pub lora_request: Option<LoraRequest>,
    pub lease: LoraLease,
}

#[derive(Clone)]
struct LoadedLora {
    request: LoraRequest,
    lease: std::sync::Arc<RwLock<()>>,
}

/// Runtime registry for dynamically loaded LoRA adapters.
pub(crate) struct LoraManager {
    /// Loaded adapters and their generation leases, keyed by public model name.
    registry: RwLock<IndexMap<String, LoadedLora>>,
    /// Monotonic adapter id allocator. LoRA ids are one-indexed.
    id_counter: AtomicU64,
    /// Serialize dynamic LoRA registry updates around engine utility calls.
    update_lock: Mutex<()>,
    /// False after a failed compensation leaves per-engine state indeterminate.
    consistent: AtomicBool,
}

#[derive(Debug)]
pub(crate) enum LoadLoraError {
    Inconsistent,
    AlreadyLoaded { lora_name: String },
    BaseModelName { lora_name: String },
    Engine(vllm_engine_core_client::Error),
    NotLoaded { lora_name: String },
}

#[derive(Debug)]
pub(crate) enum UnloadLoraError {
    Inconsistent,
    NotFound {
        lora_name: String,
    },
    IntIdMismatch {
        lora_name: String,
        expected: u64,
        actual: u64,
    },
    Engine(vllm_engine_core_client::Error),
}

#[derive(Debug)]
pub(crate) enum LoadExactLoraError {
    Inconsistent,
    BaseModelName { lora_name: String },
    Conflict { existing: LoraRequest },
    Engine(vllm_engine_core_client::Error),
    NotLoaded { lora_name: String },
}

enum ApplyLoadError {
    Engine(EngineCoreError),
    Rejected,
}

impl LoraManager {
    pub fn new() -> Self {
        Self {
            registry: RwLock::new(IndexMap::new()),
            id_counter: AtomicU64::new(0),
            update_lock: Mutex::new(()),
            consistent: AtomicBool::new(true),
        }
    }

    pub fn is_consistent(&self) -> bool {
        self.consistent.load(Ordering::Acquire)
    }

    /// Snapshot loaded LoRA adapters in load order.
    pub async fn served_lora_requests(&self) -> Vec<LoraRequest> {
        self.registry
            .read()
            .await
            .values()
            .map(|loaded| loaded.request.clone())
            .collect()
    }

    /// Resolve the requested model against one consistent LoRA registry
    /// snapshot.
    pub async fn resolve_model(
        &self,
        base_model_names: &[String],
        model_name: Option<&str>,
    ) -> LoraModelResolution {
        loop {
            let candidate = match model_name {
                Some(name) => {
                    self.registry.read().await.get(name).map(|loaded| loaded.lease.clone())
                }
                None => None,
            };
            let lease = match &candidate {
                Some(lease) => Some(lease.clone().read_owned().await),
                None => None,
            };
            let registry = self.registry.read().await;
            let current = model_name.and_then(|name| registry.get(name));
            if !same_lease(current, candidate.as_ref()) {
                drop(registry);
                drop(lease);
                tokio::task::yield_now().await;
                continue;
            }

            let mut model_names = base_model_names.to_vec();
            model_names.extend(registry.keys().cloned());
            let lora_request = current.map(|loaded| loaded.request.clone());
            return LoraModelResolution {
                model_names,
                lora_request,
                lease,
            };
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
        if !self.is_consistent() {
            return Err(LoadLoraError::Inconsistent);
        }
        if base_model_names.iter().any(|name| name == &lora_name) {
            return Err(LoadLoraError::BaseModelName { lora_name });
        }
        let previous = self
            .registry
            .read()
            .await
            .get(&lora_name)
            .map(|loaded| (loaded.request.clone(), loaded.lease.clone()));
        if previous.is_some() && !load_inplace {
            return Err(LoadLoraError::AlreadyLoaded { lora_name });
        }
        let _generation_guard = match previous.as_ref().map(|(_, lease)| lease.clone()) {
            Some(lease) => Some(lease.write_owned().await),
            None => None,
        };

        let lora_int_id = previous
            .as_ref()
            .map(|(request, _)| request.lora_int_id)
            .unwrap_or_else(|| self.id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        let lora_request = LoraRequest::new(
            lora_name.clone(),
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        );

        let mut mutation = self
            .apply_load(
                engine_core_client,
                &lora_request,
                previous.as_ref().map(|(request, _)| request),
            )
            .await
            .map_err(|error| match error {
                ApplyLoadError::Engine(error) => LoadLoraError::Engine(error),
                ApplyLoadError::Rejected => LoadLoraError::NotLoaded {
                    lora_name: lora_name.clone(),
                },
            })?;
        let lease = previous
            .map(|(_, lease)| lease)
            .unwrap_or_else(|| std::sync::Arc::new(RwLock::new(())));
        self.registry.write().await.insert(
            lora_name,
            LoadedLora {
                request: lora_request.clone(),
                lease,
            },
        );
        mutation.prove_final_state();
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
        if !self.is_consistent() {
            return Err(LoadExactLoraError::Inconsistent);
        }
        if base_model_names.iter().any(|name| name == &lora_request.lora_name) {
            return Err(LoadExactLoraError::BaseModelName {
                lora_name: lora_request.lora_name,
            });
        }

        let registry = self.registry.read().await;
        if let Some(existing) = registry.values().find(|loaded| {
            let existing = &loaded.request;
            existing.lora_name == lora_request.lora_name
                || existing.lora_int_id == lora_request.lora_int_id
                || existing.lora_path == lora_request.lora_path
        }) {
            let existing = &existing.request;
            if same_wire_identity(existing, &lora_request) {
                return Ok((existing.clone(), true));
            }
            return Err(LoadExactLoraError::Conflict {
                existing: existing.clone(),
            });
        }
        drop(registry);

        let mut mutation =
            self.apply_load(engine_core_client, &lora_request, None)
                .await
                .map_err(|error| match error {
                    ApplyLoadError::Engine(error) => LoadExactLoraError::Engine(error),
                    ApplyLoadError::Rejected => LoadExactLoraError::NotLoaded {
                        lora_name: lora_request.lora_name.clone(),
                    },
                })?;

        self.id_counter.fetch_max(lora_request.lora_int_id, Ordering::Relaxed);
        self.registry.write().await.insert(
            lora_request.lora_name.clone(),
            LoadedLora {
                request: lora_request.clone(),
                lease: std::sync::Arc::new(RwLock::new(())),
            },
        );
        mutation.prove_final_state();
        Ok((lora_request, false))
    }

    /// Apply one load or replacement across every engine rank. The returned
    /// guard remains uncommitted until the caller updates the frontend
    /// registry, so cancellation between engine success and registry commit
    /// still fails closed.
    async fn apply_load<'a>(
        &'a self,
        engine_core_client: &EngineCoreClient,
        lora_request: &LoraRequest,
        previous: Option<&LoraRequest>,
    ) -> Result<MutationGuard<'a>, ApplyLoadError> {
        let mut mutation = MutationGuard::new(&self.consistent);
        let results = match call_all_bounded::<bool, _>(
            engine_core_client,
            "add_lora",
            (lora_request,),
        )
        .await
        {
            Ok(results) => results,
            Err(error) => {
                if matches!(&error, EngineCoreError::UtilityCallTimeout { .. }) {
                    if restore_load_state(engine_core_client, lora_request, previous).await {
                        mutation.prove_final_state();
                    }
                } else {
                    // Outer errors occur before dispatch, so engine state is
                    // unchanged.
                    mutation.prove_final_state();
                }
                return Err(ApplyLoadError::Engine(error));
            }
        };

        let Some(failure_index) = results.iter().position(|result| !matches!(result, Ok(true)))
        else {
            return Ok(mutation);
        };
        if restore_load_state(engine_core_client, lora_request, previous).await {
            mutation.prove_final_state();
        }
        match results.into_iter().nth(failure_index).unwrap() {
            Ok(false) => Err(ApplyLoadError::Rejected),
            Err(error) => Err(ApplyLoadError::Engine(error)),
            Ok(true) => unreachable!(),
        }
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
        if !self.is_consistent() {
            return Err(UnloadLoraError::Inconsistent);
        }
        let (lora_request, lease) = self
            .registry
            .read()
            .await
            .get(lora_name)
            .map(|loaded| (loaded.request.clone(), loaded.lease.clone()))
            .ok_or_else(|| UnloadLoraError::NotFound {
                lora_name: lora_name.to_string(),
            })?;
        let _generation_guard = lease.write_owned().await;

        if let Some(actual) = requested_lora_int_id
            && actual != lora_request.lora_int_id
        {
            return Err(UnloadLoraError::IntIdMismatch {
                lora_name: lora_name.to_string(),
                expected: lora_request.lora_int_id,
                actual,
            });
        }

        let mut mutation = MutationGuard::new(&self.consistent);
        let results = match call_all_bounded::<bool, _>(
            engine_core_client,
            "remove_lora",
            (lora_request.lora_int_id,),
        )
        .await
        {
            Ok(results) => results,
            Err(error) => {
                if matches!(&error, EngineCoreError::UtilityCallTimeout { .. }) {
                    if restore_removed(engine_core_client, &lora_request).await {
                        mutation.prove_final_state();
                    }
                } else {
                    mutation.prove_final_state();
                }
                return Err(UnloadLoraError::Engine(error));
            }
        };
        if let Some(failure_index) = results.iter().position(Result::is_err) {
            if restore_removed(engine_core_client, &lora_request).await {
                mutation.prove_final_state();
            }
            let failure = results.into_iter().nth(failure_index).unwrap();
            return match failure {
                Err(error) => Err(UnloadLoraError::Engine(error)),
                Ok(_) => unreachable!(),
            };
        }

        let removed = self
            .registry
            .write()
            .await
            .shift_remove(lora_name)
            .map(|loaded| loaded.request)
            .unwrap_or(lora_request);
        mutation.prove_final_state();
        Ok(removed)
    }
}

fn same_wire_identity(left: &LoraRequest, right: &LoraRequest) -> bool {
    left.lora_name == right.lora_name
        && left.lora_int_id == right.lora_int_id
        && left.lora_path == right.lora_path
}

fn same_lease(
    current: Option<&LoadedLora>,
    candidate: Option<&std::sync::Arc<RwLock<()>>>,
) -> bool {
    match (current, candidate) {
        (Some(current), Some(candidate)) => std::sync::Arc::ptr_eq(&current.lease, candidate),
        (None, None) => true,
        _ => false,
    }
}

/// Hold an adapter's shared generation lease until the wrapped stream ends or
/// is dropped.
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

async fn call_all_bounded<T, A>(
    engine_core_client: &EngineCoreClient,
    method: &str,
    args: A,
) -> Result<Vec<Result<T, EngineCoreError>>, EngineCoreError>
where
    T: serde::de::DeserializeOwned,
    A: serde::Serialize + std::fmt::Debug,
{
    tokio::time::timeout(
        LORA_MUTATION_TIMEOUT,
        engine_core_client.call_utility_per_engine(method, args),
    )
    .await
    .map_err(|_| EngineCoreError::UtilityCallTimeout {
        method: method.to_string(),
        timeout: LORA_MUTATION_TIMEOUT,
    })?
}

async fn restore_load_state(
    engine_core_client: &EngineCoreClient,
    attempted: &LoraRequest,
    previous: Option<&LoraRequest>,
) -> bool {
    match previous {
        Some(previous) => restore_previous_on_all(engine_core_client, previous).await,
        None => remove_from_all(engine_core_client, attempted.lora_int_id).await,
    }
}

async fn remove_from_all(engine_core_client: &EngineCoreClient, lora_int_id: u64) -> bool {
    let calls = engine_core_client.ready_responses().into_iter().map(|ready| async move {
        engine_core_client
            .call_utility_on_engine::<bool, _>(
                ready.data_parallel_rank,
                "remove_lora",
                (lora_int_id,),
            )
            .await
    });
    // `false` also proves the desired absent state.
    tokio::time::timeout(LORA_MUTATION_TIMEOUT, join_all(calls))
        .await
        .is_ok_and(|outcomes| outcomes.iter().all(Result::is_ok))
}

async fn restore_previous_on_all(
    engine_core_client: &EngineCoreClient,
    previous: &LoraRequest,
) -> bool {
    let mut previous = previous.clone();
    previous.load_inplace = true;
    let calls = engine_core_client.ready_responses().into_iter().map(|ready| {
        let previous = previous.clone();
        async move {
            engine_core_client
                .call_utility_on_engine::<bool, _>(
                    ready.data_parallel_rank,
                    "add_lora",
                    (&previous,),
                )
                .await
        }
    });
    tokio::time::timeout(LORA_MUTATION_TIMEOUT, join_all(calls))
        .await
        .is_ok_and(|outcomes| outcomes.iter().all(|outcome| matches!(outcome, Ok(true))))
}

async fn restore_removed(
    engine_core_client: &EngineCoreClient,
    lora_request: &LoraRequest,
) -> bool {
    let calls = engine_core_client.ready_responses().into_iter().map(|ready| async move {
        engine_core_client
            .call_utility_on_engine::<bool, _>(
                ready.data_parallel_rank,
                "add_lora",
                (lora_request,),
            )
            .await
    });
    tokio::time::timeout(LORA_MUTATION_TIMEOUT, join_all(calls))
        .await
        .is_ok_and(|outcomes| outcomes.iter().all(|outcome| matches!(outcome, Ok(true))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_identity_ignores_internal_load_options() {
        let public = LoraRequest::new(
            "adapter-a".to_string(),
            17,
            "/adapters/a".to_string(),
            false,
            false,
        );
        let mut internal = public.clone();
        internal.load_inplace = true;
        internal.is_3d_lora_weight = true;
        internal.base_model_name = Some("base".to_string());
        internal.tensorizer_config_dict = Some(rmpv::Value::Map(vec![(
            rmpv::Value::from("format"),
            rmpv::Value::from("safetensors"),
        )]));

        assert!(same_wire_identity(&public, &internal));
    }

    #[test]
    fn lease_identity_rejects_reloaded_adapter() {
        let old_lease = std::sync::Arc::new(RwLock::new(()));
        let reloaded = LoadedLora {
            request: LoraRequest::new(
                "adapter-a".to_string(),
                17,
                "/adapters/a".to_string(),
                false,
                false,
            ),
            lease: std::sync::Arc::new(RwLock::new(())),
        };

        assert!(!same_lease(Some(&reloaded), Some(&old_lease)));
        assert!(same_lease(Some(&reloaded), Some(&reloaded.lease)));
    }
}
