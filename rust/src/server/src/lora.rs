// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;
use indexmap::IndexMap;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, RwLock};
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::{EngineCoreClient, Error as EngineCoreError};

#[cfg(not(test))]
const LORA_MUTATION_TIMEOUT: Duration = Duration::from_secs(30);
#[cfg(test)]
const LORA_MUTATION_TIMEOUT: Duration = Duration::from_millis(250);

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

pub(crate) type LoraLease = Option<OwnedRwLockReadGuard<()>>;

/// Snapshot of the currently served model names plus the requested LoRA, if
/// the model name resolves to a dynamic adapter.
#[derive(Debug)]
pub(crate) struct LoraModelResolution {
    pub model_names: Vec<String>,
    pub lora_request: Option<LoraRequest>,
    pub lease: LoraLease,
}

#[derive(Clone)]
struct LoadedLora {
    request: LoraRequest,
    lease: Arc<RwLock<()>>,
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
    Engine(EngineCoreError),
    NotLoaded { lora_name: String },
}

#[derive(Debug)]
pub(crate) enum LoadExactLoraError {
    Inconsistent,
    BaseModelName { lora_name: String },
    Conflict { existing: LoraRequest },
    Engine(EngineCoreError),
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
    Engine(EngineCoreError),
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

    /// Resolve the requested model and take a generation lease atomically with
    /// respect to replacement and unload operations.
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
            return LoraModelResolution {
                model_names,
                lora_request: current.map(|loaded| loaded.request.clone()),
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
        let _update_guard = self.update_lock.lock().await;
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
        let lease = previous.map(|(_, lease)| lease).unwrap_or_else(|| Arc::new(RwLock::new(())));
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
        let _update_guard = self.update_lock.lock().await;
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
            if same_wire_identity(&existing.request, &lora_request) {
                return Ok((existing.request.clone(), true));
            }
            return Err(LoadExactLoraError::Conflict {
                existing: existing.request.clone(),
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
                lease: Arc::new(RwLock::new(())),
            },
        );
        mutation.prove_final_state();
        Ok((lora_request, false))
    }

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
                    // The timeout may race a send that reached the engine but
                    // had not completed locally. Compensate best-effort, but
                    // fail closed because cross-rank ordering is unproven.
                    let _ = restore_load_state(engine_core_client, lora_request, previous).await;
                } else {
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

    /// Remove one dynamic LoRA adapter from every engine and the registry.
    pub async fn unload_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        lora_name: &str,
        requested_lora_int_id: Option<u64>,
    ) -> Result<LoraRequest, UnloadLoraError> {
        let _update_guard = self.update_lock.lock().await;
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
                    // See apply_load: a timed-out mutation has indeterminate
                    // ordering even when best-effort compensation succeeds.
                    let _ = restore_removed(engine_core_client, &lora_request).await;
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
            return Err(UnloadLoraError::Engine(
                results.into_iter().nth(failure_index).unwrap().unwrap_err(),
            ));
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

fn same_lease(current: Option<&LoadedLora>, candidate: Option<&Arc<RwLock<()>>>) -> bool {
    match (current, candidate) {
        (Some(current), Some(candidate)) => Arc::ptr_eq(&current.lease, candidate),
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
    call_all_bounded::<bool, _>(engine_core_client, "remove_lora", (lora_int_id,))
        .await
        .is_ok_and(|outcomes| outcomes.iter().all(Result::is_ok))
}

async fn restore_previous_on_all(
    engine_core_client: &EngineCoreClient,
    previous: &LoraRequest,
) -> bool {
    let mut previous = previous.clone();
    previous.load_inplace = true;
    call_all_bounded::<bool, _>(engine_core_client, "add_lora", (&previous,))
        .await
        .is_ok_and(|outcomes| outcomes.iter().all(|outcome| matches!(outcome, Ok(true))))
}

async fn restore_removed(
    engine_core_client: &EngineCoreClient,
    lora_request: &LoraRequest,
) -> bool {
    call_all_bounded::<bool, _>(engine_core_client, "add_lora", (lora_request,))
        .await
        .is_ok_and(|outcomes| outcomes.iter().all(|outcome| matches!(outcome, Ok(true))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_engine_core_client::protocol::decode_value;
    use vllm_engine_core_client::protocol::output::{EngineCoreOutputs, UtilityCallOutput};
    use vllm_engine_core_client::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
    use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task_with_ready};
    use vllm_engine_core_client::{EngineCoreClientConfig, TransportMode};
    use zeromq::ZmqMessage;
    use zeromq::prelude::{SocketRecv as _, SocketSend as _};

    async fn reply_utility(push: &mut zeromq::PushSocket, call_id: u64, result: bool) {
        let output: EngineCoreOutputs = UtilityCallOutput {
            engine_index: 0,
            timestamp: 0.0,
            output: UtilityOutput {
                call_id: call_id.into(),
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(rmpv::Value::from(
                    result,
                ))),
            },
        }
        .into();
        push.send(ZmqMessage::from(rmp_serde::to_vec_named(&output).unwrap()))
            .await
            .unwrap();
    }

    async fn recv_utility_call_id(dealer: &mut zeromq::DealerSocket, method: &str) -> u64 {
        let message = dealer.recv().await.unwrap().into_vec();
        let payload = decode_value(&message[1]).unwrap();
        let array = payload.as_array().unwrap();
        assert_eq!(array[2], rmpv::Value::from(method));
        array[1].as_u64().unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn partial_rank_load_failure_rolls_back_every_rank() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let ready = |rank| {
            let mut response = vllm_engine_core_client::mock_engine::default_ready_response();
            response.data_parallel_size = 2;
            response.data_parallel_rank = rank;
            response.supports_lora = true;
            response.max_loras = 8;
            response
        };

        let (shutdown_zero, task_zero) = spawn_mock_engine_task_with_ready(
            handshake.clone(),
            vec![0x00, 0x00],
            ready(0),
            |dealer, push| {
                Box::pin(async move {
                    let load = recv_utility_call_id(dealer, "add_lora").await;
                    reply_utility(push, load, true).await;
                    let rollback = recv_utility_call_id(dealer, "remove_lora").await;
                    reply_utility(push, rollback, true).await;
                })
            },
        );
        let (shutdown_one, task_one) = spawn_mock_engine_task_with_ready(
            handshake.clone(),
            vec![0x01, 0x00],
            ready(1),
            |dealer, push| {
                Box::pin(async move {
                    let load = recv_utility_call_id(dealer, "add_lora").await;
                    reply_utility(push, load, false).await;
                    let rollback = recv_utility_call_id(dealer, "remove_lora").await;
                    reply_utility(push, rollback, true).await;
                })
            },
        );

        let mut config = EngineCoreClientConfig::new_single(handshake)
            .with_model_name("test-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            );
        let TransportMode::HandshakeOwner { engine_count, .. } = &mut config.transport_mode else {
            unreachable!()
        };
        *engine_count = 2;
        let client = EngineCoreClient::connect(config).await.unwrap();
        let manager = LoraManager::new();
        let error = manager
            .load_lora(
                &client,
                &["test-model".to_string()],
                "adapter-a".to_string(),
                "/adapters/a".to_string(),
                false,
                false,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, LoadLoraError::NotLoaded { .. }));
        assert!(manager.is_consistent());
        assert!(manager.served_lora_requests().await.is_empty());

        let _ = shutdown_zero.send(());
        let _ = shutdown_one.send(());
        task_zero.await.unwrap();
        task_one.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn load_timeout_fails_closed_after_successful_compensation() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let mut ready = vllm_engine_core_client::mock_engine::default_ready_response();
        ready.supports_lora = true;
        ready.max_loras = 8;
        let (shutdown, engine_task) = spawn_mock_engine_task_with_ready(
            handshake.clone(),
            vec![0x00, 0x00],
            ready,
            |dealer, push| {
                Box::pin(async move {
                    let _timed_out_load = recv_utility_call_id(dealer, "add_lora").await;
                    let rollback = recv_utility_call_id(dealer, "remove_lora").await;
                    reply_utility(push, rollback, true).await;
                })
            },
        );
        let client = EngineCoreClient::connect(
            EngineCoreClientConfig::new_single(handshake)
                .with_model_name("test-model")
                .with_local_input_output_addresses(
                    Some(ipc.input_endpoint()),
                    Some(ipc.output_endpoint()),
                ),
        )
        .await
        .unwrap();
        let manager = LoraManager::new();

        let error = manager
            .load_lora(
                &client,
                &["test-model".to_string()],
                "adapter-timeout".to_string(),
                "/adapters/timeout".to_string(),
                false,
                false,
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            LoadLoraError::Engine(EngineCoreError::UtilityCallTimeout { .. })
        ));
        assert!(!manager.is_consistent());
        assert!(manager.served_lora_requests().await.is_empty());

        let _ = shutdown.send(());
        engine_task.await.unwrap();
        client.shutdown().await.unwrap();
    }
}
