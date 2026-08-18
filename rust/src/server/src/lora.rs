// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use indexmap::IndexMap;
use tokio::sync::{Mutex, RwLock};
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
    /// False after a mutation leaves per-engine state indeterminate.
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
            requests: RwLock::new(IndexMap::new()),
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
        let _update_guard = self.update_lock.lock().await;
        if !self.is_consistent() {
            return Err(LoadLoraError::Inconsistent);
        }
        if base_model_names.iter().any(|name| name == &lora_name) {
            return Err(LoadLoraError::BaseModelName { lora_name });
        }
        let previous = self.requests.read().await.get(&lora_name).cloned();
        if previous.is_some() && !load_inplace {
            return Err(LoadLoraError::AlreadyLoaded { lora_name });
        }

        let lora_int_id = previous
            .as_ref()
            .map(|request| request.lora_int_id)
            .unwrap_or_else(|| self.id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        let lora_request = LoraRequest::new(
            lora_name.clone(),
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        );

        let mut mutation = self
            .apply_load(engine_core_client, &lora_request, previous.as_ref())
            .await
            .map_err(|error| match error {
                ApplyLoadError::Engine(error) => LoadLoraError::Engine(error),
                ApplyLoadError::Rejected => LoadLoraError::NotLoaded {
                    lora_name: lora_name.clone(),
                },
            })?;
        self.requests.write().await.insert(lora_name, lora_request.clone());
        mutation.prove_final_state();
        Ok(lora_request)
    }

    async fn apply_load<'a>(
        &'a self,
        engine_core_client: &EngineCoreClient,
        lora_request: &LoraRequest,
        previous: Option<&LoraRequest>,
    ) -> Result<MutationGuard<'a>, ApplyLoadError> {
        let mut mutation = MutationGuard::new(&self.consistent);
        match call_all_bounded::<bool, _>(engine_core_client, "add_lora", (lora_request,)).await {
            Ok(results) if results.iter().all(|&loaded| loaded) => Ok(mutation),
            Ok(_) => {
                if restore_load_state(engine_core_client, lora_request, previous).await {
                    mutation.prove_final_state();
                }
                Err(ApplyLoadError::Rejected)
            }
            Err(error) => {
                let timed_out = matches!(&error, EngineCoreError::UtilityCallTimeout { .. });
                let restored = restore_load_state(engine_core_client, lora_request, previous).await;
                if !timed_out && restored {
                    mutation.prove_final_state();
                }
                Err(ApplyLoadError::Engine(error))
            }
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
        let _update_guard = self.update_lock.lock().await;
        if !self.is_consistent() {
            return Err(UnloadLoraError::Inconsistent);
        }
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

        let mut mutation = MutationGuard::new(&self.consistent);
        match call_all_bounded::<bool, _>(
            engine_core_client,
            "remove_lora",
            (lora_request.lora_int_id,),
        )
        .await
        {
            Ok(_) => {
                let removed =
                    self.requests.write().await.shift_remove(lora_name).unwrap_or(lora_request);
                mutation.prove_final_state();
                Ok(removed)
            }
            Err(error) => {
                let timed_out = matches!(&error, EngineCoreError::UtilityCallTimeout { .. });
                let restored = restore_removed(engine_core_client, &lora_request).await;
                if !timed_out && restored {
                    mutation.prove_final_state();
                }
                Err(UnloadLoraError::Engine(error))
            }
        }
    }
}

async fn call_all_bounded<T, A>(
    engine_core_client: &EngineCoreClient,
    method: &str,
    args: A,
) -> Result<Vec<T>, EngineCoreError>
where
    T: serde::de::DeserializeOwned,
    A: serde::Serialize + std::fmt::Debug,
{
    tokio::time::timeout(
        LORA_MUTATION_TIMEOUT,
        engine_core_client.call_utility(method, args),
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
        .is_ok()
}

async fn restore_previous_on_all(
    engine_core_client: &EngineCoreClient,
    previous: &LoraRequest,
) -> bool {
    let mut previous = previous.clone();
    previous.load_inplace = true;
    call_all_bounded::<bool, _>(engine_core_client, "add_lora", (&previous,))
        .await
        .is_ok_and(|results| results.into_iter().all(|loaded| loaded))
}

async fn restore_removed(
    engine_core_client: &EngineCoreClient,
    lora_request: &LoraRequest,
) -> bool {
    call_all_bounded::<bool, _>(engine_core_client, "add_lora", (lora_request,))
        .await
        .is_ok_and(|results| results.into_iter().all(|loaded| loaded))
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
            response.effective_data_parallel_size = 2;
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
