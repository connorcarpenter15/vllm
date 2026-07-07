use super::*;

fn test_adapter(dir: &tempfile::TempDir) -> pb::LoraAdapter {
    pb::LoraAdapter {
        lora_id: 17,
        lora_name: "adapter-a".to_string(),
        source_path: dir.path().to_string_lossy().into_owned(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn lifecycle_selects_adapter_for_generation() {
    let (mut engine_rpc, server_task, engine_task) = lora_test_server(true, false).await;
    let model = engine_rpc
        .get_model_info(pb::GetModelInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(model.supports_lora);
    assert_eq!(model.max_loras, 4);
    let dir = tempfile::tempdir().unwrap();
    let adapter = test_adapter(&dir);

    let loaded = engine_rpc
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(adapter.clone()),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!loaded.already_loaded);
    let loaded_again = engine_rpc
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(adapter.clone()),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(loaded_again.already_loaded);

    for conflict in [
        pb::LoraAdapter {
            lora_id: 18,
            ..adapter.clone()
        },
        pb::LoraAdapter {
            lora_name: "adapter-b".to_string(),
            ..adapter.clone()
        },
        pb::LoraAdapter {
            lora_id: 19,
            lora_name: "adapter-c".to_string(),
            source_path: adapter.source_path.clone(),
        },
    ] {
        let error = engine_rpc
            .load_lora(pb::LoadLoraRequest {
                adapter: Some(conflict),
            })
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::AlreadyExists);
    }

    let error = engine_rpc
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(pb::LoraAdapter {
                lora_id: 20,
                lora_name: "relative".to_string(),
                source_path: "relative/path".to_string(),
            }),
        })
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    let error = engine_rpc
        .unload_lora(pb::UnloadLoraRequest {
            lora_name: "missing".to_string(),
        })
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::NotFound);
    assert_eq!(
        engine_rpc
            .list_loras(pb::ListLorasRequest {})
            .await
            .unwrap()
            .into_inner()
            .adapters
            .len(),
        1
    );

    let mut missing_request = base_request();
    missing_request.lora_name = "missing".to_string();
    assert_eq!(
        engine_rpc.generate(missing_request).await.unwrap_err().code(),
        tonic::Code::NotFound
    );

    let mut request = base_request();
    request.lora_name = "adapter-a".to_string();
    let responses = engine_rpc
        .generate(request)
        .await
        .unwrap()
        .into_inner()
        .collect::<Vec<_>>()
        .await;
    assert!(responses.iter().all(Result::is_ok));

    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        engine_rpc.unload_lora(pb::UnloadLoraRequest {
            lora_name: "adapter-a".to_string(),
        }),
    )
    .await
    .expect("unload timed out")
    .unwrap();
    assert!(
        engine_rpc
            .list_loras(pb::ListLorasRequest {})
            .await
            .unwrap()
            .into_inner()
            .adapters
            .is_empty()
    );

    server_task.abort();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn lifecycle_rejects_disabled_engine() {
    let (mut engine_rpc, server_task, engine_task) = lora_test_server(false, true).await;
    let error = engine_rpc.list_loras(pb::ListLorasRequest {}).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    server_task.abort();
    drop(engine_task);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn lifecycle_rejects_new_loads_while_draining() {
    let (mut engine_rpc, server_task, engine_task) = lora_test_server(true, true).await;
    engine_rpc.drain(pb::DrainRequest {}).await.unwrap();
    let dir = tempfile::tempdir().unwrap();

    let error = engine_rpc
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(pb::LoraAdapter {
                lora_id: 17,
                lora_name: "adapter-a".to_string(),
                source_path: dir.path().to_string_lossy().into_owned(),
            }),
        })
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unavailable);

    server_task.abort();
    drop(engine_task);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn lifecycle_load_timeout_marks_registry_inconsistent() {
    let (mut engine_rpc, server_task, engine_task) =
        lora_scripted_test_server(true, |dealer, _push| {
            boxed_test_future(async move {
                let _load = recv_engine_message(dealer).await;
                std::future::pending::<()>().await;
            })
        })
        .await;
    let dir = tempfile::tempdir().unwrap();

    let error = engine_rpc
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(test_adapter(&dir)),
        })
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Internal);
    assert_eq!(
        engine_rpc.list_loras(pb::ListLorasRequest {}).await.unwrap_err().code(),
        tonic::Code::FailedPrecondition
    );
    let mut request = base_request();
    request.lora_name = "adapter-a".to_string();
    assert_eq!(
        engine_rpc.generate(request).await.unwrap_err().code(),
        tonic::Code::FailedPrecondition
    );

    server_task.abort();
    drop(engine_task);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn lifecycle_load_timeout_runs_bounded_compensation() {
    let (mut engine_rpc, server_task, engine_task) =
        lora_scripted_test_server(true, |dealer, push| {
            boxed_test_future(async move {
                let _load = recv_engine_message(dealer).await;
                let remove = recv_engine_message(dealer).await;
                let remove: rmpv::Value = rmp_serde::from_slice(&remove[1]).unwrap();
                let remove = remove.as_array().unwrap();
                assert_eq!(remove[2], rmpv::Value::from("remove_lora"));
                send_utility_bool(push, remove[1].as_u64().unwrap(), true).await;
            })
        })
        .await;
    let dir = tempfile::tempdir().unwrap();

    let error = engine_rpc
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(test_adapter(&dir)),
        })
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Internal);
    assert!(
        engine_rpc
            .list_loras(pb::ListLorasRequest {})
            .await
            .unwrap()
            .into_inner()
            .adapters
            .is_empty()
    );

    server_task.abort();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn lifecycle_cancelled_load_marks_registry_inconsistent() {
    let dispatched = std::sync::Arc::new(tokio::sync::Notify::new());
    let engine_notice = dispatched.clone();
    let (mut engine_rpc, server_task, engine_task) =
        lora_scripted_test_server(true, move |dealer, _push| {
            let engine_notice = engine_notice.clone();
            boxed_test_future(async move {
                let _load = recv_engine_message(dealer).await;
                engine_notice.notify_one();
                std::future::pending::<()>().await;
            })
        })
        .await;
    let dir = tempfile::tempdir().unwrap();
    let mut loading_client = engine_rpc.clone();
    let load = tokio::spawn(async move {
        loading_client
            .load_lora(pb::LoadLoraRequest {
                adapter: Some(test_adapter(&dir)),
            })
            .await
    });
    dispatched.notified().await;
    load.abort();
    assert!(load.await.unwrap_err().is_cancelled());

    assert_eq!(
        engine_rpc.list_loras(pb::ListLorasRequest {}).await.unwrap_err().code(),
        tonic::Code::FailedPrecondition
    );

    server_task.abort();
    drop(engine_task);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn lifecycle_cancelled_unload_marks_registry_inconsistent() {
    let dispatched = std::sync::Arc::new(tokio::sync::Notify::new());
    let engine_notice = dispatched.clone();
    let (mut engine_rpc, server_task, engine_task) =
        lora_scripted_test_server(true, move |dealer, push| {
            let engine_notice = engine_notice.clone();
            boxed_test_future(async move {
                let load = recv_engine_message(dealer).await;
                let load: rmpv::Value = rmp_serde::from_slice(&load[1]).unwrap();
                send_utility_bool(push, load.as_array().unwrap()[1].as_u64().unwrap(), true).await;

                let _unload = recv_engine_message(dealer).await;
                engine_notice.notify_one();
                std::future::pending::<()>().await;
            })
        })
        .await;
    let dir = tempfile::tempdir().unwrap();
    engine_rpc
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(test_adapter(&dir)),
        })
        .await
        .unwrap();

    let mut unloading_client = engine_rpc.clone();
    let unload = tokio::spawn(async move {
        unloading_client
            .unload_lora(pb::UnloadLoraRequest {
                lora_name: "adapter-a".to_string(),
            })
            .await
    });
    dispatched.notified().await;
    unload.abort();
    assert!(unload.await.unwrap_err().is_cancelled());

    assert_eq!(
        engine_rpc.list_loras(pb::ListLorasRequest {}).await.unwrap_err().code(),
        tonic::Code::FailedPrecondition
    );

    server_task.abort();
    drop(engine_task);
}
