use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_utility_call_unregisters_waiter() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let (received_tx, received_rx) = oneshot::channel();
    let (_shutdown, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        vec![0x00, 0x00],
        |dealer, _push| {
            Box::pin(async move {
                let _utility = recv_engine_message(dealer).await;
                let _ = received_tx.send(());
                std::future::pending::<()>().await;
            })
        },
    );
    let client = std::sync::Arc::new(
        connect_client_with_ipc(
            handshake_test_config(
                handshake_address,
                1,
                "test-model",
                Duration::from_secs(2),
                0,
                None,
            ),
            &ipc,
        )
        .await,
    );
    let call_client = client.clone();
    let call = tokio::spawn(async move {
        call_client.call_utility_per_engine::<bool, _>("add_lora", ()).await
    });

    received_rx.await.unwrap();
    assert_eq!(client.pending_utility_call_count(), 1);
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    assert_eq!(client.pending_utility_call_count(), 0);

    engine_task.abort();
    let client = match std::sync::Arc::try_unwrap(client) {
        Ok(client) => client,
        Err(_) => panic!("utility task retained the client after cancellation"),
    };
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn call_utility_per_engine_preserves_partial_results() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();

    let (shutdown_0, task_0) = spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        vec![0x00, 0x00],
        two_rank_ready(0),
        |dealer, push| {
            Box::pin(async move {
                let utility = recv_engine_message(dealer).await;
                let call_id = decode_value(&utility[1]).as_array().unwrap()[1].as_u64().unwrap();
                send_outputs(
                    push,
                    UtilityCallOutput {
                        engine_index: 0,
                        timestamp: 0.0,
                        output: UtilityOutput {
                            call_id: call_id.into(),
                            failure_message: None,
                            result: Some(utility_result_value(true)),
                        },
                    }
                    .into(),
                )
                .await;
            })
        },
    );
    let (shutdown_1, task_1) = spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        vec![0x01, 0x00],
        two_rank_ready(1),
        |dealer, push| {
            Box::pin(async move {
                let utility = recv_engine_message(dealer).await;
                let call_id = decode_value(&utility[1]).as_array().unwrap()[1].as_u64().unwrap();
                send_outputs(
                    push,
                    UtilityCallOutput {
                        engine_index: 1,
                        timestamp: 0.0,
                        output: UtilityOutput {
                            call_id: call_id.into(),
                            failure_message: Some("rank failed".to_string()),
                            result: None,
                        },
                    }
                    .into(),
                )
                .await;
            })
        },
    );

    let client = connect_client_with_ipc(
        handshake_test_config(
            handshake_address,
            2,
            "test-model",
            Duration::from_secs(2),
            0,
            None,
        ),
        &ipc,
    )
    .await;

    let results = client.call_utility_per_engine::<bool, _>("add_lora", ()).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(matches!(results[0], Ok(true)));
    assert!(matches!(
        &results[1],
        Err(Error::UtilityCallFailed { message, .. }) if message == "rank failed"
    ));

    let _ = shutdown_0.send(());
    let _ = shutdown_1.send(());
    task_0.await.unwrap();
    task_1.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn call_utility_on_engine_targets_selected_nonzero_rank() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();

    let ready = |rank| EngineCoreReadyResponse {
        data_parallel_size: 6,
        data_parallel_rank: rank,
        ..crate::mock_engine::default_ready_response()
    };
    let (shutdown_0, task_0) = spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        vec![0x04, 0x00],
        ready(4),
        |_dealer, _push| Box::pin(async {}),
    );
    let (shutdown_1, task_1) = spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        vec![0x05, 0x00],
        ready(5),
        |dealer, push| {
            Box::pin(async move {
                let utility = recv_engine_message(dealer).await;
                let payload = decode_value(&utility[1]);
                let array = payload.as_array().expect("utility payload");
                assert_eq!(array[2], Value::from("remove_lora"));
                assert_eq!(array[3], Value::Array(vec![Value::from(17)]));
                let call_id = array[1].as_u64().expect("call_id");
                send_outputs(
                    push,
                    UtilityCallOutput {
                        engine_index: 5,
                        timestamp: 0.0,
                        output: UtilityOutput {
                            call_id: call_id.into(),
                            failure_message: None,
                            result: Some(utility_result_value(true)),
                        },
                    }
                    .into(),
                )
                .await;
            })
        },
    );

    let client = connect_client_with_ipc(
        handshake_test_config(
            handshake_address,
            2,
            "test-model",
            Duration::from_secs(2),
            0,
            None,
        ),
        &ipc,
    )
    .await;

    let removed = timeout(
        Duration::from_secs(2),
        client.call_utility_on_engine::<bool, _>(5, "remove_lora", (17u64,)),
    )
    .await
    .expect("targeted utility call timed out")
    .unwrap();
    assert!(removed);

    let _ = shutdown_0.send(());
    let _ = shutdown_1.send(());
    task_0.await.unwrap();
    task_1.await.unwrap();
    client.shutdown().await.unwrap();
}
