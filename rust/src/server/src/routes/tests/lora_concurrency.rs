use super::*;

async fn post_load(app: &mut axum::Router, path: &str, load_inplace: bool) -> StatusCode {
    app.call(
        Request::builder()
            .method("POST")
            .uri("/v1/load_lora_adapter")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "lora_name": "adapter-a",
                    "lora_path": path,
                    "load_inplace": load_inplace
                })
                .to_string(),
            ))
            .expect("build load request"),
    )
    .await
    .expect("call load route")
    .status()
}

async fn post_completion(app: &mut axum::Router) -> StatusCode {
    app.call(
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "adapter-a",
                    "prompt": "hello",
                    "max_tokens": 1,
                    "stream": false
                })
                .to_string(),
            ))
            .expect("build completion request"),
    )
    .await
    .expect("call completion route")
    .status()
}

async fn receive_lora_utility(
    dealer: &mut DealerSocket,
    expected_path: &str,
    expected_inplace: bool,
) -> u64 {
    let utility = recv_engine_message(dealer).await;
    assert_eq!(utility[0].as_ref(), &[0x03]);
    let payload = decode_value(&utility[1]).expect("decode utility payload");
    let array = payload.as_array().expect("utility payload array");
    assert_eq!(array[2], Value::from("add_lora"));
    let lora = array[3].as_array().unwrap()[0].as_array().unwrap();
    assert_eq!(lora[0], Value::from("adapter-a"));
    assert_eq!(lora[1], Value::from(1));
    assert_eq!(lora[2], Value::from(expected_path));
    assert_eq!(lora[5], Value::from(expected_inplace));
    array[1].as_u64().expect("call id")
}

async fn reply_utility(push: &mut PushSocket, call_id: u64, result: bool) {
    send_outputs(push, utility_outputs(call_id, utility_result_value(result))).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn successful_replacement_waits_for_active_generation() {
    let (generation_started_tx, generation_started_rx) = tokio::sync::oneshot::channel();
    let (release_generation_tx, release_generation_rx) = tokio::sync::oneshot::channel();
    let (mut app, engine_task) = test_admin_app_with_engine_script(move |dealer, push| {
        boxed_test_future(async move {
            let initial = receive_lora_utility(dealer, "org/adapter-a", false).await;
            reply_utility(push, initial, true).await;

            let add = recv_engine_message(dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
            assert_adapter_a_lora_request(&request);
            let _ = generation_started_tx.send(());

            tokio::select! {
                result = release_generation_rx => result.expect("release generation"),
                _early = recv_engine_message(dealer) => {
                    panic!("replacement reached the engine before generation completed")
                }
            }
            send_outputs(
                push,
                engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
            )
            .await;

            let replacement = receive_lora_utility(dealer, "org/adapter-b", true).await;
            reply_utility(push, replacement, true).await;
        })
    })
    .await;
    assert_eq!(
        post_load(&mut app, "org/adapter-a", false).await,
        StatusCode::OK
    );

    let mut generation_app = app.clone();
    let generation = tokio::spawn(async move { post_completion(&mut generation_app).await });
    generation_started_rx.await.unwrap();

    let mut replacement_app = app.clone();
    let mut replacement =
        tokio::spawn(async move { post_load(&mut replacement_app, "org/adapter-b", true).await });
    assert!(tokio::time::timeout(Duration::from_millis(50), &mut replacement).await.is_err());
    release_generation_tx.send(()).unwrap();

    assert_eq!(generation.await.unwrap(), StatusCode::OK);
    assert_eq!(replacement.await.unwrap(), StatusCode::OK);
    drop(app);
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn rolled_back_replacement_hides_intermediate_adapter_from_generation() {
    let (replacement_started_tx, replacement_started_rx) = tokio::sync::oneshot::channel();
    let (fail_replacement_tx, fail_replacement_rx) = tokio::sync::oneshot::channel();
    let (mut app, engine_task) = test_admin_app_with_engine_script(move |dealer, push| {
        boxed_test_future(async move {
            let initial = receive_lora_utility(dealer, "org/adapter-a", false).await;
            reply_utility(push, initial, true).await;

            let replacement = receive_lora_utility(dealer, "org/adapter-b", true).await;
            let _ = replacement_started_tx.send(());
            tokio::select! {
                result = fail_replacement_rx => result.expect("fail replacement"),
                _early = recv_engine_message(dealer) => {
                    panic!("generation reached the engine during replacement")
                }
            }
            reply_utility(push, replacement, false).await;

            let restore = receive_lora_utility(dealer, "org/adapter-a", true).await;
            reply_utility(push, restore, true).await;

            let add = recv_engine_message(dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
            assert_adapter_a_lora_request(&request);
            send_outputs(
                push,
                engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
            )
            .await;
        })
    })
    .await;
    assert_eq!(
        post_load(&mut app, "org/adapter-a", false).await,
        StatusCode::OK
    );

    let mut replacement_app = app.clone();
    let replacement =
        tokio::spawn(async move { post_load(&mut replacement_app, "org/adapter-b", true).await });
    replacement_started_rx.await.unwrap();

    let mut generation_app = app.clone();
    let mut generation = tokio::spawn(async move { post_completion(&mut generation_app).await });
    assert!(tokio::time::timeout(Duration::from_millis(50), &mut generation).await.is_err());
    fail_replacement_tx.send(()).unwrap();

    assert_eq!(
        replacement.await.unwrap(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(generation.await.unwrap(), StatusCode::OK);
    drop(app);
    engine_task.finish().await;
}
