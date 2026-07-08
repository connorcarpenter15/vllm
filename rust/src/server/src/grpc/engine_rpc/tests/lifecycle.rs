use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn health_reports_ready_without_probe() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let response = client
        .health(pb::HealthRequest {
            include_inference_probe: false,
            ..Default::default()
        })
        .await
        .expect("health")
        .into_inner();
    assert_eq!(response.state, pb::HealthState::Ready as i32);
    assert!(response.checks.iter().any(|check| check.name == "engine"));
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_unknown_request_is_idempotent() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let response = client
        .abort(pb::AbortRequest {
            request_id: "not-in-flight".to_string(),
            ..Default::default()
        })
        .await
        .expect("abort")
        .into_inner();
    assert_eq!(response.status, pb::AbortStatus::Aborted as i32);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_all_is_unsupported() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let response = client
        .abort(pb::AbortRequest {
            abort_all: true,
            ..Default::default()
        })
        .await
        .expect("abort")
        .into_inner();
    assert_eq!(response.status, pb::AbortStatus::Unsupported as i32);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_empty_request_id_is_invalid() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let status = client
        .abort(pb::AbortRequest::default())
        .await
        .expect_err("empty request_id should be rejected");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn drain_completes_when_idle() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let response = client.drain(pb::DrainRequest {}).await.expect("drain").into_inner();
    assert_eq!(response.state, pb::DrainState::Complete as i32);
    assert_eq!(response.in_flight_requests, 0);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn drain_stops_admission_and_updates_health() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    client.drain(pb::DrainRequest {}).await.expect("drain");
    let status = client
        .generate(base_request())
        .await
        .expect_err("draining engine must reject new requests");
    assert_eq!(status.code(), tonic::Code::Unavailable);
    let health = client.health(pb::HealthRequest::default()).await.unwrap().into_inner();
    assert_eq!(health.state, pb::HealthState::Draining as i32);
    server_task.abort();
}
