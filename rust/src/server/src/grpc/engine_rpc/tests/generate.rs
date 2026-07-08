use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streams_tokens_then_finishes() {
    let (mut client, server_task, engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let stream = client.generate(base_request()).await.expect("generate").into_inner();
    let responses: Vec<pb::GenerateResponse> =
        stream.map(|result| result.expect("stream item")).collect().await;

    assert!(responses.iter().all(|response| response.request_id == "req-1"));
    let text: String = responses
        .iter()
        .filter_map(|response| match &response.event {
            Some(pb::generate_response::Event::Token(token)) => Some(token.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "hi");
    let finished = responses
        .iter()
        .rev()
        .find_map(|response| match &response.event {
            Some(pb::generate_response::Event::Finished(finished)) => {
                Some((finished, response.usage.as_ref()))
            }
            _ => None,
        })
        .expect("finished event present");
    assert_eq!(finished.0.reason, pb::FinishReason::Stop as i32);
    let usage = finished.1.expect("usage on terminal response");
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (5, 3, 8)
    );

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn accepts_token_ids_input() {
    let (mut client, server_task, engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let stream = client
        .generate(pb::GenerateRequest {
            request_id: "req-tok".to_string(),
            model: "test-model".to_string(),
            input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
            stream: true,
            ..Default::default()
        })
        .await
        .expect("generate")
        .into_inner();
    let responses: Vec<pb::GenerateResponse> =
        stream.map(|result| result.expect("stream item")).collect().await;
    let text: String = responses
        .iter()
        .filter_map(|response| match &response.event {
            Some(pb::generate_response::Event::Token(token)) => Some(token.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "hi");
    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn rejects_missing_input() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let status = client
        .generate(pb::GenerateRequest {
            request_id: "req-no-input".to_string(),
            model: "test-model".to_string(),
            input: None,
            ..Default::default()
        })
        .await
        .expect_err("should fail without input");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn rejects_wrong_model() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let status = client
        .generate(pb::GenerateRequest {
            request_id: "req-wrong".to_string(),
            model: "other-model".to_string(),
            input: Some(pb::generate_request::Input::Prompt("hi".to_string())),
            ..Default::default()
        })
        .await
        .expect_err("should fail with wrong model");
    assert_eq!(status.code(), tonic::Code::NotFound);
    server_task.abort();
}
