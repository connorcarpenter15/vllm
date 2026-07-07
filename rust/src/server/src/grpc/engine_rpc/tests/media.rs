use super::*;
use vllm_chat::MediaContentPart;

fn image_item(source: pb::media_item::Source) -> pb::MediaItem {
    pb::MediaItem {
        modality: pb::Modality::Image as i32,
        source: Some(source),
        ..Default::default()
    }
}

#[test]
fn media_parts_maps_url_data_uri_and_raw_bytes() {
    let media = vec![
        pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::Url("http://h/a.png".to_string())),
            uuid: "uid-1".to_string(),
            ..Default::default()
        },
        pb::MediaItem {
            modality: pb::Modality::Unspecified as i32,
            source: Some(pb::media_item::Source::DataUri(
                "data:image/png;base64,AAAA".to_string(),
            )),
            ..Default::default()
        },
        pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::RawBytes(vec![1, 2, 3])),
            mime_type: "image/png".to_string(),
            ..Default::default()
        },
    ];

    let parts = super::super::convert::media_parts_from_request(&media).expect("convert media");
    assert_eq!(parts.len(), 3);

    match &parts[0] {
        MediaContentPart::ImageUrl { url, detail, uuid } => {
            assert_eq!(url, "http://h/a.png");
            assert!(detail.is_none());
            assert_eq!(uuid.as_deref(), Some("uid-1"));
        }
        _ => panic!("part 0 should be an ImageUrl from a url source"),
    }
    match &parts[1] {
        MediaContentPart::ImageUrl { url, uuid, .. } => {
            assert_eq!(url, "data:image/png;base64,AAAA");
            assert!(uuid.is_none());
        }
        _ => panic!("part 1 should be an ImageUrl from a data_uri source"),
    }
    match &parts[2] {
        MediaContentPart::ImageData {
            data, mime_type, ..
        } => {
            assert_eq!(data, &[1, 2, 3]);
            assert_eq!(mime_type.as_deref(), Some("image/png"));
        }
        _ => panic!("part 2 should be ImageData from a raw_bytes source"),
    }
}

#[test]
fn media_parts_empty_input_yields_no_parts() {
    assert!(super::super::convert::media_parts_from_request(&[]).unwrap().is_empty());
}

#[test]
fn media_parts_rejects_item_without_source() {
    let media = vec![pb::MediaItem {
        modality: pb::Modality::Image as i32,
        source: None,
        ..Default::default()
    }];
    let status = super::super::convert::media_parts_from_request(&media)
        .expect_err("a media item with no source must be rejected");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
}

#[test]
fn media_parts_rejects_video_modality_in_v1() {
    let media = vec![pb::MediaItem {
        modality: pb::Modality::Video as i32,
        source: Some(pb::media_item::Source::Url("http://h/clip.mp4".to_string())),
        ..Default::default()
    }];
    let status = super::super::convert::media_parts_from_request(&media)
        .expect_err("video modality must be rejected in v1");
    assert_eq!(status.code(), tonic::Code::Unimplemented);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn generate_with_media_requires_token_ids_input() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "req-media-text".to_string(),
            model: "test-model".to_string(),
            input: Some(pb::generate_request::Input::Prompt("hi".to_string())),
            media: vec![image_item(pb::media_item::Source::Url(
                "http://h/a.png".to_string(),
            ))],
            stream: true,
            ..Default::default()
        })
        .await
        .expect_err("media with a text prompt should be rejected");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn generate_with_media_on_text_only_backend_fails_closed() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "req-media-textonly".to_string(),
            model: "test-model".to_string(),
            input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
            media: vec![image_item(pb::media_item::Source::Url(
                "http://h/a.png".to_string(),
            ))],
            stream: true,
            ..Default::default()
        })
        .await
        .expect_err("media on a text-only backend should fail closed");

    assert_eq!(status.code(), tonic::Code::Internal);
    server_task.abort();
}
