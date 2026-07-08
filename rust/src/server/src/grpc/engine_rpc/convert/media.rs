use tonic::Status;
use vllm_chat::MediaContentPart;

use super::super::pb;

pub fn media_parts_from_request(media: &[pb::MediaItem]) -> Result<Vec<MediaContentPart>, Status> {
    let mut parts = Vec::with_capacity(media.len());
    for item in media {
        let modality = pb::Modality::try_from(item.modality).map_err(|_| {
            Status::invalid_argument(format!("unknown media modality {}", item.modality))
        })?;
        match modality {
            pb::Modality::Image | pb::Modality::Unspecified => {}
            other => {
                return Err(Status::unimplemented(format!(
                    "media modality {other:?} is not supported by the vLLM engine RPC service (image only in v1)"
                )));
            }
        }
        let uuid = (!item.uuid.is_empty()).then(|| item.uuid.clone());
        let part = match item.source.as_ref() {
            Some(pb::media_item::Source::Url(url)) => MediaContentPart::ImageUrl {
                url: url.clone(),
                detail: None,
                uuid,
            },
            Some(pb::media_item::Source::DataUri(uri)) => MediaContentPart::ImageUrl {
                url: uri.clone(),
                detail: None,
                uuid,
            },
            Some(pb::media_item::Source::RawBytes(bytes)) => MediaContentPart::ImageData {
                data: bytes.clone(),
                mime_type: (!item.mime_type.is_empty()).then(|| item.mime_type.clone()),
                uuid,
                detail: None,
            },
            None => {
                return Err(Status::invalid_argument(
                    "media item has no source (expected url, data_uri, or raw_bytes)",
                ));
            }
        };
        parts.push(part);
    }
    Ok(parts)
}
