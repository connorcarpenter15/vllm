// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Metadata-only multimodal preparation for encoder-cache consumers.

use std::collections::{BTreeMap, HashMap};

use llm_multimodal::{
    FieldLayout, MediaContentPart, Modality, ModelSpecificValue, PreprocessedEncoderInputs,
};
use ndarray::{ArrayD, IxDyn};
use serde::Deserialize;
use serde_json::Value;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmKwargValue, MmKwargsItem,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::{MultimodalModelInfo, PreparedItem, PreparedMedia, ResolvedMultimodalSpec};
use crate::error::{Result, bail_multimodal, multimodal};

/// One producer-generated encoder-cache item received through transfer params.
///
/// The identifier matches a request media UUID. `num_encoder_tokens` and the
/// flattened metadata map are authoritative outputs of the producer's
/// multimodal processor.
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderCacheItem {
    #[serde(rename = "mm_hash")]
    identifier: String,
    num_encoder_tokens: usize,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

impl MultimodalModelInfo {
    /// Build prompt replacements and engine metadata without fetching media.
    ///
    /// Returns `None` when any request item cannot be matched or reconstructed;
    /// callers then use ordinary raw-media preprocessing for the whole request.
    pub(super) fn prepare_from_encoder_cache(
        &self,
        media_parts: &[MediaContentPart],
        cache_items: &[EncoderCacheItem],
    ) -> Result<Option<Vec<PreparedMedia>>> {
        if cache_items.is_empty() {
            return Ok(None);
        }

        let mut by_identifier = HashMap::with_capacity(cache_items.len());
        for item in cache_items {
            if item.identifier.is_empty()
                || item.num_encoder_tokens == 0
                || item.metadata.is_empty()
                || by_identifier.insert(item.identifier.as_str(), item).is_some()
            {
                return Ok(None);
            }
        }

        let mut images = Vec::new();
        for media_part in media_parts {
            let Some((modality, identifier)) = media_identity(media_part) else {
                return Ok(None);
            };
            // The initial EC handoff contract is image-only. Other modalities
            // keep using their existing raw-media preparation path.
            if modality != Modality::Image {
                return Ok(None);
            }
            let Some(identifier) = identifier else {
                return Ok(None);
            };
            let Some(item) = by_identifier.get(identifier).copied() else {
                return Ok(None);
            };
            images.push(item);
        }

        Ok(Some(vec![self.prepare_cached_images(images)?]))
    }

    fn prepare_cached_images(&self, cache_items: Vec<&EncoderCacheItem>) -> Result<PreparedMedia> {
        let support =
            self.image.as_ref().ok_or_else(|| crate::error::Error::UnsupportedModality {
                modality: Modality::Image.to_string(),
            })?;

        let mut replacements = Vec::with_capacity(cache_items.len());
        let mut items = Vec::with_capacity(cache_items.len());
        for cache_item in cache_items {
            let preprocessed = synthetic_preprocessed(&support.spec, cache_item)?;
            let mut item_replacements =
                support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
            if item_replacements.len() != 1 {
                bail_multimodal!(
                    "expected exactly one prompt replacement for cached image item `{}`, got {}",
                    cache_item.identifier,
                    item_replacements.len()
                );
            }
            replacements.push(item_replacements.pop().unwrap());
            items.push(PreparedItem {
                data: metadata_kwargs(&support.spec, cache_item)?,
                hash: cache_item.identifier.clone(),
                uuid: None,
            });
        }

        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }
}

fn media_identity(part: &MediaContentPart) -> Option<(Modality, Option<&str>)> {
    match part {
        MediaContentPart::ImageUrl { uuid, .. }
        | MediaContentPart::ImageData { uuid, .. }
        | MediaContentPart::ImageEmbeds { uuid, .. } => Some((Modality::Image, uuid.as_deref())),
        MediaContentPart::VideoUrl { uuid, .. } | MediaContentPart::VideoData { uuid, .. } => {
            Some((Modality::Video, uuid.as_deref()))
        }
        MediaContentPart::AudioUrl { uuid, .. } | MediaContentPart::AudioData { uuid, .. } => {
            Some((Modality::Audio, uuid.as_deref()))
        }
        MediaContentPart::Text { .. } => None,
    }
}

fn synthetic_preprocessed(
    spec: &ResolvedMultimodalSpec,
    item: &EncoderCacheItem,
) -> Result<PreprocessedEncoderInputs> {
    let mut model_specific = HashMap::with_capacity(item.metadata.len());
    for (key, value) in &item.metadata {
        let layout = spec.field_layout_for(key).ok_or_else(|| {
            multimodal!("cached metadata field `{key}` is not declared by the model")
        })?;
        if !matches!(layout, FieldLayout::Batched) {
            bail_multimodal!("cached image metadata field `{key}` is not batched");
        }
        let (mut shape, data) = integer_tensor(value)?;
        shape.insert(0, 1);
        model_specific.insert(key.clone(), ModelSpecificValue::IntTensor { data, shape });
    }

    Ok(PreprocessedEncoderInputs {
        encoder_input: ArrayD::zeros(IxDyn(&[0])),
        feature_token_counts: vec![item.num_encoder_tokens],
        // Specs that require original media sizes cannot safely use the
        // metadata-only path unless their published metadata supersedes them.
        item_sizes: Vec::new(),
        model_specific,
    })
}

fn metadata_kwargs(spec: &ResolvedMultimodalSpec, item: &EncoderCacheItem) -> Result<MmKwargsItem> {
    let mut data = MmKwargsItem::new();
    for (key, value) in &item.metadata {
        let layout = spec.field_layout_for(key).ok_or_else(|| {
            multimodal!("cached metadata field `{key}` is not declared by the model")
        })?;
        if !matches!(layout, FieldLayout::Batched) {
            bail_multimodal!("cached image metadata field `{key}` is not batched");
        }
        let (shape, values) = integer_tensor(value)?;
        let value = MmKwargValue::Tensor(
            WireTensor::from_i64(shape, values).map_err(crate::error::Error::Multimodal)?,
        );
        let keep_on_cpu = spec.keep_on_cpu_keys.contains(key);
        data.insert(
            key.clone(),
            MmFieldElem {
                data: Some(value),
                field: MmField::Batched(MmBatchedField { keep_on_cpu }),
            },
        );
    }
    Ok(data)
}

fn integer_tensor(value: &Value) -> Result<(Vec<usize>, Vec<i64>)> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .map(|value| (Vec::new(), vec![value]))
            .ok_or_else(|| multimodal!("cached image metadata must contain int64 values")),
        Value::Array(values) if !values.is_empty() => {
            let mut child_shape = None;
            let mut data = Vec::new();
            for value in values {
                let (shape, mut values) = integer_tensor(value)?;
                if child_shape.as_ref().is_some_and(|expected| expected != &shape) {
                    bail_multimodal!("cached image metadata tensor must be rectangular");
                }
                child_shape.get_or_insert(shape);
                data.append(&mut values);
            }
            let mut shape = vec![values.len()];
            shape.extend(child_shape.unwrap_or_default());
            Ok((shape, data))
        }
        _ => Err(multimodal!(
            "cached image metadata must be a non-empty int64 tensor"
        )),
    }
}
