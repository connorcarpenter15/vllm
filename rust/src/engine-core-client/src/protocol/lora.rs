use serde_default::DefaultFromSerde;
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::OpaqueValue;

/// LoRA adapter descriptor carried on the Python EngineCore MessagePack wire.
///
/// This field order matches `vllm.lora.request.LoRARequest`, which is an
/// `array_like=True` msgspec struct.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct LoraRequest {
    pub lora_name: String,
    pub lora_int_id: i64,
    pub lora_path: String,
    #[serde(default)]
    pub base_model_name: Option<String>,
    #[serde(default)]
    pub tensorizer_config_dict: Option<OpaqueValue>,
    #[serde(default)]
    pub load_inplace: bool,
    #[serde(default)]
    pub is_3d_lora_weight: bool,
}

impl LoraRequest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.lora_name.is_empty() {
            return Err("lora_name is required");
        }
        if self.lora_int_id <= 0 {
            return Err("lora_int_id must be positive");
        }
        if self.lora_path.is_empty() {
            return Err("lora_path is required");
        }
        Ok(())
    }
}
