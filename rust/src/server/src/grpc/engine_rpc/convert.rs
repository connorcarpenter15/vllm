//! Conversion between the private engine RPC protobuf and internal types.

mod kv;
mod media;
mod request;
mod response;

pub use kv::{mark_prefill_request, role_from_kv_role, validate_disaggregated_request};
pub use media::media_parts_from_request;
pub use request::to_text_request;
pub use response::{error_response, event_to_responses};

#[cfg(test)]
mod tests;
