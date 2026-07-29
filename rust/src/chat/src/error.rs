// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::{AsReport as _, Macro};

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
pub enum Error {
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    #[error("cannot continue the final message when the last message is not from the assistant")]
    ContinueFinalAssistantWithoutFinalAssistant,
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    #[error("multimodal input is not supported by this chat renderer")]
    UnsupportedMultimodalRenderer,
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    #[error("`{modality}` input is not supported by this model")]
    UnsupportedModality { modality: String },
    #[error("multimodal preprocessing error: {0}")]
    Multimodal(#[message] String),
    #[error(transparent)]
    MediaConnector(#[from] llm_multimodal::MediaConnectorError),
    #[error(transparent)]
    MediaTracker(#[from] llm_multimodal::MultiModalError),
    #[error("{kind} parsing is not available for model `{model_id}`")]
    ParserUnavailableForModel {
        kind: &'static str,
        model_id: String,
    },
    #[error("{kind} parsing is disabled by frontend configuration")]
    ParserDisabled { kind: &'static str },
    #[error(
        "{kind} parser `{name}` is not registered{}",
        available_parser_hint(.available_names)
    )]
    ParserUnavailableByName {
        kind: &'static str,
        name: String,
        available_names: Vec<String>,
    },
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        kind: &'static str,
        name: String,
        #[source]
        error: BoxedError,
    },
    #[error(
        "gpt_oss uses native Harmony output parsing; generic {kind} parser override `{selection}` is not supported"
    )]
    HarmonyParserOverrideUnsupported {
        kind: &'static str,
        selection: String,
    },
    #[error("harmony output parsing failed")]
    HarmonyOutputParsing {
        #[source]
        error: BoxedError,
    },
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong { max_model_len: u32, prompt_len: u32 },
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant { message: String },
    #[error("failed to build structural tag: {message}")]
    StructuralTag { message: String },
    #[error(transparent)]
    Text(#[from] vllm_text::Error),
    #[error(transparent)]
    Tokenizer(#[from] vllm_tokenizer::TokenizerError),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MediaErrorKind {
    InvalidInput,
    Unavailable,
    Internal,
}

impl Error {
    /// Whether this error represents invalid user request parameters.
    pub fn is_request_validation_error(&self) -> bool {
        match self {
            Self::PromptTooLong { .. } => true,
            Self::Text(error) => error.is_request_validation_error(),
            Self::UnsupportedMultimodalRenderer
            | Self::UnsupportedMultimodalContent(_)
            | Self::UnsupportedModality { .. } => true,
            _ => self.media_error_kind() == Some(MediaErrorKind::InvalidInput),
        }
    }

    /// Whether this error represents a transient media-access failure.
    pub fn is_unavailable_error(&self) -> bool {
        self.media_error_kind() == Some(MediaErrorKind::Unavailable)
    }

    /// A diagnostic suitable for server logs that does not expose media URLs
    /// or payloads.
    pub fn media_diagnostic(&self) -> String {
        use llm_multimodal::MultiModalError;

        match self {
            Self::MediaConnector(error) => media_connector_diagnostic(error),
            Self::MediaTracker(MultiModalError::Media(error)) => media_connector_diagnostic(error),
            Self::MediaTracker(MultiModalError::UnsupportedContent(kind)) => {
                format!("unsupported multimodal content: {kind}")
            }
            Self::MediaTracker(MultiModalError::Join(error)) => {
                format!("multimodal task failed: {error}")
            }
            Self::MediaTracker(MultiModalError::Validation(_)) => {
                "multimodal validation failed".to_string()
            }
            _ => self.to_report_string(),
        }
    }

    fn media_error_kind(&self) -> Option<MediaErrorKind> {
        use llm_multimodal::MultiModalError;

        match self {
            Self::MediaConnector(error) | Self::MediaTracker(MultiModalError::Media(error)) => {
                Some(media_connector_error_kind(error))
            }
            Self::MediaTracker(
                MultiModalError::UnsupportedContent(_) | MultiModalError::Validation(_),
            ) => Some(MediaErrorKind::InvalidInput),
            Self::MediaTracker(MultiModalError::Join(_)) => Some(MediaErrorKind::Internal),
            _ => None,
        }
    }
}

fn media_connector_error_kind(error: &llm_multimodal::MediaConnectorError) -> MediaErrorKind {
    use llm_multimodal::MediaConnectorError;

    match error {
        MediaConnectorError::UnsupportedScheme(_)
        | MediaConnectorError::InvalidUrl(_)
        | MediaConnectorError::DisallowedDomain(_)
        | MediaConnectorError::DisallowedLocalPath(_)
        | MediaConnectorError::Base64Decode(_)
        | MediaConnectorError::DataUrl(_)
        | MediaConnectorError::PayloadTooLarge { .. }
        | MediaConnectorError::Image(_)
        | MediaConnectorError::AudioDecode(_)
        | MediaConnectorError::VideoDecode(_) => MediaErrorKind::InvalidInput,
        MediaConnectorError::Http(_)
        | MediaConnectorError::Io(_)
        | MediaConnectorError::Timeout(_) => MediaErrorKind::Unavailable,
        MediaConnectorError::Blocking(_) => MediaErrorKind::Internal,
    }
}

fn media_connector_diagnostic(error: &llm_multimodal::MediaConnectorError) -> String {
    use llm_multimodal::MediaConnectorError;

    match error {
        MediaConnectorError::UnsupportedScheme(scheme) => {
            format!("unsupported media scheme: {scheme}")
        }
        MediaConnectorError::InvalidUrl(_) => "invalid media URL".to_string(),
        MediaConnectorError::DisallowedDomain(_) => "media domain is not allowed".to_string(),
        MediaConnectorError::DisallowedLocalPath(_) => {
            "local media path is not allowed".to_string()
        }
        MediaConnectorError::Http(error) => match error.status() {
            Some(status) => format!("HTTP media fetch failed with status {status}"),
            None if error.is_timeout() => "HTTP media fetch timed out".to_string(),
            None if error.is_connect() => "HTTP media connection failed".to_string(),
            None => "HTTP media fetch failed".to_string(),
        },
        MediaConnectorError::Io(error) => {
            format!("media I/O failed with kind {:?}", error.kind())
        }
        MediaConnectorError::Base64Decode(error) => {
            format!("base64 media decoding failed: {error}")
        }
        MediaConnectorError::DataUrl(_) => "data URI parsing failed".to_string(),
        MediaConnectorError::PayloadTooLarge { media, limit } => {
            format!("{media} payload exceeds the configured limit of {limit} bytes")
        }
        MediaConnectorError::Blocking(error) => {
            format!("media blocking task failed: {error}")
        }
        MediaConnectorError::Image(error) => format!("image decoding failed: {error}"),
        MediaConnectorError::AudioDecode(_) => "audio decoding failed".to_string(),
        MediaConnectorError::VideoDecode(_) => "video decoding failed".to_string(),
        MediaConnectorError::Timeout(duration) => {
            format!("media fetch timed out after {duration:?}")
        }
    }
}

impl From<llm_multimodal::TransformError> for Error {
    fn from(error: llm_multimodal::TransformError) -> Self {
        Self::Multimodal(error.to_report_string())
    }
}

impl From<llm_multimodal::registry::ModelRegistryError> for Error {
    fn from(error: llm_multimodal::registry::ModelRegistryError) -> Self {
        Self::Multimodal(error.to_report_string())
    }
}

/// Format the available-parser suffix used in user-facing error messages.
fn available_parser_hint(available_names: &[String]) -> String {
    if available_names.is_empty() {
        String::new()
    } else {
        format!(" (choose from: {})", available_names.join(", "))
    }
}
