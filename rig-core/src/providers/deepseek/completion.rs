//! DeepSeek completion API implementation

use std::iter;

use crate::{
    completion::{self, CompletionError},
    json_utils,
};

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::Client;

// ================================================================
// DeepSeek Completion API
// ================================================================

pub const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";

/// Response format options for the completion API
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ResponseFormat {
    /// Return response as text (default)
    Text,
    /// Return response as a JSON object
    JsonObject,
}

impl Default for ResponseFormat {
    fn default() -> Self {
        Self::Text
    }
}

/// Extension trait for CompletionRequestBuilder to add Deepseek-specific functionality
pub trait DeepseekRequestBuilderExt {
    /// Set the response format for the completion request
    fn response_format(self, format: ResponseFormat) -> Self;
}

impl<'a> completion::CompletionRequestBuilder<CompletionModel> {
    /// Set the response format for the completion request
    pub fn response_format(self, format: ResponseFormat) -> Self {
        self.additional_params(json!({
            "response_format": format
        }))
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Success(T),
    Error(ApiErrorResponse),
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChoiceObject>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChoiceObject {
    pub message: ChoiceMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub index: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChoiceMessage {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolUse>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    #[serde(rename = "prompt_cache_hit_tokens")]
    pub cache_hit_tokens: Option<i64>,
    #[serde(rename = "prompt_cache_miss_tokens")]
    pub cache_miss_tokens: Option<i64>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        let first_choice = resp
            .choices
            .get(0)
            .ok_or_else(|| CompletionError::ResponseError("No choices found".to_owned()))?;

        // If there's a tool call
        if let Some(tool_calls) = &first_choice.message.tool_calls {
            if let Some(tool_call) = tool_calls.get(0) {
                return Ok(completion::CompletionResponse {
                    choice: completion::ModelChoice::ToolCall(
                        tool_call.name.clone(),
                        tool_call.arguments.clone(),
                    ),
                    raw_response: resp,
                });
            }
        }

        // Otherwise, check content
        if let Some(text) = &first_choice.message.content {
            Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(text.clone()),
                raw_response: resp,
            })
        } else {
            Err(CompletionError::ResponseError(
                "No content or tool call in response".to_owned(),
            ))
        }
    }
}

// Model capabilities
#[derive(Debug, Clone, Copy)]
pub struct ModelCapabilities {
    pub supports_fim: bool,
    pub supports_prefix_completion: bool,
    pub max_tokens: usize,
    pub supports_parallel_function_calling: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            supports_fim: true,
            supports_prefix_completion: true,
            max_tokens: 8192, // 8K output tokens in beta
            supports_parallel_function_calling: true,
        }
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
    capabilities: ModelCapabilities,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
            capabilities: ModelCapabilities::default(),
        }
    }

    pub fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    async fn completion(
        &self,
        request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        // Check if this is a FIM request
        if let Some(suffix_marker) = request.prompt_with_context().find("<fim_suffix>") {
            // Handle FIM request
            let content = request.prompt_with_context();
            let (prompt, rest) = content.split_at(suffix_marker);
            let suffix = rest.trim_start_matches("<fim_suffix>").trim();

            let mut body = json!({
                "model": self.model,
                "prompt": prompt.trim(),
                "suffix": suffix,
                "max_tokens": request.max_tokens.unwrap_or(1024),
                "temperature": request.temperature.unwrap_or(0.7),
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "echo": false,
            });

            if let Some(params) = &request.additional_params {
                json_utils::merge_inplace(&mut body, params.clone());
            }

            let resp = self
                .client
                .post("/beta/completions")
                .json(&body)
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

            // Handle FIM response...
            if resp.status().is_success() {
                let ds_resp = resp
                    .json::<ApiResponse<CompletionResponse>>()
                    .await
                    .map_err(|e| CompletionError::ResponseError(e.to_string()))?;

                match ds_resp {
                    ApiResponse::Success(ok) => ok.try_into(),
                    ApiResponse::Error(e) => {
                        Err(CompletionError::ProviderError(e.error.message.clone()))
                    }
                }
            } else {
                let error = resp
                    .json::<ApiErrorResponse>()
                    .await
                    .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
                Err(CompletionError::ProviderError(error.error.message))
            }
        } else {
            // Regular chat completion with parallel function calling enabled
            let messages: Vec<_> = request
                .chat_history
                .iter()
                .map(|m| json!({ "role": m.role, "content": m.content }))
                .chain(iter::once(json!({
                    "role": "user",
                    "content": request.prompt_with_context()
                })))
                .collect();

            let mut body = json!({
                "model": self.model,
                "messages": messages,
                "max_tokens": request.max_tokens.unwrap_or(2048),
                "temperature": request.temperature.unwrap_or(0.7),
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            });

            // Handle tool/function calling
            if !request.tools.is_empty() {
                let tools: Vec<_> = request
                    .tools
                    .into_iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters,
                            }
                        })
                    })
                    .collect();
                body["tools"] = tools.into();
                body["parallel_function_calling"] = true.into();
            }

            // Add any additional parameters
            if let Some(params) = &request.additional_params {
                json_utils::merge_inplace(&mut body, params.clone());
            }

            let resp = self
                .client
                .post("/beta/chat/completions")
                .json(&body)
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

            if resp.status().is_success() {
                let ds_resp = resp
                    .json::<ApiResponse<CompletionResponse>>()
                    .await
                    .map_err(|e| CompletionError::ResponseError(e.to_string()))?;

                match ds_resp {
                    ApiResponse::Success(ok) => ok.try_into(),
                    ApiResponse::Error(e) => {
                        Err(CompletionError::ProviderError(e.error.message.clone()))
                    }
                }
            } else {
                let error = resp
                    .json::<ApiErrorResponse>()
                    .await
                    .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
                Err(CompletionError::ProviderError(error.error.message))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::CompletionRequest;
    use serde_json::json;

    #[test]
    fn test_model_capabilities() {
        // All capabilities enabled by default since we're using beta
        let client = Client::new("test", "https://api.deepseek.com");
        let model = CompletionModel::new(client, DEEPSEEK_CHAT);
        let caps = model.capabilities();
        assert!(caps.supports_fim);
        assert!(caps.supports_prefix_completion);
        assert_eq!(caps.max_tokens, 8192);
        assert!(caps.supports_parallel_function_calling);
    }

    #[test]
    fn test_response_format_serialization() {
        let text_format = ResponseFormat::Text;
        let json_format = ResponseFormat::JsonObject;

        let serialized_text = serde_json::to_value(text_format).unwrap();
        let serialized_json = serde_json::to_value(json_format).unwrap();

        assert_eq!(serialized_text, json!({"type": "text"}));
        assert_eq!(serialized_json, json!({"type": "json_object"}));
    }

    #[test]
    fn test_request_builder_json_mode() {
        let client = Client::new("test", "https://api.deepseek.com");
        let model = CompletionModel::new(client, DEEPSEEK_CHAT);

        let request = model
            .completion_request("test")
            .response_format(ResponseFormat::JsonObject)
            .build();

        let expected_format = json!({"type": "json_object"});
        assert_eq!(
            request
                .additional_params
                .unwrap()
                .get("response_format")
                .unwrap(),
            &expected_format
        );
    }

    #[test]
    fn test_response_try_from() {
        // Test message response
        let resp = CompletionResponse {
            id: "test".to_string(),
            model: DEEPSEEK_CHAT.to_string(),
            choices: vec![ChoiceObject {
                message: ChoiceMessage {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
                index: 0,
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cache_hit_tokens: Some(5),
                cache_miss_tokens: Some(5),
            }),
        };

        let rig_resp = completion::CompletionResponse::try_from(resp).unwrap();
        assert!(matches!(
            rig_resp.choice,
            completion::ModelChoice::Message(_)
        ));

        // Test tool call response
        let resp = CompletionResponse {
            id: "test".to_string(),
            model: DEEPSEEK_CHAT.to_string(),
            choices: vec![ChoiceObject {
                message: ChoiceMessage {
                    content: None,
                    tool_calls: Some(vec![ToolUse {
                        id: "1".to_string(),
                        name: "test_tool".to_string(),
                        arguments: json!({"arg": "value"}),
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
                index: 0,
            }],
            usage: None,
        };

        let rig_resp = completion::CompletionResponse::try_from(resp).unwrap();
        assert!(matches!(
            rig_resp.choice,
            completion::ModelChoice::ToolCall(_, _)
        ));
    }

    #[test]
    fn test_error_response() {
        let error_resp = ApiErrorResponse {
            error: ErrorDetail {
                message: "test error".to_string(),
                r#type: "invalid_request".to_string(),
                param: None,
                code: None,
            },
        };

        let api_resp: ApiResponse<CompletionResponse> = ApiResponse::Error(error_resp);
        assert!(matches!(api_resp, ApiResponse::Error(_)));
    }

    #[test]
    fn test_invalid_response() {
        // Test response with no choices
        let resp = CompletionResponse {
            id: "test".to_string(),
            model: DEEPSEEK_CHAT.to_string(),
            choices: vec![],
            usage: None,
        };
        assert!(matches!(
            completion::CompletionResponse::try_from(resp),
            Err(CompletionError::ResponseError(_))
        ));

        // Test response with no content and no tool calls
        let resp = CompletionResponse {
            id: "test".to_string(),
            model: DEEPSEEK_CHAT.to_string(),
            choices: vec![ChoiceObject {
                message: ChoiceMessage {
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
                index: 0,
            }],
            usage: None,
        };
        assert!(matches!(
            completion::CompletionResponse::try_from(resp),
            Err(CompletionError::ResponseError(_))
        ));
    }
}
