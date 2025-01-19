//! DeepSeek completion API implementation

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
};

use super::client::Client;

// ================================================================
// DeepSeek Completion API Constants
// ================================================================
/// Default DeepSeek API base URL
pub const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com/betad";
/// `deepseek-chat` completion model
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";

// ================================================================
// Response Types
// ================================================================
#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        match response.choices.first() {
            Some(Choice {
                message: Message { content, .. },
                ..
            }) => Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(content.clone()),
                raw_response: response,
            }),
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message".into(),
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {}\nCompletion tokens: {}\nTotal tokens: {}",
            self.prompt_tokens, self.completion_tokens, self.total_tokens
        )
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Completion Model Implementation
// ================================================================
#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let prompt_with_context = completion_request.prompt_with_context();

        // Convert chat history to DeepSeek message format
        let mut messages = completion_request
            .chat_history
            .into_iter()
            .map(|msg| Message {
                role: msg.role,
                content: msg.content,
            })
            .collect::<Vec<_>>();

        // Add the current prompt
        messages.push(Message {
            role: "user".to_string(),
            content: prompt_with_context,
        });

        let mut request = json!({
            "model": self.model,
            "messages": messages,
        });

        // Add temperature if specified
        if let Some(temperature) = completion_request.temperature {
            json_utils::merge_inplace(&mut request, json!({ "temperature": temperature }));
        }

        // DeepSeek doesn't support tools/functions, so we error if they're requested
        if !completion_request.tools.is_empty() {
            return Err(CompletionError::RequestError(
                "DeepSeek does not support function calling".into(),
            ));
        }

        // Add any additional provider-specific parameters
        if let Some(params) = completion_request.additional_params {
            json_utils::merge_inplace(&mut request, params);
        }

        let response = self
            .client
            .post("/v1/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(completion) => {
                    tracing::info!(target: "rig",
                        "DeepSeek completion token usage: {}",
                        completion.usage
                    );
                    completion.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}
