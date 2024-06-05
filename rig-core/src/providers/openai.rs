use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError},
    extractor::ExtractorBuilder,
    json_utils,
    model::ModelBuilder,
    rag::RagAgentBuilder,
    vector_store::{NoIndex, VectorStoreIndex},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Main OpenAI Client
// ================================================================
const OPENAI_API_BASE_URL: &str = "https://api.openai.com";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, OPENAI_API_BASE_URL)
    }

    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("OpenAI reqwest client should build"),
        }
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model)
    }

    pub fn embeddings(&self, model: &str) -> embeddings::EmbeddingsBuilder<EmbeddingModel> {
        embeddings::EmbeddingsBuilder::new(self.embedding_model(model))
    }

    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    pub fn model(&self, model: &str) -> ModelBuilder<CompletionModel> {
        ModelBuilder::new(self.completion_model(model))
    }

    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }

    pub fn rag_agent<C: VectorStoreIndex, T: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, C, T> {
        RagAgentBuilder::new(self.completion_model(model))
    }

    pub fn tool_rag_agent<T: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, NoIndex, T> {
        RagAgentBuilder::new(self.completion_model(model))
    }

    pub fn context_rag_agent<C: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, C, NoIndex> {
        RagAgentBuilder::new(self.completion_model(model))
    }
}

// ================================================================
// OpenAI Embedding API
// ================================================================
#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    async fn embed_documents(
        &self,
        documents: Vec<String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let response = self
            .client
            .post("/v1/embeddings")
            .json(&json!({
                "model": self.model,
                "input": documents,
            }))
            .send()
            .await?
            .json::<EmbeddingResponse>()
            .await?;

        // tracing::debug!("Request: {}", serde_json::to_string_pretty(&json!({
        //     "model": self.model,
        //     "input": documents,
        // })).expect("Request should serialize"));

        // let raw_response = self.client.0.post("https://api.openai.com/v1/embeddings")
        //     .json(&json!({
        //         "model": self.model,
        //         "input": documents,
        //     }))
        //     .send()
        //     .await?
        //     .json::<serde_json::Value>()
        //     .await?;

        // tracing::debug!("Response: {}", serde_json::to_string_pretty(&raw_response).expect("Response should serialize"));
        // let response: EmbeddingResponse = serde_json::from_value(raw_response)?;

        Ok(response
            .data
            .into_iter()
            .zip(documents.into_iter())
            .map(|(embedding, document)| embeddings::Embedding {
                document,
                vec: embedding.embedding,
            })
            .collect())
    }
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

// ================================================================
// OpenAI Completion API
// ================================================================
#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(value: CompletionResponse) -> std::prelude::v1::Result<Self, Self::Error> {
        match value.choices.as_slice() {
            [Choice {
                message:
                    Message {
                        content: Some(content),
                        ..
                    },
                ..
            }, ..] => Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(content.to_string()),
                raw_response: value,
            }),
            [Choice {
                message:
                    Message {
                        tool_calls: Some(calls),
                        ..
                    },
                ..
            }, ..] => {
                let call = calls.first().ok_or(CompletionError::ResponseError(
                    "Tool selection is empty".into(),
                ))?;

                Ok(completion::CompletionResponse {
                    choice: completion::ModelChoice::ToolCall(
                        call.function.name.clone(),
                        serde_json::from_str(&call.function.arguments)?,
                    ),
                    raw_response: value,
                })
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a message or tool call".into(),
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
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
    type T = CompletionResponse;

    async fn completion(
        &self,
        mut completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        // Add preamble to chat history (if available)
        let mut full_history = if let Some(preamble) = &completion_request.preamble {
            vec![completion::Message {
                role: "system".into(),
                content: preamble.clone(),
            }]
        } else {
            vec![]
        };

        // Add context documents to chat history
        full_history.append(
            completion_request
                .documents
                .into_iter()
                .map(|doc| completion::Message {
                    role: "system".into(),
                    content: serde_json::to_string(&doc).expect("Document should serialize"),
                })
                .collect::<Vec<_>>()
                .as_mut(),
        );

        // Add context documents to chat history
        full_history.append(&mut completion_request.chat_history);

        // Add context documents to chat history
        full_history.push(completion::Message {
            role: "user".into(),
            content: completion_request.prompt,
        });

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        // println!("Request: {}", serde_json::to_string_pretty(&request).expect("Request should serialize"));

        let response = self
            .client
            .post("/v1/chat/completions")
            .json(
                &if let Some(params) = completion_request.additional_params {
                    json_utils::merge(request, params)
                } else {
                    request
                },
            )
            .send()
            .await?
            .json::<CompletionResponse>()
            .await?;

        // let raw_response = self.client.0.post("https://api.openai.com/v1/chat/completions")
        //     .json(&if let Some(params) = additional_params {json_utils::merge(request, params)} else {request})
        //     .send()
        //     .await?
        //     .json::<serde_json::Value>()
        //     .await?;

        // println!("Response: {}", serde_json::to_string_pretty(&raw_response).expect("Response should serialize"));
        // let response: CompletionResponse = serde_json::from_value(raw_response)?;

        response.try_into()
    }
}