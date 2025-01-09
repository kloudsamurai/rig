//! DeepSeek client API implementation

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::completion::{CompletionModel, DEEPSEEK_API_BASE_URL};

// ================================================================
// Main DeepSeek Client
// ================================================================

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
}

/// Create a new DeepSeek client using the builder
///
/// # Example
/// ```
/// use rig::providers::deepseek::{ClientBuilder, self};
///
/// // Initialize the DeepSeek client
/// let client = ClientBuilder::new("your-api-key")
///    .build();
/// ```
impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: DEEPSEEK_API_BASE_URL,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn build(self) -> Client {
        Client::new(self.api_key, self.base_url)
    }
}

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new DeepSeek client with the given API key and base URL.
    /// Note, you probably want to use the `ClientBuilder` instead.
    ///
    /// Panics:
    /// - If the API key cannot be parsed as a header value.
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("API key should parse"),
                    );
                    headers
                })
                .build()
                .expect("DeepSeek reqwest client should build"),
        }
    }

    /// Create a new DeepSeek client from the `DEEPSEEK_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set");
        ClientBuilder::new(&api_key).build()
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );
        self.http_client.post(url)
    }

    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an agent builder with the given completion model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::deepseek::{ClientBuilder, self};
    ///
    /// // Initialize the DeepSeek client
    /// let client = ClientBuilder::new("your-api-key").build();
    ///
    /// let agent = client.agent(deepseek::DEEPSEEK_CHAT)
    ///    .preamble("You are a helpful AI assistant.")
    ///    .temperature(0.7)
    ///    .build();
    /// ```
    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    pub fn extractor<T>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel>
    where
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + Clone + 'static,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }

    /// Get the base URL for this client
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}
