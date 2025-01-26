//! DeepSeek API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::deepseek;
//!
//! let client = deepseek::Client::new("YOUR_API_KEY");
//!
//! let model = client.completion_model(deepseek::DEEPSEEK_CHAT);
//! ```

pub mod client;
pub mod completion;

pub use client::{Client, ClientBuilder};
pub use completion::{CompletionModel, DEEPSEEK_API_BASE_URL, DEEPSEEK_CHAT};
