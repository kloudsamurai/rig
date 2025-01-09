use rig::{
    completion::{CompletionModel, ToolDefinition},
    embeddings::EmbeddingsBuilder,
    providers::deepseek::{ClientBuilder, DEEPSEEK_CHAT},
    tool::{Tool, ToolEmbedding, ToolSet},
    vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{error::Error, path::PathBuf};

// GitHub Search Tool
#[derive(Debug, Deserialize)]
struct GitHubSearchArgs {
    query: String,
    language: Option<String>,
    max_results: Option<u32>,
}

#[derive(Debug, Serialize)]
struct GitHubRepo {
    name: String,
    url: String,
    description: Option<String>,
    stars: u32,
}

#[derive(Debug, Serialize)]
struct GitHubSearchResult {
    repositories: Vec<GitHubRepo>,
    total_count: u32,
}

#[derive(Debug, thiserror::Error)]
#[error("GitHub search error: {0}")]
struct GitHubError(String);

#[derive(Deserialize, Serialize)]
struct GitHubSearch;

impl Tool for GitHubSearch {
    const NAME: &'static str = "github_search";
    type Error = GitHubError;
    type Args = GitHubSearchArgs;
    type Output = GitHubSearchResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search GitHub repositories for code and projects. Best for finding open source code, libraries, and programming examples.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for GitHub repositories"
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional programming language filter"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "Maximum number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Mock implementation - in reality, would use GitHub's API
        Ok(GitHubSearchResult {
            repositories: vec![GitHubRepo {
                name: "example/repo".to_string(),
                url: "https://github.com/example/repo".to_string(),
                description: Some("An example repository".to_string()),
                stars: 1000,
            }],
            total_count: 1,
        })
    }
}

impl ToolEmbedding for GitHubSearch {
    type InitError = GitHubError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(GitHubSearch)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec![
            "Search GitHub repositories for code and programming examples".into(),
            "Find open source libraries and projects on GitHub".into(),
            "Search for code implementations and solutions on GitHub".into(),
            "Discover programming libraries and frameworks".into(),
        ]
    }

    fn context(&self) -> Self::Context {}
}

// Filesystem Search Tool
#[derive(Debug, Deserialize)]
struct FileSearchArgs {
    pattern: String,
    directory: Option<String>,
    file_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct FileMatch {
    path: String,
    name: String,
    is_directory: bool,
}

#[derive(Debug, Serialize)]
struct FileSearchResult {
    matches: Vec<FileMatch>,
    total_count: usize,
}

#[derive(Debug, thiserror::Error)]
#[error("File search error: {0}")]
struct FileSearchError(String);

#[derive(Deserialize, Serialize)]
struct FileSearch;

impl Tool for FileSearch {
    const NAME: &'static str = "file_search";
    type Error = FileSearchError;
    type Args = FileSearchArgs;
    type Output = FileSearchResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search local files and directories. Best for finding documents, configuration files, and local content.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The search pattern for files and directories"
                    },
                    "directory": {
                        "type": "string",
                        "description": "Optional directory to search in (default: current directory)"
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Optional file type filter (e.g., 'pdf', 'txt')"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Mock implementation - in reality, would use filesystem operations
        Ok(FileSearchResult {
            matches: vec![FileMatch {
                path: "docs/example.txt".to_string(),
                name: "example.txt".to_string(),
                is_directory: false,
            }],
            total_count: 1,
        })
    }
}

impl ToolEmbedding for FileSearch {
    type InitError = FileSearchError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(FileSearch)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec![
            "Search local files and directories".into(),
            "Find documents and configuration files".into(),
            "Search for local content and data files".into(),
            "Locate files by name or type".into(),
        ]
    }

    fn context(&self) -> Self::Context {}
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ClientBuilder::new(&api_key).build();
    let model = client.completion_model(DEEPSEEK_CHAT);

    // Create a toolset with both tools
    let toolset = ToolSet::builder()
        .dynamic_tool(GitHubSearch)
        .dynamic_tool(FileSearch)
        .build();

    // Create embeddings for semantic tool selection
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(toolset.schemas()?)?
        .build()
        .await?;

    // Create vector store for tool selection
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone());
    let index = vector_store.index(model.clone());

    // Create an agent with dynamic tool selection
    let agent = client
        .agent(DEEPSEEK_CHAT)
        .preamble("You are a helpful assistant that can search both GitHub and local files. Choose the most appropriate search tool based on the user's query.")
        .dynamic_tools(1, index, toolset)
        .build();

    // Example queries to test tool selection
    let queries = vec![
        "Find Rust implementations of a B-tree", // Should choose GitHub
        "Look for configuration files in the docs folder", // Should choose filesystem
    ];

    for query in queries {
        println!("\nQuery: {}", query);
        let response = agent.prompt(query).await?;
        println!("Response: {}", response);
    }

    Ok(())
}
