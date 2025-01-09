use rig::{
    completion::{CompletionModel, ToolDefinition},
    providers::deepseek::{ClientBuilder, DEEPSEEK_CHAT},
    tool::{Tool, ToolEmbedding},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

#[derive(Debug, Deserialize)]
struct CodeExecutionArgs {
    code: String,
    test_cases: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CodeExecutionResult {
    output: String,
    test_results: Vec<bool>,
}

#[derive(Debug, thiserror::Error)]
#[error("Code execution error: {0}")]
struct CodeExecutionError(String);

#[derive(Debug, thiserror::Error)]
#[error("Initialization error")]
struct InitError;

/// A tool that can execute Python code and run test cases
#[derive(Deserialize, Serialize)]
struct PythonExecutor;

impl Tool for PythonExecutor {
    const NAME: &'static str = "python_executor";
    type Error = CodeExecutionError;
    type Args = CodeExecutionArgs;
    type Output = CodeExecutionResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Execute Python code and run test cases to verify the solution"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete Python code solution to execute"
                    },
                    "test_cases": {
                        "type": "array",
                        "description": "Test cases to verify the solution",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["code", "test_cases"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // In a real implementation, this would use a sandboxed environment
        // For this example, we'll just return mock results
        Ok(CodeExecutionResult {
            output: format!("Executed code:\n{}", args.code),
            test_results: args.test_cases.iter().map(|_| true).collect(),
        })
    }
}

impl ToolEmbedding for PythonExecutor {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(PythonExecutor)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec![
            "Execute Python code solutions".into(),
            "Run Python test cases".into(),
            "Verify Python code correctness".into(),
        ]
    }

    fn context(&self) -> Self::Context {}
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ClientBuilder::new(&api_key).build();
    let model = client.completion_model(DEEPSEEK_CHAT);

    // The coding problem
    let problem = r#"
Write a function that finds the longest palindromic substring in a given string.
For example:
Input: "babad"
Output: "bab" or "aba"
Input: "cbbd"
Output: "bb"

Please implement the solution and provide test cases to verify it works correctly.
"#;

    let request = model
        .completion_request(problem)
        .temperature(0.7)
        .tool(PythonExecutor)
        .build();

    let response = model.completion(request).await?;

    match response.choice {
        rig::completion::ModelChoice::Message(content) => {
            println!("Response:\n{}", content);
        }
        rig::completion::ModelChoice::ToolCall(name, args) => {
            println!("Tool called: {}", name);

            // Parse and execute the code
            let executor = PythonExecutor;
            let args: CodeExecutionArgs = serde_json::from_value(args)?;
            let result = executor.call(args).await?;

            println!("\nExecution output:");
            println!("{}", result.output);
            println!("\nTest results:");
            for (i, passed) in result.test_results.iter().enumerate() {
                println!(
                    "Test {}: {}",
                    i + 1,
                    if *passed { "PASSED" } else { "FAILED" }
                );
            }
        }
    }

    Ok(())
}
