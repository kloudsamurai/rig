use rig::{
    completion::{CompletionModel, ToolDefinition},
    providers::deepseek::{ClientBuilder, DEEPSEEK_CHAT},
};
use serde_json::json;

#[tokio::test]
async fn test_code_assistant_tool_calling() {
    // Skip if no API key is present
    let api_key = match std::env::var("DEEPSEEK_API_KEY") {
        Ok(key) => key,
        Err(_) => return,
    };

    let client = ClientBuilder::new(&api_key).build();
    let model = client.completion_model(DEEPSEEK_CHAT);

    // Define a simple tool for testing
    let test_tool = ToolDefinition {
        name: "verify_solution".to_string(),
        description: "Verify if the provided solution correctly solves the problem".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to verify"
                },
                "is_correct": {
                    "type": "boolean",
                    "description": "Whether the solution is correct"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of why the solution is correct or incorrect"
                }
            },
            "required": ["code", "is_correct", "explanation"]
        }),
    };

    let problem = "Write a function that checks if a string is a palindrome.";

    let request = model
        .completion_request(problem)
        .temperature(0.7)
        .tool(test_tool)
        .build();

    let response = model.completion(request).await.unwrap();

    match response.choice {
        rig::completion::ModelChoice::Message(content) => {
            println!("Got message response: {}", content);
            assert!(!content.is_empty());
        }
        rig::completion::ModelChoice::ToolCall(name, args) => {
            println!("Got tool call: {} with args: {}", name, args);
            assert_eq!(name, "verify_solution");
            assert!(args["code"].as_str().is_some());
            assert!(args["is_correct"].as_bool().is_some());
            assert!(args["explanation"].as_str().is_some());
        }
    }
}
