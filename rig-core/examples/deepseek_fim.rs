use rig::{
    completion::CompletionRequest,
    providers::deepseek::{ClientBuilder, DEEPSEEK_CHAT},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new(std::env::var("DEEPSEEK_API_KEY")?).build()?;

    let model = client.completion_model(DEEPSEEK_CHAT);

    let request = CompletionRequest::new(
        // Using FIM marker to indicate where the suffix starts
        "def fibonacci(n):<fim_suffix>    return fib(n-1) + fib(n-2)",
        None,
    );

    let response = model.completion(request).await?;
    println!("FIM Response:\n{}", response.choice);

    Ok(())
}
