use rig::{
    completion::CompletionModel,
    providers::deepseek::{ClientBuilder, DEEPSEEK_CHAT},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ClientBuilder::new(&api_key).build();

    let model = client.completion_model(DEEPSEEK_CHAT);

    let request = model
        .completion_request("What is the capital of France?")
        .build();

    let response = model.completion(request).await?;
    println!("Response: {:?}", response.choice);

    Ok(())
}
