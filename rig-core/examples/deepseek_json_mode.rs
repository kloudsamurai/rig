use rig::{
    completion::CompletionModel,
    providers::deepseek::{
        ClientBuilder, DeepseekRequestBuilderExt, ResponseFormat, DEEPSEEK_CHAT,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ClientBuilder::new(&api_key).build();

    let model = client.completion_model(DEEPSEEK_CHAT);

    let request = model
        .completion_request("List three European capitals with their countries in JSON format.")
        .temperature(0.7)
        .response_format(ResponseFormat::JsonObject)
        .build();

    let response = model.completion(request).await?;
    println!("JSON Response:\n{:?}", response.choice);

    Ok(())
}
