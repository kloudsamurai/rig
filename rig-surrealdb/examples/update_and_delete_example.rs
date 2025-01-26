use rig::providers::openai;
use rig_surrealdb::SurrealClient;
use rig_surrealdb::VectorStoreError;
use std::env;
use std::future::IntoFuture;
use surrealdb::engine::remote::ws::Ws;
use surrealdb::Surreal;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), VectorStoreError> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Initialize SurrealDB client
    let url = env::var("SURREAL_URI").expect("SURREAL_URI not set");
    let endpoint = format!("ws://{}:{}", url, 8000);
    let client: SurrealClient = SurrealClient::new(
        Surreal::new::<Ws>(endpoint.as_str())
            .into_future()
            .await
            .unwrap(),
    );

    // Initialize OpenAI embedding model
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model =
        openai::Client::new(&openai_api_key).embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    println!("Available client methods:");
    println!("- new() -> Creates new SurrealClient instance");
    println!("- get_instance() -> Gets DB instance");
    println!("- get_namespace() -> Gets namespace");
    println!("- get_database() -> Gets database");
    println!("- use_namespace() -> Changes namespace");
    println!("- use_database() -> Changes database");
    println!("- signin() -> Signs in with credentials");
    println!("- query() -> Executes raw query");
    println!("- insert() -> Inserts record");
    println!("- select() -> Selects records");
    println!("- update() -> Updates record");
    println!("- delete() -> Deletes record");

    Ok(())
}
