use rig::providers::openai::{self, Client};
use rig_surrealdb::{vector_index::SurrealVectorIndex, IndexConfig, VectorSimilarityFunction};
use std::env;
use std::sync::Arc;
use surrealdb::api::conn::Connection;
use surrealdb::engine::remote::ws::Client as SurrealClient;
use tokio::sync::Mutex;
use tracing_subscriber;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Initialize SurrealDB client
    let surreal_uri = env::var("SURREAL_URI").expect("SURREAL_URI not set");
    let surreal_client = SurrealClient::new(&surreal_uri).await.unwrap();

    let config = IndexConfig::new(
        "vectors".to_string(),
        "embedding".to_string(),
        VectorSimilarityFunction::Cosine,
        2,    // dimensions for this example
        100,  // batch size
        3,    // max retries
        1000, // retry delay ms
    );

    let vector_index = SurrealVectorIndex::new(Arc::new(Mutex::new(surreal_client)), config)
        .await
        .unwrap();

    // Example of handling an error when adding a duplicate ID
    match vector_index
        .create_vector(
            "existing_id",
            vec![0.1, 0.2],
            Some(serde_json::json!({ "text": "Duplicate entry" })),
        )
        .await
    {
        Ok(_) => println!("Vector created successfully."),
        Err(e) => eprintln!("An error occurred: {}", e),
    }

    // Initialize OpenAI embedding model
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = Client::new(&openai_api_key).embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Example of handling an error when querying a non-existent index
    match vector_index.read_vector("nonexistent_id").await {
        Ok(Some(_)) => {
            // Process result...
        }
        Ok(None) => eprintln!("Vector not found"),
        Err(e) => eprintln!("An error occurred: {}", e),
    }
}
