use rig::providers::openai::{Client, TEXT_EMBEDDING_ADA_002};
use rig::vector_store::VectorStoreError;
use rig_surrealdb::{vector_index::*, SurrealClient};
use serde::Deserialize;
use std::env;
use tracing_subscriber;

#[derive(Deserialize)]
struct Document {
    title: String,
    content: String,
}

#[tokio::main]
async fn main() -> Result<(), VectorStoreError> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Initialize clients
    let surreal_uri = env::var("SURREAL_URI").expect("SURREAL_URI not set");
    let client = SurrealClient::new(surreal_uri)?;

    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = Client::new(&openai_api_key).embedding_model(TEXT_EMBEDDING_ADA_002);

    // Add sample documents to the database
    // ... (code to add documents with embeddings)

    // Create a vector index
    // ... (code to create vector index)

    // Get the vector index
    let index = client
        .get_index(model, "document_index", SearchParams::default())
        .await?;

    // Perform hybrid search
    let results = index
        .hybrid_search::<Document>(
            "machine learning trends",
            10,
            "documents",
            SearchParams::default(),
        )
        .await?;

    // Handle the results
    for (score, id, doc) in results {
        println!("ID: {}, Score: {:.4}, Title: {}", id, score, doc.title);
    }

    Ok(())
}
