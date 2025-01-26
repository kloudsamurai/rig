use rig::{
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};
use rig_surrealdb::{vector_index::*, SurrealClient};
use serde::Deserialize;
use std::env;
use tracing_subscriber;

#[derive(Deserialize)]
struct Article {
    title: String,
    content: String,
    category: String,
    views: u32,
}

#[tokio::main]
async fn main() -> Result<(), VectorStoreError> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Initialize SurrealDB client
    let surreal_uri = env::var("SURREAL_URI").expect("SURREAL_URI not set");
    let client = SurrealClient::new(surreal_uri)?;

    // Initialize OpenAI embedding model
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = Client::new(&openai_api_key).embedding_model(TEXT_EMBEDDING_ADA_002);

    // Sample articles
    let articles = vec![
        (
            "article1",
            "Advancements in AI",
            "Content about AI innovations.",
            "Technology",
            1500,
        ),
        (
            "article2",
            "Healthy Eating",
            "Content about nutrition and health.",
            "Health",
            800,
        ),
        (
            "article3",
            "Rust Programming",
            "Content about Rust features.",
            "Technology",
            2200,
        ),
    ];

    // Add articles to the database
    for (id, title, content, category, views) in articles {
        // Combine title and content for embedding
        let text = format!("{}: {}", title, content);

        // Generate embedding
        let embedding = model.embed_text(&text).await?.vec;

        // Create metadata as JSON string
        let metadata = serde_json::json!({
            "title": title,
            "content": content,
            "category": category,
            "views": views
        })
        .to_string();

        // Add embedding and metadata to the database
        client
            .add_embedding(id, embedding, &metadata, "articles")
            .await?;
    }

    // Create a vector index
    let index_config = IndexConfig::new(
        "article_index".to_string(),
        "embedding".to_string(),
        VectorSimilarityFunction::Cosine,
        1536, // Dimensions
        10,   // M
        30,   // Ef_construction
        100,  // Ef
    );
    client.create_vector_index(index_config, "articles").await?;

    // Define pre-filter and post-filter
    let pre_filter = FilterBuilder::new();

    let post_filter = FilterBuilder::new();

    let search_params = SearchParams {
        pre_filter: Some(pre_filter),
        post_filter: Some(post_filter),
        params: None,
        search_type: SearchType::Approximate,
        limit: 10,
    };

    // Get the vector index
    let index = client
        .get_index(model, "article_index", search_params.clone())
        .await?;

    // Perform the search
    let results = index
        .top_n::<Article>("Latest AI advancements", 10, "articles", search_params)
        .await?;

    // Handle the results
    for (score, id, article) in results {
        println!(
            "ID: {}, Score: {:.4}, Title: {}, Views: {}",
            id, score, article.title, article.views
        );
    }

    Ok(())
}
