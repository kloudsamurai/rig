use rig::{providers::openai::Client, vector_store::VectorStoreIndex};
use rig_surrealdb::{FilterBuilder, SearchParams, SearchType, SurrealClient};
use serde::Deserialize;
use std::env;

#[derive(Debug, Deserialize)]
struct TestRecord {
    id: String,
    text: String,
    category: String,
}

#[tokio::test]
async fn test_pre_filter() {
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);
    let model = openai_client.embedding_model("text-embedding-ada-002");

    const SURREAL_URI: &str = "rocksdb://test.db";
    const SURREAL_NS: &str = "test";
    const SURREAL_DB: &str = "test";
    const SURREAL_USER: &str = "root";
    const SURREAL_PASS: &str = "root";

    let client = SurrealClient::new(SURREAL_URI).unwrap();

    // Create test data
    let test_data = vec![
        ("1", "cat in the hat", "children"),
        ("2", "war and peace", "literature"),
        ("3", "green eggs and ham", "children"),
    ];

    for (id, text, category) in test_data {
        client
            .add_embedding(
                id,
                vec![0.0; 1536],
                &format!("{{\"text\":\"{}\",\"category\":\"{}\"}}", text, category),
                "test_data",
            )
            .await
            .unwrap();
    }

    let index = client
        .get_index(model, "test_index", SearchParams::default())
        .await
        .unwrap();

    // Test pre-filter by category
    let filter = FilterBuilder::new()
        .with_field("category")
        .with_operator("=")
        .with_value("'children'")
        .build()
        .unwrap();

    let search_params = SearchParams {
        pre_filter: Some(filter),
        post_filter: None,
        params: None,
        limit: 3,
        search_type: SearchType::Similarity,
    };

    let results = index
        .top_n::<TestRecord>("children's book", search_params, "test_data")
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|(_, _, r)| r.category == "children"));
}

#[tokio::test]
async fn test_post_filter() {
    // Similar setup as above...
    // Test post-filter functionality
}

#[tokio::test]
async fn test_combined_filters() {
    // Similar setup as above...
    // Test combination of pre and post filters
}
