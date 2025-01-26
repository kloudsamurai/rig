use rig::vector_store::VectorStoreIndex;
use rig::{
    embeddings::{Embedding, EmbeddingsBuilder},
    providers::openai,
    Embed, OneOrMany,
};
use rig_surrealdb::{vector_index::SearchParams, SurrealClient};

#[derive(Embed, Clone, serde::Deserialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn test_advanced_filtering() {
    let client = setup_client().await;
    let embedding_model = get_test_embedding_model();

    // Add test data
    client
        .add_embedding(
            "id1",
            vec![0.1, 0.2, 0.3],
            r#"{"category": "A", "value": 10}"#,
            "test_table",
        )
        .await
        .unwrap();

    client
        .add_embedding(
            "id2",
            vec![0.4, 0.5, 0.6],
            r#"{"category": "B", "value": 20}"#,
            "test_table",
        )
        .await
        .unwrap();

    client
        .add_embedding(
            "id3",
            vec![0.7, 0.8, 0.9],
            r#"{"category": "A", "value": 30}"#,
            "test_table",
        )
        .await
        .unwrap();

    // Create vector index
    let index_config = IndexConfig::new("test_index")
        .embedding_property("embedding")
        .similarity_function(VectorSimilarityFunction::Cosine);
    client
        .create_vector_index(index_config, "test_table")
        .await
        .unwrap();

    // Get vector index
    let index = client
        .get_index(embedding_model, "test_index", SearchParams::default())
        .await
        .unwrap();

    // Define pre_filter and post_filter
    let pre_filter = FilterBuilder::new()
        .field("value")
        .operator("<")
        .value("25")
        .build()
        .unwrap();
    let post_filter = FilterBuilder::new()
        .field("category")
        .operator("=")
        .value("'A'")
        .build()
        .unwrap();

    let search_params = SearchParams {
        pre_filter: Some(pre_filter),
        post_filter: Some(post_filter),
        params: None,
    };

    // Perform search
    let results = index
        .top_n::<serde_json::Value>("test query", 10, "test_table", search_params)
        .await
        .unwrap();

    // Verify results
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, "id1");
}

#[tokio::test]
async fn test_top_n_ids_with_filtering() {
    let client = setup_client().await;
    let embedding_model = get_test_embedding_model();

    // Assume test data is already added from previous tests

    // Get vector index
    let index = client
        .get_index(embedding_model, "test_index", SearchParams::default())
        .await
        .unwrap();

    // Define pre_filter
    let pre_filter = FilterBuilder::new()
        .field("category")
        .operator("=")
        .value("'B'")
        .build()
        .unwrap();

    let search_params = SearchParams {
        pre_filter: Some(pre_filter),
        post_filter: None,
        params: None,
    };

    // Perform search
    let results = index
        .top_n_ids("test query", 10, "test_table", search_params)
        .await
        .unwrap();

    // Verify results
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, "id2");
}

#[tokio::test]
async fn test_invalid_filters() {
    let client = setup_client().await;
    let embedding_model = get_test_embedding_model();

    // Get vector index
    let index = client
        .get_index(embedding_model, "test_index", SearchParams::default())
        .await
        .unwrap();

    // Define invalid pre_filter (e.g., syntax error)
    let pre_filter = FilterBuilder::new()
        .field("invalid_field")
        .operator("=")
        .value("'value")
        .build();

    let search_params = SearchParams {
        pre_filter,
        post_filter: None,
        params: None,
    };

    // Perform search and expect an error
    let result = index
        .top_n::<serde_json::Value>("test query", 10, "test_table", search_params)
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_no_filter() {
    let client = setup_client().await;
    let embedding_model = get_test_embedding_model();

    // Get vector index
    let index = client
        .get_index(embedding_model, "test_index", SearchParams::default())
        .await
        .unwrap();

    // Use default `SearchParams`
    let search_params = SearchParams::default();

    // Perform search
    let results = index
        .top_n::<serde_json::Value>("test query", 10, "test_table", search_params)
        .await
        .unwrap();

    // Verify results (all entries are considered)
    assert_eq!(results.len(), 3);
}

fn get_test_embedding_model() -> TestEmbeddingModel {
    TestEmbeddingModel {}
}

struct TestEmbeddingModel;

#[async_trait::async_trait]
impl EmbeddingModel for TestEmbeddingModel {
    async fn embed_text(&self, _text: &str) -> Result<Embedding, VectorStoreError> {
        Ok(Embedding {
            vec: vec![0.1, 0.2, 0.3],
        })
    }
}

#[tokio::test]
async fn vector_search_test() {
    // Setup a local SurrealDB container for testing
    let client = SurrealClient::new("rocksdb://test.db", "test", "test", "root", "root")
        .await
        .expect("Failed to create SurrealDB client");

    // Initialize OpenAI client
    let openai_client = openai::Client::from_env();

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let embeddings = create_embeddings(model.clone()).await;

    // Add documents to SurrealDB
    for (doc, embeddings) in embeddings {
        client
            .add_embedding(
                &doc.id,
                embeddings.first().vec.clone(),
                &doc.definition,
                "DocumentEmbeddings",
            )
            .await?;
    }

    // Create vector index configuration
    let index_config = IndexConfig::new("vector_index")
        .embedding_property("embedding")
        .similarity_function(VectorSimilarityFunction::Cosine);

    // Create the vector index
    println!("Creating vector index...");
    client
        .create_vector_index(index_config.clone(), "DocumentEmbeddings")
        .await?;

    // Create a vector index instance
    let index = client
        .get_index(model, "vector_index", SearchParams::default())
        .await
        .unwrap();

    // Query the index
    let results = index
        .top_n::<serde_json::Value>("What is a glarb?", 1, "DocumentEmbeddings")
        .await
        .unwrap();

    let (_, _, value) = &results.first().unwrap();

    assert_eq!(
        value,
        &serde_json::json!({
            "id": "doc1",
            "document": "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
            "embedding": serde_json::Value::Null
        })
    )
}

#[tokio::test]
async fn test_delete() {
    let client = setup_client().await;

    // Add test data
    client
        .add_embedding("test1", vec![0.1, 0.2], "test data", "test_table")
        .await
        .unwrap();

    // Verify exists
    let exists = client
        .db
        .query("SELECT * FROM type::thing($table, $id)")
        .bind(("table", Value::from("test_table")))
        .bind(("id", Value::from("test1")))
        .await
        .unwrap()
        .take::<Option<Value>>(0)
        .unwrap()
        .is_some();
    assert!(exists);

    // Delete
    client
        .delete_embedding("test1", "test_table")
        .await
        .unwrap();

    // Verify deleted
    let exists = client
        .db
        .query("SELECT * FROM type::thing($table, $id)")
        .bind(("table", Value::from("test_table")))
        .bind(("id", Value::from("test1")))
        .await
        .unwrap()
        .take::<Option<Value>>(0)
        .unwrap()
        .is_some();
    assert!(!exists);

    // Test deleting non-existent record
    let result = client.delete_embedding("non_existent", "test_table").await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        VectorStoreError::MissingIdError(_)
    ));
}

#[tokio::test]
async fn test_batch_delete() {
    let client = setup_client().await;

    // Add test data
    let ids = vec!["test1", "test2", "test3"];
    for id in &ids {
        client
            .add_embedding(id, vec![0.1, 0.2], "test data", "test_table")
            .await
            .unwrap();
    }

    // Delete batch
    client
        .delete_batch(ids.clone(), "test_table")
        .await
        .unwrap();

    // Verify deleted
    for id in ids {
        let exists = client
            .db
            .query("SELECT * FROM type::thing($table, $id)")
            .bind(("table", Value::from("test_table")))
            .bind(("id", Value::from(id)))
            .await
            .unwrap()
            .take::<Option<Value>>(0)
            .unwrap()
            .is_some();
        assert!(!exists);
    }

    // Test deleting batch with non-existent record
    let ids = vec!["test4", "non_existent"];
    let result = client.delete_batch(ids, "test_table").await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        VectorStoreError::MissingIdError(_)
    ));
}

#[tokio::test]
async fn test_update() {
    let client = setup_client().await;

    // Add test data
    client
        .add_embedding("test1", vec![0.1, 0.2], "test data", "test_table")
        .await
        .unwrap();

    // Test updating with empty ID
    let result = client
        .update_embedding("", "test_table", "data", None)
        .await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        VectorStoreError::InvalidDataError(_)
    ));

    // Test updating with empty table name
    let result = client.update_embedding("test1", "", "data", None).await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        VectorStoreError::InvalidDataError(_)
    ));

    // Test updating with invalid embedding dimensions
    let result = client
        .update_embedding("test1", "test_table", "data", Some(vec![]))
        .await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        VectorStoreError::InvalidDataError(_)
    ));

    // Test updating with very large embedding
    let large_embedding = vec![0.0; 100000];
    let result = client
        .update_embedding("test1", "test_table", "data", Some(large_embedding))
        .await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        VectorStoreError::InvalidDataError(_)
    ));

    // Test updating with special characters in metadata
    let special_metadata = "test data with special chars: !@#$%^&*()";
    client
        .update_embedding("test1", "test_table", special_metadata, None)
        .await
        .unwrap();

    // Verify the update
    let record = client
        .db
        .query("SELECT * FROM type::thing($table, $id)")
        .bind(("table", Value::from("test_table")))
        .bind(("id", Value::from("test1")))
        .await
        .unwrap()
        .take::<Option<Value>>(0)
        .unwrap();

    assert!(record.is_some());
    let record = record.unwrap();
    assert_eq!(
        record.get("metadata").and_then(Value::as_string),
        Some(special_metadata.to_string())
    );

    // Test updating non-existent record
    let result = client
        .update_embedding("non_existent", "test_table", "data", None)
        .await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        VectorStoreError::MissingIdError(_)
    ));

    // Test updating with empty metadata
    let result = client
        .update_embedding("test1", "test_table", "", None)
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_batch_update() {
    let client = setup_client().await;

    // Add test data
    let ids = vec!["test1", "test2", "test3"];
    for id in &ids {
        client
            .add_embedding(id, vec![0.1, 0.2], "test data", "test_table")
            .await
            .unwrap();
    }

    // Update the batch
    let updates = ids
        .iter()
        .map(|id| {
            (
                id.to_string(),
                "updated data".to_string(),
                Some(vec![0.3, 0.4]),
            )
        })
        .collect::<Vec<_>>();

    client.update_batch(updates, "test_table").await.unwrap();

    // Verify the updates
    for id in ids {
        let record = client
            .db
            .query("SELECT * FROM type::thing($table, $id)")
            .bind(("table", Value::from("test_table")))
            .bind(("id", Value::from(id)))
            .await
            .unwrap()
            .take::<Option<Value>>(0)
            .unwrap();

        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(
            record.get("metadata").and_then(Value::as_string),
            Some("updated data".to_string())
        );
        assert_eq!(
            record.get("embedding").and_then(Value::as_array),
            Some(vec![Value::from(0.3), Value::from(0.4)])
        );
    }
}

#[tokio::test]
async fn test_hybrid_search() {
    let client = setup_client().await;

    // Add test data
    client
        .add_embedding(
            "test1",
            vec![0.1, 0.2],
            "test data about cats",
            "test_table",
        )
        .await
        .unwrap();
    client
        .add_embedding(
            "test2",
            vec![0.3, 0.4],
            "test data about dogs",
            "test_table",
        )
        .await
        .unwrap();

    // Test basic hybrid search
    let results = client
        .hybrid_search::<serde_json::Value>("cats", 1, "test_table", SearchParams::default())
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, "test1");

    // Test with empty query
    let result = client
        .hybrid_search::<serde_json::Value>("", 1, "test_table", SearchParams::default())
        .await;
    assert!(result.is_err());

    // Test with invalid table name
    let result = client
        .hybrid_search::<serde_json::Value>("cats", 1, "", SearchParams::default())
        .await;
    assert!(result.is_err());

    // Test with limit 0
    let results = client
        .hybrid_search::<serde_json::Value>("cats", 0, "test_table", SearchParams::default())
        .await
        .unwrap();
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_rerank() {
    let client = setup_client().await;

    // Add test data
    client
        .add_embedding(
            "test1",
            vec![0.1, 0.2],
            "test data about cats",
            "test_table",
        )
        .await
        .unwrap();
    client
        .add_embedding(
            "test2",
            vec![0.3, 0.4],
            "test data about dogs",
            "test_table",
        )
        .await
        .unwrap();

    // Perform initial search
    let results = client
        .top_n::<serde_json::Value>("cats", 2, "test_table", SearchParams::default())
        .await
        .unwrap();

    // Rerank results
    let reranked_results = client
        .rerank(results, |record| {
            if record
                .get("metadata")
                .and_then(Value::as_string)
                .unwrap_or_default()
                .contains("cats")
            {
                1.0
            } else {
                0.5
            }
        })
        .await
        .unwrap();

    assert_eq!(reranked_results.len(), 2);
    assert_eq!(reranked_results[0].1, "test1");
}

#[tokio::test]
async fn test_hybrid_search_edge_cases() {
    let client = setup_client().await;

    // Test empty table
    let results = client
        .hybrid_search::<serde_json::Value>("cats", 1, "empty_table", SearchParams::default())
        .await;
    assert!(matches!(results, Err(VectorStoreError::DatastoreError(_))));

    // Test with special characters
    client
        .add_embedding(
            "test1",
            vec![0.1, 0.2],
            "test data about cats & dogs",
            "test_table",
        )
        .await
        .unwrap();
    let results = client
        .hybrid_search::<serde_json::Value>("cats & dogs", 1, "test_table", SearchParams::default())
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, "test1");

    // Test with very long query
    let long_query = "a".repeat(1000);
    let results = client
        .hybrid_search::<serde_json::Value>(&long_query, 1, "test_table", SearchParams::default())
        .await;
    assert!(results.is_ok());
}

#[tokio::test]
async fn test_graph_query() {
    let client = setup_client().await;

    // Add test data
    client
        .db
        .query(
            "
            CREATE person:john SET name = 'John Doe';
            CREATE person:jane SET name = 'Jane Doe';
            RELATE person:john -> knows -> person:jane;
        ",
        )
        .await
        .unwrap();

    // Perform graph query
    let results = client
        .graph_query::<serde_json::Value>(
            "
            SELECT ->knows->person AS friends
            FROM person:john
        ",
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["friends"][0]["name"], "Jane Doe");
}

#[tokio::test]
async fn test_full_text_search() {
    let client = setup_client().await;

    // Add test data
    client
        .db
        .query(
            "
            CREATE document:1 SET content = 'This is a test document about cats.';
            CREATE document:2 SET content = 'This is another test document about dogs.';
        ",
        )
        .await
        .unwrap();

    // Perform full-text search
    let results = client
        .full_text_search::<serde_json::Value>("cats", "document", "content")
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["content"], "This is a test document about cats.");
}

#[tokio::test]
async fn test_openai_embedding() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = EmbeddingModel::openai(&api_key, "text-embedding-ada-002");

    let embedding = model.embed_text("test").await.unwrap();
    assert!(!embedding.is_empty());
}

#[tokio::test]
async fn test_huggingface_embedding() {
    let api_key = std::env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY not set");
    let model = EmbeddingModel::huggingface(&api_key, "sentence-transformers/all-MiniLM-L6-v2");

    let embedding = model.embed_text("test").await.unwrap();
    assert!(!embedding.is_empty());
}

#[tokio::test]
async fn test_delete_transaction_rollback() {
    let client = setup_client().await;

    // Add test data
    let ids = vec!["test1", "test2", "test3"];
    for id in &ids {
        client
            .add_embedding(id, vec![0.1, 0.2], "test data", "test_table")
            .await
            .unwrap();
    }

    // Try to delete batch with one invalid ID
    let ids = vec!["test1", "non_existent"];
    let result = client.delete_batch(ids, "test_table").await;
    assert!(result.is_err());

    // Verify no records were deleted
    for id in &["test1", "test2", "test3"] {
        let exists = client
            .db
            .query("SELECT * FROM type::thing($table, $id)")
            .bind(("table", Value::from("test_table")))
            .bind(("id", Value::from(id)))
            .await
            .unwrap()
            .take::<Option<Value>>(0)
            .unwrap()
            .is_some();
        assert!(exists);
    }
}

async fn setup_client() -> SurrealClient {
    let uri = "ws://localhost:8000";
    let ns = "test";
    let db = "test";
    let user = "root";
    let pass = "root";

    SurrealClient::new(uri, ns, db, user, pass)
        .await
        .expect("Failed to create SurrealDB client")
}

async fn create_embeddings(model: openai::EmbeddingModel) -> Vec<(Word, OneOrMany<Embedding>)> {
    let words = vec![
        Word {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Word {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Word {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }
    ];

    EmbeddingsBuilder::new(model)
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap()
}
