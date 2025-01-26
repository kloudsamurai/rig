//! Simple end-to-end example of the vector search capabilities of the `rig-surrealdb` crate.
//! This example expects a running SurrealDB instance.
//! It:
//! 1. Generates embeddings for a set of 3 "documents"
//! 2. Adds the documents to the SurrealDB
//! 3. Creates a vector index on the embeddings
//! 4. Queries the vector index
//! 5. Returns the results
use std::env;

use futures::StreamExt;
use metrics::gauge;
use rig::{
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex as _,
    Embed,
};
use rig_surrealdb::{
    vector_index::{SearchParams, SearchType},
    FilterBuilder, HnswConfig, IndexConfig, SurrealClient, VectorSimilarityFunction,
};

#[derive(Embed, Clone, Debug)]
pub struct Word {
    pub id: String,
    #[embed]
    pub definition: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    // Initialize SurrealDB client
    let surreal_uri = env::var("SURREAL_URI").expect("SURREAL_URI not set");
    let surreal_ns = env::var("SURREAL_NS").expect("SURREAL_NS not set");
    let surreal_db = env::var("SURREAL_DB").expect("SURREAL_DB not set");
    let surreal_user = env::var("SURREAL_USER").expect("SURREAL_USER not set");
    let surreal_pass = env::var("SURREAL_PASS").expect("SURREAL_PASS not set");

    let surreal_client = SurrealClient::new(surrealdb::api::engine::remote::ws::Client::new(
        &surreal_uri,
    ));

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .document(Word {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        })?
        .document(Word {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        })?
        .document(Word {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        })?
        .build()
        .await?;

    // Add documents to SurrealDB
    for (doc, embeddings) in embeddings {
        surreal_client
            .add_embedding(
                &doc.id,
                embeddings.first().vec.clone(),
                &doc.definition,
                "DocumentEmbeddings",
            )
            .await?;
    }

    // Create vector index configuration with HNSW parameters optimized for SurrealDB 2.1
    let index_config = IndexConfig::new(
        "vector_index".to_string(),
        "embedding".to_string(),
        VectorSimilarityFunction::Cosine,
        1536, // TEXT_EMBEDDING_ADA_002 dimensions
        100,  // batch size
        3,    // max retries
        1000, // retry delay in ms
    );

    // Create the vector index
    println!("Creating vector index...");
    surreal_client
        .create_vector_index(index_config.clone(), "DocumentEmbeddings")
        .await?;

    // Create a vector index instance
    let index = surreal_client
        .get_index(model, "vector_index", SearchParams::default())
        .await?;

    // The struct that will reprensent a node in the database. Used to deserialize the results of the query (passed to the `top_n` methods)
    // ❗IMPORTANT: The field names must match the property names in the database
    #[derive(serde::Deserialize)]
    struct Document {
        #[allow(dead_code)]
        id: String,
        document: String,
    }

    let search_params = SearchParams {
        pre_filter: Some(
            FilterBuilder::new()
                .field("category")
                .operator("=")
                .value("'Technology'")
                .build()
                .unwrap(),
        ),
        post_filter: None,
        params: None,
        search_type: SearchType::Exact,
        limit: 5,
    };

    match index
        .top_n::<Document>(
            "Latest advancements",
            5,
            "DocumentEmbeddings",
            search_params,
        )
        .await
    {
        Ok(results) => {
            for (score, id, doc) in results {
                println!("ID: {}, Score: {:.4}, Title: {}", id, score, doc.document);
            }
        }
        Err(e) => {
            eprintln!("Search failed: {}", e);
        }
    }

    let id_results = index
        .top_n_ids("What is a linglingdong?", 1, "DocumentEmbeddings")
        .await?
        .into_iter()
        .map(|(score, id)| (score, id))
        .collect::<Vec<_>>();

    println!("ID results: {:?}", id_results);

    gauge!("update_embedding.active_requests", -1);
    Ok(())
}
