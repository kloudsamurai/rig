//! This example demonstrates how to perform a vector search on a Neo4j database.
//! It is based on the [Neo4j Embeddings & Vector Index Tutorial](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/).
//! The tutorial uses the `recommendations` dataset and the `moviePlots` index, which is created in the tutorial.
//! They both need to be configured and the database running before running this example.
//!
//! Neo4j provides a demo database for the `recommendations` dataset (see [Github Neo4j-Graph-Examples/recommendations](https://github.com/neo4j-graph-examples/recommendations/tree/main?tab=readme-ov-file#setup)).
//!
//!     const NEO4J_URI: &str = "neo4j+s://demo.neo4jlabs.com:7687";
//!     const NEO4J_DB: &str = "recommendations";
//!     const NEO4J_USERNAME: &str = "recommendations";
//!     const NEO4J_PASSWORD: &str = "recommendations";
//!
//! [examples/vector_search_simple.rs](examples/vector_search_simple.rs) provides an example starting from an empty database.
//! [examples/vector_search_movies_add_embeddings.rs](examples/vector_search_movies_add_embeddings.rs) provides an example of
//! how to add embeddings to an existing `recommendations` database.
use neo4rs::ConfigBuilder;
use rig_neo4j::{
    vector_index::{IndexConfig, SearchParams},
    Neo4jClient,
};

use std::env;

use rig::{
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let neo4j_uri = env::var("NEO4J_URI").expect("NEO4J_URI not set");
    let neo4j_username = env::var("NEO4J_USERNAME").expect("NEO4J_USERNAME not set");
    let neo4j_password = env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD not set");

    let neo4j_client = Neo4jClient::from_config(
        ConfigBuilder::default()
            .uri(neo4j_uri)
            .user(neo4j_username)
            .password(neo4j_password)
            .db("neo4j")
            .build()
            .unwrap(),
    )
    .await?;

    // // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Define the properties that will be retrieved from querying the graph nodes
    #[derive(Debug, Deserialize, Serialize)]
    struct Movie {
        title: String,
        plot: String,
    }

    // Create a vector index on our vector store
    // ❗IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = neo4j_client.index(
        model,
        IndexConfig::default().index_name("moviePlots"),
        SearchParams::new(Some("node.year > 1990".to_string())),
    );

    index
        .create_and_await_vector_index("Movie".to_string(), None)
        .await?;

    // Query the index
    let results = index
        .top_n::<Movie>("a historical movie on quebec", 5)
        .await?
        .into_iter()
        .map(
            |(score, id, doc)| rig_neo4j::vector_index::display::SearchResult {
                title: doc.title,
                id,
                description: doc.plot,
                score,
            },
        )
        .collect::<Vec<_>>();

    println!(
        "{:#}",
        rig_neo4j::vector_index::display::SearchResults(&results)
    );

    let id_results = index
        .top_n_ids("A movie where the bad guy wins", 1)
        .await?
        .into_iter()
        .map(|(score, id)| (score, id))
        .collect::<Vec<_>>();

    println!("ID results: {:?}", id_results);

    Ok(())
}
