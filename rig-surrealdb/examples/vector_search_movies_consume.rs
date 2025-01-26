//! This example demonstrates how to perform a vector search on a SurrealDB database.
//! It uses a movie dataset to demonstrate vector search capabilities.
use rig_surrealdb::{
    vector_index::{SearchParams, SearchType},
    FilterBuilder, SurrealClient,
};
use std::iter::Filter;

use std::env;

use rig::{
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};
use serde::{Deserialize, Serialize};

#[path = "./display/lib.rs"]
mod display;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    const INDEX_NAME: &str = "moviePlotsEmbedding";

    // Initialize OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let surreal_uri = env::var("SURREAL_URI").expect("SURREAL_URI not set");
    let surreal_ns = env::var("SURREAL_NS").expect("SURREAL_NS not set");
    let surreal_db = env::var("SURREAL_DB").expect("SURREAL_DB not set");
    let surreal_user = env::var("SURREAL_USER").expect("SURREAL_USER not set");
    let surreal_pass = env::var("SURREAL_PASS").expect("SURREAL_PASS not set");

    let surreal_client = SurrealClient::from_connection(&surreal_uri)?;

    // // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Define the properties that will be retrieved from querying the graph nodes
    #[derive(Debug, Deserialize, Serialize)]
    struct Movie {
        title: String,
        plot: String,
    }

    // Get the vector index
    let index = surreal_client
        .get_index(
            model,
            INDEX_NAME,
            SearchParams {
                pre_filter: Some(
                    FilterBuilder::new()
                        .field("year")
                        .operator(">")
                        .value("1990")
                        .build()
                        .unwrap(),
                ),
                post_filter: None,
                params: None,
                limit: 10,
                search_type: SearchType::Exact,
            },
        )
        .await?;

    let search_params = SearchParams {
        pre_filter: Some(
            FilterBuilder::new()
                .field("genre")
                .operator("=")
                .value("'Action'")
                .build()
                .unwrap(),
        ),
        post_filter: Some(
            FilterBuilder::new()
                .field("rating")
                .operator(">=")
                .value("8.0")
                .build()
                .unwrap(),
        ),
        params: None,
        limit: 5,
        search_type: SearchType::Exact,
    };

    match index
        .top_n::<Movie>("hero defeats villain", 5, "Movie", search_params)
        .await
    {
        Ok(results) => {
            for result in results {
                println!(
                    "{:?}",
                    display::SearchResult {
                        title: result.2.title,
                        id: result.1,
                        description: result.2.plot,
                        score: result.0,
                    }
                );
            }
        }
        Err(e) => {
            eprintln!("Search failed: {}", e);
        }
    }

    let id_results = index
        .top_n_ids("A movie where the bad guy wins", 1, "Movie")
        .await?
        .into_iter()
        .map(|(score, id)| (score, id))
        .collect::<Vec<_>>();

    println!("ID results: {:?}", id_results);

    Ok(())
}
