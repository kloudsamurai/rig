//! This example shows how to perform a vector search on a SurrealDB database.
//! It uses a movie dataset to demonstrate vector search capabilities.
//!
//! ❗IMPORTANT: The `recommendations` database has 28k nodes, so this example will take a while to run.

use std::env;

use rig::{
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};

use rig_surrealdb::{
    vector_index::{IndexConfig, SearchParams, VectorSimilarityFunction}, 
    SurrealClient
};
use serde::{Deserialize, Serialize};

#[path = "./display/lib.rs"]
mod display;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Movie {
    title: String,
    plot: String,
    to_encode: Option<String>,
}

const NODE_LABEL: &str = "Movie";
const INDEX_NAME: &str = "moviePlots";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let surreal_uri = env::var("SURREAL_URI").expect("SURREAL_URI not set");
    let surreal_ns = env::var("SURREAL_NS").expect("SURREAL_NS not set");
    let surreal_db = env::var("SURREAL_DB").expect("SURREAL_DB not set");
    let surreal_user = env::var("SURREAL_USER").expect("SURREAL_USER not set");
    let surreal_pass = env::var("SURREAL_PASS").expect("SURREAL_PASS not set");

    let surreal_client = SurrealClient::new(&surreal_uri, &surreal_ns, &surreal_db, &surreal_user, &surreal_pass).await?;

    // Add embeddings to the SurrealDB database
    let batch_size = 1000;
    let mut batch_n = 1;
    let mut movies_batch = Vec::<Movie>::new();

    // Query movies from SurrealDB
    let movies: Vec<Movie> = surreal_client
        .db
        .query("SELECT title, plot FROM type::table($table)")
        .bind(("table", Value::from(NODE_LABEL)))
        .await?
        .take(0)?;

    for movie in movies {
        movies_batch.push(Movie {
            title: movie.title,
            plot: movie.plot,
            to_encode: Some(format!("Title: {}\nPlot: {}", movie.title, movie.plot)),
        });

        // Import a batch; flush buffer
        if movies_batch.len() == batch_size {
            import_batch(&surreal_client, &movies_batch, batch_n).await?;
            movies_batch.clear();
            batch_n += 1;
        }
    }

    // Import any remaining movies
    if !movies_batch.is_empty() {
        import_batch(&surreal_client, &movies_batch, batch_n).await?;
    }

    // Show counters
    let count: u64 = surreal_client
        .db
        .query("SELECT count() FROM type::table($table) WHERE embedding != NONE")
        .bind(("table", Value::from(NODE_LABEL)))
        .await?
        .take(0)?;

    println!(
        "Embeddings generated and attached to nodes.\n\
         Movie nodes with embeddings: {}.",
        count
    );

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Since we are starting from scratch, we need to create the DB vector index
    surreal_client
        .create_vector_index(
            IndexConfig::new(INDEX_NAME).similarity_function(vector_index::VectorSimilarityFunction::Cosine),
            NODE_LABEL,
            &model,
        )
        .await?;

    // ❗IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = surreal_client
        .get_index(
            model,
            INDEX_NAME,
            SearchParams::new(Some("node.year > 1990".to_string())),
        )
        .await?;

    // Query the index
    let results = index
        .top_n::<Movie>("a historical movie on quebec", 5, NODE_LABEL)
        .await?
        .into_iter()
        .map(|(score, id, doc)| display::SearchResult {
            title: doc.title,
            id,
            description: doc.plot,
            score,
        })
        .collect::<Vec<_>>();

    println!("{:#}", display::SearchResults(&results));

    let id_results = index
        .top_n_ids("What is a linglingdong?", 1, NODE_LABEL)
        .await?
        .into_iter()
        .map(|(score, id)| (score, id))
        .collect::<Vec<_>>();

    println!("ID results: {:?}", id_results);

    Ok(())
}

async fn import_batch(client: &SurrealClient, nodes: &[Movie], batch_n: i32) -> Result<(), anyhow::Error> {
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let to_encode_list: Vec<String> = nodes
        .iter()
        .map(|node| node.to_encode.clone().unwrap())
        .collect();

    for (i, node) in nodes.iter().enumerate() {
        let embedding = openai_client
            .embedding_model(TEXT_EMBEDDING_ADA_002)
            .embed_text(&to_encode_list[i])
            .await?;

        client.add_embedding(
            &format!("movie_{}_{}", batch_n, i),
            embedding.vec,
            &serde_json::to_string(node)?,
            NODE_LABEL
        ).await?;
    }

    println!("Processed batch {}", batch_n);
    Ok(())
}
