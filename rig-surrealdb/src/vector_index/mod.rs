use rig::{
    embeddings::EmbeddingModel,
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use serde::{de::DeserializeOwned, Deserialize};
use std::sync::Arc;
use surrealdb::{engine::remote::ws::Client, Surreal};

/// Vector store implementation for SurrealDB that enables similarity search
pub struct SurrealVectorStore<M: EmbeddingModel> {
    /// SurrealDB client instance
    client: Arc<Surreal<Client>>,
    /// Embedding model used for generating vectors
    model: M,
    /// Name of collection storing vectors
    collection: String,
    /// Name of property containing embeddings
    embedding_property: String,
}

impl<M: EmbeddingModel> SurrealVectorStore<M> {
    /// Create a new SurrealDB vector store
    pub fn new(
        client: Arc<Surreal<Client>>,
        model: M,
        collection: String,
        embedding_property: String,
    ) -> Self {
        Self {
            client,
            model,
            collection,
            embedding_property,
        }
    }

    /// Builds a SurrealQL query for vector similarity search
    fn build_search_query(&self, n: usize) -> String {
        format!(
            r#"SELECT *, vector::similarity({}, $query_vector) as score
               FROM {}
               ORDER BY score DESC
               LIMIT {}"#,
            self.embedding_property, self.collection, n
        )
    }
}

#[derive(Debug, Deserialize)]
struct SearchResult<T> {
    id: String,
    score: f64,
    #[serde(flatten)]
    payload: T,
}

impl<M: EmbeddingModel + Send + Sync> VectorStoreIndex for SurrealVectorStore<M> {
    async fn top_n<T>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: DeserializeOwned + Send,
    {
        let embedding = self.model.embed_text(query).await?;
        let query = self.build_search_query(n);

        let mut results = self
            .client
            .query(&query)
            .bind(("query_vector", embedding.vec))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<SearchResult<T>> = results
            .take(0)
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(rows
            .into_iter()
            .map(|row| (row.score, row.id, row.payload))
            .collect())
    }

    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        let query = self.build_search_query(n);

        let mut results = self
            .client
            .query(&query)
            .bind(("query_vector", embedding.vec))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<SearchResult<()>> = results
            .take(0)
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(rows.into_iter().map(|row| (row.score, row.id)).collect())
    }
}
