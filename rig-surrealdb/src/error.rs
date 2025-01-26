use thiserror::Error;

#[derive(Error, Debug)]
pub enum VectorStoreError {
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Invalid embedding property: {0}")]
    InvalidEmbeddingProperty(String),

    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(usize),

    #[error("Invalid max elements: {0}")]
    InvalidMaxElements(usize),

    #[error("Database error: {0}")]
    DatabaseError(#[from] surrealdb::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}
