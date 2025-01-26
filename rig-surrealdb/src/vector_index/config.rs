use rig::vector_store::VectorStoreError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Configuration for a vector index
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexConfig {
    /// Name of the index
    pub index_name: String,
    /// Property containing the embeddings
    pub embedding_property: String,
    /// Similarity function to use
    pub similarity_function: VectorSimilarityFunction,
    /// Type of index to create
    pub index_type: IndexType,
    /// Dimensionality of the embeddings
    pub dimensions: usize,
    /// Maximum number of elements in the index
    pub max_elements: usize,
    /// Advanced configuration options
    pub advanced_config: AdvancedIndexConfig,
    /// Batch size for operations
    pub batch_size: usize,
    /// Maximum number of retries for operations
    pub max_retries: usize,
    /// Delay between retries in milliseconds
    pub retry_delay: u64,
}

/// Available similarity functions
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum VectorSimilarityFunction {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
    Jaccard,
    Hamming,
}

/// Type of index to create
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum IndexType {
    HNSW(HnswConfig),
    IVF(IvfConfig),
    Flat(FlatConfig),
    BruteForce,
}

/// Advanced configuration options
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AdvancedIndexConfig {
    /// HNSW-specific configuration
    pub hnsw: Option<HnswConfig>,
    /// IVF-specific configuration
    pub ivf: Option<IvfConfig>,
    /// Flat index configuration
    pub flat: Option<FlatConfig>,
    /// Quantization configuration
    pub quantization: Option<QuantizationConfig>,
    /// Number of threads to use
    pub num_threads: Option<usize>,
    /// Allow replacing deleted elements
    pub allow_replace_deleted: Option<bool>,
    /// Maximum number of connections
    pub max_connections: Option<usize>,
    /// Minimum number of connections
    pub min_connections: Option<usize>,
}

impl IndexConfig {
    /// Creates a new IndexConfig with default values
    ///
    /// # Arguments
    /// * `index_name` - Name of the index
    /// * `embedding_property` - Property containing the embeddings
    /// * `similarity_function` - Similarity function to use
    /// * `index_type` - Type of index to create
    /// * `dimensions` - Dimensionality of the embeddings
    /// * `max_elements` - Maximum number of elements in the index
    ///
    /// # Returns
    /// A new IndexConfig instance
    pub fn new(
        index_name: String,
        embedding_property: String,
        similarity_function: VectorSimilarityFunction,
        index_type: IndexType,
        dimensions: usize,
        max_elements: usize,
    ) -> Self {
        Self {
            index_name,
            embedding_property,
            similarity_function,
            index_type,
            dimensions,
            max_elements,
            batch_size: 100,
            max_retries: 3,
            retry_delay: 100,
            advanced_config: AdvancedIndexConfig {
                hnsw: None,
                ivf: None,
                flat: None,
                quantization: None,
                num_threads: None,
                allow_replace_deleted: None,
                max_connections: None,
                min_connections: None,
            },
        }
    }

    /// Validates the configuration parameters
    ///
    /// # Returns
    /// Result<(), VectorStoreError> - Ok if valid, Err with specific error otherwise
    ///
    /// # Errors
    /// Returns VectorStoreError if any configuration parameter is invalid
    pub fn validate(&self) -> Result<(), VectorStoreError> {
        if self.index_name.is_empty() {
            return Err(VectorStoreError::InvalidConfigurationError(
                "Index name cannot be empty".to_string(),
            ));
        }

        if self.embedding_property.is_empty() {
            return Err(VectorStoreError::InvalidEmbeddingProperty(
                self.embedding_property.clone(),
            ));
        }

        if self.dimensions == 0 {
            return Err(VectorStoreError::InvalidConfigurationError(
                "Dimensions must be greater than 0".to_string(),
            ));
        }

        if self.max_elements == 0 {
            return Err(VectorStoreError::InvalidMaxElements(self.max_elements));
        }

        if self.batch_size == 0 || self.batch_size > 1000 {
            return Err(VectorStoreError::InvalidBatchSize(self.batch_size));
        }

        if self.max_retries > 10 {
            return Err(VectorStoreError::InvalidConfigurationError(
                "Max retries cannot exceed 10".to_string(),
            ));
        }

        if let Some(conns) = self.advanced_config.max_connections {
            if conns == 0 || conns > 100 {
                return Err(VectorStoreError::InvalidConfigurationError(
                    "Max connections must be between 1 and 100".to_string(),
                ));
            }
        }

        if let Some(threads) = self.advanced_config.num_threads {
            if threads == 0 || threads > 64 {
                return Err(VectorStoreError::InvalidThreadCount(threads));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = IndexConfig::new(
            "test_index".to_string(),
            "embedding".to_string(),
            VectorSimilarityFunction::Cosine,
            IndexType::BruteForce,
            768,
            1000,
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_index_name() {
        let config = IndexConfig::new(
            "".to_string(),
            "embedding".to_string(),
            VectorSimilarityFunction::Cosine,
            IndexType::BruteForce,
            768,
            1000,
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_embedding_property() {
        let config = IndexConfig::new(
            "test_index".to_string(),
            "".to_string(),
            VectorSimilarityFunction::Cosine,
            IndexType::BruteForce,
            768,
            1000,
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_dimensions() {
        let config = IndexConfig::new(
            "test_index".to_string(),
            "embedding".to_string(),
            VectorSimilarityFunction::Cosine,
            IndexType::BruteForce,
            0,
            1000,
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_max_elements() {
        let config = IndexConfig::new(
            "test_index".to_string(),
            "embedding".to_string(),
            VectorSimilarityFunction::Cosine,
            IndexType::BruteForce,
            768,
            0,
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_batch_size() {
        let mut config = IndexConfig::new(
            "test_index".to_string(),
            "embedding".to_string(),
            VectorSimilarityFunction::Cosine,
            IndexType::BruteForce,
            768,
            1000,
        );
        config.batch_size = 0;
        assert!(config.validate().is_err());
    }
}
