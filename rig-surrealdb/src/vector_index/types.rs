use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub ef_construction: usize,
    pub max_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfConfig {
    pub ncentroids: usize,
    pub niter: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatConfig {
    pub dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub bits: usize,
    pub quantizer_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    HNSW(HnswConfig),
    IVF(IvfConfig),
    Flat(FlatConfig),
    BruteForce,
}

// Re-export all types
pub use super::config::IndexConfig;
pub use super::SearchParams;