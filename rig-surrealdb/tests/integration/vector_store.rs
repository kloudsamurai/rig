use rig_surrealdb::error::VectorStoreError;
use rig_surrealdb::vector_index::VectorIndex;
use surrealdb::engine::memory::Mem;
use surrealdb::Surreal;

#[tokio::test]
async fn test_vector_index_creation() -> Result<(), VectorStoreError> {
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns("test").use_db("test").await.unwrap();

    let index = VectorIndex::new(&db, "test_index").await?;
    assert_eq!(index.name(), "test_index");
    
    Ok(())
}

#[tokio::test]
async fn test_vector_index_error_handling() {
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns("test").use_db("test").await.unwrap();

    // Test invalid index name
    let result = VectorIndex::new(&db, "").await;
    assert!(matches!(result, Err(VectorStoreError::InvalidDataError(_))));
}
