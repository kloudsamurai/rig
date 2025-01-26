use rig_surrealdb::error::VectorStoreError;

#[test]
fn test_error_display() {
    let err = VectorStoreError::AuthError;
    assert_eq!(err.to_string(), "Authentication failed");
    
    let err = VectorStoreError::ConnectionError;
    assert_eq!(err.to_string(), "Connection error");
    
    let err = VectorStoreError::TimeoutError;
    assert_eq!(err.to_string(), "Timeout error");
}

#[test]
fn test_error_conversion() {
    let io_err = std::io::Error::new(std::io::ErrorKind::Other, "test");
    let err = VectorStoreError::DatastoreError(Box::new(io_err));
    assert_eq!(err.to_string(), "Database error: test");
}
