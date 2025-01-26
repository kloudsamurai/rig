use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use surrealdb::engine::remote::ws::{Client, Ws};
use surrealdb::Surreal;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use crate::error::VectorStoreError;

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_size: usize,
    pub min_idle: usize,
    pub timeout: Duration,
    pub max_lifetime: Duration,
    pub idle_timeout: Duration,
}

impl PoolConfig {
    pub fn validate(&self) -> Result<(), VectorStoreError> {
        if self.max_size == 0 {
            return Err(VectorStoreError::InvalidConfigurationError(
                "Max pool size must be greater than 0".to_string(),
            ));
        }
        
        if self.min_idle > self.max_size {
            return Err(VectorStoreError::InvalidConfigurationError(
                "Minimum idle connections cannot exceed max pool size".to_string(),
            ));
        }
        
        if self.timeout.as_secs() == 0 {
            return Err(VectorStoreError::InvalidConfigurationError(
                "Timeout must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

pub struct ConnectionPool {
    connections: Arc<Mutex<Vec<Surreal<Client>>>>,
    semaphore: Arc<Semaphore>,
    _config: PoolConfig,
    _last_used: Arc<Mutex<HashMap<usize, Instant>>>,
}

impl ConnectionPool {
    pub fn new(config: PoolConfig) -> Result<Self, VectorStoreError> {
        config.validate()?;
        
        let semaphore = Arc::new(Semaphore::new(config.max_size));
        let connections = Arc::new(Mutex::new(Vec::with_capacity(config.max_size)));
        let last_used = Arc::new(Mutex::new(HashMap::new()));
        
        Ok(Self {
            connections,
            semaphore,
            _config: config,
            _last_used: last_used,
        })
    }

    pub async fn get(&self) -> Result<Surreal<Client>, VectorStoreError> {
        let _permit = self.semaphore.clone().acquire_owned().await.map_err(|e| {
            VectorStoreError::ConnectionError(format!("Failed to acquire connection: {}", e))
        })?;

        let mut connections = self.connections.lock().await;
        if let Some(conn) = connections.pop() {
            return Ok(conn);
        }

        // Create new connection if pool is empty
        let db = Surreal::new::<Ws>("ws://localhost:8000").await.map_err(|e| {
            VectorStoreError::ConnectionError(format!("Failed to create new connection: {}", e))
        })?;

        Ok(db)
    }

    pub async fn put(&self, client: Surreal<Client>) {
        let mut connections = self.connections.lock().await;
        connections.push(client);
    }

    pub async fn close(&self) {
        let mut connections = self.connections.lock().await;
        connections.clear();
    }

    pub fn size(&self) -> usize {
        self.semaphore.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_pool_config_validation() {
        let config = PoolConfig {
            max_size: 10,
            min_idle: 5,
            timeout: Duration::from_secs(30),
            max_lifetime: Duration::from_secs(3600),
            idle_timeout: Duration::from_secs(600),
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_pool_config() {
        let config = PoolConfig {
            max_size: 0,
            min_idle: 5,
            timeout: Duration::from_secs(30),
            max_lifetime: Duration::from_secs(3600),
            idle_timeout: Duration::from_secs(600),
        };

        assert!(config.validate().is_err());
    }
}
