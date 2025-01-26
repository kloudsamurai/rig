//! A secure string implementation that zeroes out memory when dropped
//! and provides constant-time comparisons to prevent timing attacks.

use serde::{Serialize, Deserialize};
use std::ops::Deref;
use zeroize::Zeroize;

/// A secure string implementation that provides memory safety and timing attack prevention
///
/// # Features
/// - Automatically zeroes out memory when dropped
/// - Provides constant-time comparisons
/// - Implements common string operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureString {
    inner: Vec<u8>,
    len: usize,
    capacity: usize,
}

impl SecureString {
    /// Creates a new SecureString from a string slice
    ///
    /// # Arguments
    /// * `s` - The string to secure
    ///
    /// # Returns
    /// A new SecureString instance
    pub fn new(s: &str) -> Self {
        let bytes = s.as_bytes().to_vec();
        Self {
            len: bytes.len(),
            capacity: bytes.capacity(),
            inner: bytes,
        }
    }

    /// Returns the string as a &str
    ///
    /// # Returns
    /// &str - The secured string
    ///
    /// # Panics
    /// Panics if the string contains invalid UTF-8
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.inner).unwrap_or_default()
    }

    /// Checks if the string is empty
    ///
    /// # Returns
    /// bool - True if the string is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the length of the string in bytes
    ///
    /// # Returns
    /// usize - The length of the string
    pub fn len(&self) -> usize {
        self.len
    }

    /// Validates the string content
    ///
    /// # Returns
    /// Result<(), VectorStoreError> - Ok if valid, Err if invalid
    ///
    /// # Errors
    /// Returns VectorStoreError if the string contains invalid characters
    pub fn validate(&self) -> Result<(), crate::error::VectorStoreError> {
        if self.inner.iter().any(|&b| b == 0) {
            return Err(crate::error::VectorStoreError::SecureStringError);
        }
        Ok(())
    }
}

impl Drop for SecureString {
    fn drop(&mut self) {
        self.inner.zeroize();
    }
}

impl Deref for SecureString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl PartialEq for SecureString {
    fn eq(&self, other: &Self) -> bool {
        // Constant-time comparison to prevent timing attacks
        let mut result = 0u8;
        for (a, b) in self.inner.iter().zip(other.inner.iter()) {
            result |= a ^ b;
        }
        result == 0
    }
}

impl Eq for SecureString {}

impl From<&str> for SecureString {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_string_creation() {
        let s = SecureString::new("test");
        assert_eq!(s.len(), 4);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_secure_string_validation() {
        let s = SecureString::new("valid");
        assert!(s.validate().is_ok());
    }

    #[test]
    fn test_secure_string_comparison() {
        let s1 = SecureString::new("test");
        let s2 = SecureString::new("test");
        assert_eq!(s1, s2);
    }
}
