//! Human-readable diagnostics only. JSON is not an NTP interchange or persistence format.

use alloc::string::String;

use serde::{Serialize, de::DeserializeOwned};

/// Serialize a value for human diagnostics.
///
/// # Errors
///
/// Returns a serde error when the value cannot be represented as JSON.
pub fn to_pretty_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(value)
}

/// Parse a diagnostic JSON value. Do not use this as the protocol codec.
///
/// # Errors
///
/// Returns a serde error for malformed or type-incompatible JSON.
pub fn from_json<T: DeserializeOwned>(value: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(value)
}
