//! Framework-neutral NTP v1 certification and stream conformance.
//!
//! The protocol crate owns wire and single-value invariants. This crate aggregates evidence across
//! a producer stream, checks profile/capability promises, validates temporal and articulated-state
//! relationships, and produces stable machine-readable failure reasons.

mod mirror;
mod profile;
mod report;
mod validator;

pub use mirror::{MirrorError, validate_mirror_pair};
pub use profile::{ProfileAssessment, assess_profile};
pub use report::{CertificationReport, FailureCode, Finding, Severity};
pub use validator::{ConformanceOptions, ConformanceValidator, validate_stream};

/// Revisions certified by this implementation.
pub const CONFORMANCE_SCHEMA: &str = "ntp-conformance/1.0.0";
