#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MAX_STABLE_SIGNALS: usize = 88;
pub const UPPER_BODY_JOINT_COUNT: usize = 6;
const MAX_STABLE_SIGNAL_ID: u16 = 88;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RuntimeMode {
    Performance,
    Quality,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ActiveProvider {
    OnnxRuntimeCpu,
    OnnxRuntimeCuda,
    OnnxRuntimeTensorRt,
    OnnxRuntimeDirectMl,
    CoreMl,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuntimeModeContract {
    pub precision: String,
    pub scheduling: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackingModelMetadata {
    pub package_schema_version: String,
    pub model_family: String,
    pub model_version: String,
    pub source_checkpoint_digest: String,
    pub ntp_schema_revision: String,
    pub signal_registry_revision: String,
    pub normalization_revision: String,
    pub calibration_revision: String,
    pub feature_revision: String,
    pub onnx_opset: u32,
    pub input_shape: Vec<usize>,
    pub output_names: Vec<String>,
    pub model_digest: String,
    pub smoke_only: bool,
    pub input_layout: String,
    pub input_color: String,
    pub input_range: (f32, f32),
    pub precision_support: Vec<String>,
    pub guaranteed_profile: String,
    pub supported_signals: Vec<u16>,
    pub supported_structures: Vec<String>,
    pub supported_features: Vec<String>,
    pub temporal_state: String,
    pub allowed_backends: Vec<String>,
    pub runtime_modes: std::collections::BTreeMap<String, RuntimeModeContract>,
    pub adapter_schema_version: String,
    pub geometry_topology_revision: Option<String>,
    pub dynamic_dimensions: Vec<String>,
    pub required_operators: Vec<String>,
    pub custom_operator_domains: Vec<String>,
}

impl TrackingModelMetadata {
    /// Validate stable identifiers, shapes, digests, and explicitly declared runtime behavior.
    ///
    /// # Errors
    ///
    /// Returns [`TrackingRuntimeError::InvalidMetadata`] when a field is incomplete or unsafe.
    pub fn validate(&self) -> Result<(), TrackingRuntimeError> {
        let nonempty = [
            self.package_schema_version.as_str(),
            self.model_family.as_str(),
            self.model_version.as_str(),
            self.ntp_schema_revision.as_str(),
            self.signal_registry_revision.as_str(),
            self.normalization_revision.as_str(),
            self.calibration_revision.as_str(),
            self.feature_revision.as_str(),
            self.guaranteed_profile.as_str(),
            self.temporal_state.as_str(),
            self.adapter_schema_version.as_str(),
        ];
        if nonempty.into_iter().any(str::is_empty)
            || self.input_shape.len() != 4
            || self.input_shape.contains(&0)
            || self.output_names.is_empty()
            || self.allowed_backends.is_empty()
            || self.required_operators.is_empty()
            || !self.custom_operator_domains.is_empty()
            || !is_sha256(&self.source_checkpoint_digest)
            || !is_sha256(&self.model_digest)
            || self
                .supported_signals
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .supported_signals
                .iter()
                .any(|signal| !(1..=MAX_STABLE_SIGNAL_ID).contains(signal))
        {
            return Err(TrackingRuntimeError::InvalidMetadata);
        }
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackingModelInput<'a> {
    pub rgb: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
    pub capture_timestamp_ns: u64,
    pub generation: u32,
}

impl TrackingModelInput<'_> {
    /// Validate that the borrowed RGB view covers every declared row without copying it.
    ///
    /// # Errors
    ///
    /// Returns [`TrackingRuntimeError::InvalidInput`] for empty or truncated RGB views.
    pub fn validate(&self) -> Result<(), TrackingRuntimeError> {
        let minimum_stride = self
            .width
            .checked_mul(3)
            .ok_or(TrackingRuntimeError::InvalidInput)?;
        let minimum_bytes = self
            .row_stride
            .checked_mul(self.height)
            .ok_or(TrackingRuntimeError::InvalidInput)?;
        if self.width == 0
            || self.height == 0
            || self.row_stride < minimum_stride
            || self.rgb.len() < minimum_bytes
        {
            return Err(TrackingRuntimeError::InvalidInput);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ModelTrackingState {
    #[default]
    Observed,
    Fused,
    Predicted,
    Occluded,
    OutOfFrame,
    TrackingLost,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelScalar {
    pub value: f32,
    pub confidence: f32,
    pub state: ModelTrackingState,
    pub sample_capture_timestamp_ns: u64,
    pub prediction_horizon_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelVector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelQuaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelPose {
    pub position: ModelVector3,
    pub rotation: ModelQuaternion,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelGeometry {
    pub head_camera_pose: Option<ModelPose>,
    pub torso_camera_pose: Option<ModelPose>,
    pub upper_body_joint_positions: [Option<ModelVector3>; UPPER_BODY_JOINT_COUNT],
    pub upper_body_joint_rotations: [Option<ModelQuaternion>; UPPER_BODY_JOINT_COUNT],
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackingModelOutput {
    pub signals: [Option<ModelScalar>; MAX_STABLE_SIGNALS],
    pub geometry: ModelGeometry,
    pub produced_timestamp_ns: u64,
    pub provider: ActiveProvider,
    pub preprocess_ns: u64,
    pub inference_ns: u64,
    pub readback_ns: u64,
}

impl TrackingModelOutput {
    #[must_use]
    pub fn preallocated(provider: ActiveProvider) -> Self {
        Self {
            signals: [None; MAX_STABLE_SIGNALS],
            geometry: ModelGeometry::default(),
            produced_timestamp_ns: 0,
            provider,
            preprocess_ns: 0,
            inference_ns: 0,
            readback_ns: 0,
        }
    }

    pub fn clear(&mut self) {
        self.signals.fill(None);
        self.geometry = ModelGeometry::default();
        self.produced_timestamp_ns = 0;
        self.preprocess_ns = 0;
        self.inference_ns = 0;
        self.readback_ns = 0;
    }
}

pub trait TrackingModelSession: Send {
    fn metadata(&self) -> &TrackingModelMetadata;

    /// Execute one absolute current-state inference into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns a structured runtime error without exposing backend error types.
    fn infer(
        &mut self,
        input: TrackingModelInput<'_>,
        output: &mut TrackingModelOutput,
    ) -> Result<(), TrackingRuntimeError>;

    fn reset_temporal_state(&mut self);
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedModelPackage {
    pub root: PathBuf,
    pub model_path: PathBuf,
    pub metadata: TrackingModelMetadata,
}

/// Load and digest-verify a portable model package without loading an inference backend.
///
/// # Errors
///
/// Returns a structured package or metadata error for missing, malformed, or tampered content.
pub fn verify_model_package(root: &Path) -> Result<VerifiedModelPackage, TrackingRuntimeError> {
    let metadata_path = root.join("runtime-metadata.json");
    let model_path = root.join("model.onnx");
    for relative in [
        "model.onnx",
        "schema.json",
        "signal-registry-revision.json",
        "normalization.json",
        "runtime-metadata.json",
        "calibration-schema.json",
        "adapter-contract.json",
        "test-vectors/input.npz",
        "test-vectors/expected.npz",
        "test-vectors/parity.json",
    ] {
        if !root.join(relative).is_file() {
            return Err(TrackingRuntimeError::MissingPackagePath(relative.into()));
        }
    }
    let metadata: TrackingModelMetadata =
        serde_json::from_slice(&fs::read(&metadata_path).map_err(TrackingRuntimeError::Io)?)
            .map_err(TrackingRuntimeError::Json)?;
    metadata.validate()?;
    let schema: PackageSchema = serde_json::from_slice(
        &fs::read(root.join("schema.json")).map_err(TrackingRuntimeError::Io)?,
    )
    .map_err(TrackingRuntimeError::Json)?;
    if !schema.custom_operator_domains.is_empty() {
        return Err(TrackingRuntimeError::CustomOperatorsUnsupported);
    }
    if schema.required_operators != metadata.required_operators
        || schema.dynamic_dimensions != metadata.dynamic_dimensions
        || schema.custom_operator_domains != metadata.custom_operator_domains
    {
        return Err(TrackingRuntimeError::OperatorContractMismatch);
    }
    let model = fs::read(&model_path).map_err(TrackingRuntimeError::Io)?;
    let actual = format!("{:x}", Sha256::digest(model));
    if actual != metadata.model_digest {
        return Err(TrackingRuntimeError::DigestMismatch);
    }
    Ok(VerifiedModelPackage {
        root: root.to_path_buf(),
        model_path,
        metadata,
    })
}

#[derive(Debug, Deserialize)]
struct PackageSchema {
    dynamic_dimensions: Vec<String>,
    required_operators: Vec<String>,
    custom_operator_domains: Vec<String>,
}

#[derive(Debug)]
pub enum TrackingRuntimeError {
    InvalidInput,
    InvalidMetadata,
    MissingPackagePath(String),
    DigestMismatch,
    CustomOperatorsUnsupported,
    OperatorContractMismatch,
    UnsupportedProvider(String),
    Backend(String),
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for TrackingRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput => formatter.write_str("invalid tracking model input"),
            Self::InvalidMetadata => formatter.write_str("invalid tracking model metadata"),
            Self::MissingPackagePath(path) => write!(formatter, "model package is missing {path}"),
            Self::DigestMismatch => formatter.write_str("model digest does not match metadata"),
            Self::CustomOperatorsUnsupported => {
                formatter.write_str("custom ONNX operator domains are not portable")
            }
            Self::OperatorContractMismatch => {
                formatter.write_str("model operator contract differs between package metadata")
            }
            Self::UnsupportedProvider(provider) => {
                write!(formatter, "unsupported provider {provider}")
            }
            Self::Backend(message) => write!(formatter, "backend failed: {message}"),
            Self::Io(error) => write!(formatter, "model package I/O failed: {error}"),
            Self::Json(error) => write!(formatter, "model package metadata is invalid: {error}"),
        }
    }
}

impl std::error::Error for TrackingRuntimeError {}
