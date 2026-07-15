use std::fmt;

use nana_tracking_protocol::{ContractRevisions, SignalId, TrackingProfile};
use serde::{Deserialize, Serialize};

use crate::{CONFORMANCE_SCHEMA, ProfileAssessment};

/// Stable failure categories for CI and producer certification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    InvalidConfiguration,
    DescriptorContract,
    IncompatibleRevision,
    ProfileMismatch,
    ExperimentalSignal,
    FeatureDependency,
    FrameContract,
    SignalRange,
    NonFiniteValue,
    StateValueMismatch,
    InvalidConfidence,
    CapabilityStateMismatch,
    StructureCapabilityMismatch,
    InvalidCoordinateBinding,
    InvalidQuaternion,
    InvalidUnitVector,
    InvalidBoneLength,
    SkeletonScalarMismatch,
    WrongSession,
    StaleGeneration,
    SequenceNotMonotonic,
    CaptureTimestampNotMonotonic,
    CaptureToResultExceeded,
    SampleAgeExceeded,
    PredictionHorizonExceeded,
    PredictionConfidenceIncreased,
    DerivedOrthogonality,
    MirrorMismatch,
    EmptyEvidence,
    DecodeFailure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    pub code: FailureCode,
    pub severity: Severity,
    pub frame_index: Option<u64>,
    pub signal_id: Option<u16>,
    pub message: String,
}

impl Finding {
    pub(crate) fn error(
        code: FailureCode,
        frame_index: Option<u64>,
        signal_id: Option<SignalId>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            severity: Severity::Error,
            frame_index,
            signal_id: signal_id.map(SignalId::get),
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertificationReport {
    pub report_schema: String,
    pub revisions: ContractRevisions,
    pub profile: ProfileAssessment,
    pub certified_profile: Option<TrackingProfile>,
    pub frames_seen: u64,
    pub missing_sequences: u64,
    pub passed: bool,
    pub findings: Vec<Finding>,
}

impl CertificationReport {
    pub(crate) fn new(revisions: ContractRevisions, profile: ProfileAssessment) -> Self {
        Self {
            report_schema: CONFORMANCE_SCHEMA.into(),
            revisions,
            profile,
            certified_profile: None,
            frames_seen: 0,
            missing_sequences: 0,
            passed: false,
            findings: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, finding: Finding) {
        self.findings.push(finding);
    }

    pub(crate) fn finalize(&mut self) {
        if self.frames_seen == 0
            && !self
                .findings
                .iter()
                .any(|finding| finding.code == FailureCode::EmptyEvidence)
        {
            self.push(Finding::error(
                FailureCode::EmptyEvidence,
                None,
                None,
                "certification requires at least one result frame",
            ));
        }
        self.passed = !self
            .findings
            .iter()
            .any(|finding| finding.severity == Severity::Error);
        self.certified_profile = self.passed.then_some(self.profile.computed_profile);
    }
}

impl fmt::Display for CertificationReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "NTP conformance: {} ({:?}, {} frames, {} missing sequences)",
            if self.passed { "PASS" } else { "FAIL" },
            self.profile.computed_profile,
            self.frames_seen,
            self.missing_sequences
        )?;
        for finding in &self.findings {
            write!(formatter, "- {:?}", finding.code)?;
            if let Some(frame) = finding.frame_index {
                write!(formatter, " frame={frame}")?;
            }
            if let Some(signal) = finding.signal_id {
                write!(formatter, " signal={signal}")?;
            }
            writeln!(formatter, ": {}", finding.message)?;
        }
        Ok(())
    }
}
