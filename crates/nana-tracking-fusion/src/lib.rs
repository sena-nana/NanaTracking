//! Exact-frame, framework-neutral NTP fusion.

use core::fmt;

use nana_tracking_protocol::{
    ContractError, NanaGeometryResult, NanaRigResult, NanaSkeletonResult, NanaTrackingDescriptor,
    NanaTrackingQuality, NanaTrackingResult, RegionQuality, SideMap, SignalBitSet, SignalId,
    SignalSample, SignalState, StructureFeatures, Tracked, TrackingFeatures, Validate,
};

/// Bounded decision policy. It selects an existing normalized value and never averages signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialFusionPolicy {
    /// Two normalized scalar values within this distance may corroborate one another.
    pub agreement_tolerance: f32,
    /// The extension must exceed the reference by this margin to replace a non-gaze scalar.
    pub confidence_switch_margin: f32,
}

impl Default for SpatialFusionPolicy {
    fn default() -> Self {
        Self {
            agreement_tolerance: 0.08,
            confidence_switch_margin: 0.12,
        }
    }
}

impl SpatialFusionPolicy {
    fn validate(self) -> Result<Self, FusionError> {
        if !self.agreement_tolerance.is_finite()
            || self.agreement_tolerance < 0.0
            || !self.confidence_switch_margin.is_finite()
            || !(0.0..=1.0).contains(&self.confidence_switch_margin)
        {
            return Err(FusionError::InvalidPolicy);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FusionError {
    InvalidPolicy,
    RevisionMismatch,
    SessionMismatch,
    GenerationMismatch,
    SequenceMismatch,
    CaptureTimestampMismatch,
    ReferenceContract(ContractError),
    ExtensionContract(ContractError),
    OutputContract(ContractError),
}

impl fmt::Display for FusionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy => formatter.write_str("invalid fusion policy"),
            Self::RevisionMismatch => formatter.write_str("input NTP revisions differ"),
            Self::SessionMismatch => formatter.write_str("input session IDs differ"),
            Self::GenerationMismatch => formatter.write_str("input generations differ"),
            Self::SequenceMismatch => formatter.write_str("input sequences differ"),
            Self::CaptureTimestampMismatch => {
                formatter.write_str("input capture timestamps differ")
            }
            Self::ReferenceContract(error) => write!(formatter, "invalid reference input: {error}"),
            Self::ExtensionContract(error) => write!(formatter, "invalid extension input: {error}"),
            Self::OutputContract(error) => write!(formatter, "invalid fused output: {error}"),
        }
    }
}

impl std::error::Error for FusionError {}

/// Form the union capability without clipping additional Spatial, Full, or optional signals.
///
/// # Errors
///
/// Returns an error when inputs do not use the same versioned NTP contract.
pub fn fused_descriptor(
    reference: &NanaTrackingDescriptor,
    extension: &NanaTrackingDescriptor,
) -> Result<NanaTrackingDescriptor, FusionError> {
    reference
        .validate()
        .map_err(FusionError::ReferenceContract)?;
    extension
        .validate()
        .map_err(FusionError::ExtensionContract)?;
    if reference.revisions != extension.revisions {
        return Err(FusionError::RevisionMismatch);
    }
    let mut signals = SignalBitSet::new();
    for id in reference
        .supported_signals
        .iter()
        .chain(extension.supported_signals.iter())
    {
        signals.insert(id);
    }
    let structures = StructureFeatures(
        reference.supported_structures.bits() | extension.supported_structures.bits(),
    );
    let features = TrackingFeatures(reference.features.bits() | extension.features.bits());
    let mut descriptor = NanaTrackingDescriptor::from_capabilities(signals, structures, features);
    descriptor.revisions = reference.revisions;
    Ok(descriptor)
}

/// Fuse two normalized results for one exact capture.
///
/// The reference wins head/eye/face geometry and continuous gaze conflicts. The extension may fill
/// unavailable state and may replace other scalar values only by the configured confidence margin.
/// Agreement changes the state to `Fused` without averaging the two values.
///
/// # Errors
///
/// Returns an error for invalid inputs, incompatible contracts, or any frame-identity mismatch.
pub fn fuse_spatial_results(
    reference_descriptor: &NanaTrackingDescriptor,
    reference: &NanaTrackingResult,
    extension_descriptor: &NanaTrackingDescriptor,
    extension: &NanaTrackingResult,
    policy: SpatialFusionPolicy,
) -> Result<(NanaTrackingDescriptor, NanaTrackingResult), FusionError> {
    let policy = policy.validate()?;
    reference_descriptor
        .validate_result(reference)
        .map_err(FusionError::ReferenceContract)?;
    extension_descriptor
        .validate_result(extension)
        .map_err(FusionError::ExtensionContract)?;
    ensure_same_capture(reference, extension)?;
    let descriptor = fused_descriptor(reference_descriptor, extension_descriptor)?;

    let mut rig = NanaRigResult::unsupported();
    for slot in 0..nana_tracking_protocol::STABLE_SIGNAL_COUNT {
        let Some(id) = SignalId::from_stable_slot(slot) else {
            continue;
        };
        let (Some(reference_sample), Some(extension_sample)) =
            (reference.rig.get(id), extension.rig.get(id))
        else {
            continue;
        };
        let sample = fuse_signal(id, reference_sample, extension_sample, policy);
        let _ = rig.set(id, sample);
    }

    let geometry = fuse_geometry(&reference.geometry, &extension.geometry);
    let skeleton = fuse_skeleton(&reference.skeleton, &extension.skeleton);
    let quality = fuse_quality(&reference.quality, &extension.quality);
    let output = NanaTrackingResult {
        session_id: reference.session_id,
        generation: reference.generation,
        sequence: reference.sequence,
        capture_timestamp_ns: reference.capture_timestamp_ns,
        produced_timestamp_ns: reference
            .produced_timestamp_ns
            .max(extension.produced_timestamp_ns),
        rig,
        geometry,
        skeleton,
        quality,
    };
    descriptor
        .validate_result(&output)
        .map_err(FusionError::OutputContract)?;
    Ok((descriptor, output))
}

fn ensure_same_capture(
    reference: &NanaTrackingResult,
    extension: &NanaTrackingResult,
) -> Result<(), FusionError> {
    if reference.session_id != extension.session_id {
        return Err(FusionError::SessionMismatch);
    }
    if reference.generation != extension.generation {
        return Err(FusionError::GenerationMismatch);
    }
    if reference.sequence != extension.sequence {
        return Err(FusionError::SequenceMismatch);
    }
    if reference.capture_timestamp_ns != extension.capture_timestamp_ns {
        return Err(FusionError::CaptureTimestampMismatch);
    }
    Ok(())
}

fn fused_confidence(left: f32, right: f32) -> f32 {
    (1.0 - (1.0 - left) * (1.0 - right)).clamp(0.0, 1.0)
}

fn fuse_signal(
    id: SignalId,
    reference: &SignalSample,
    extension: &SignalSample,
    policy: SpatialFusionPolicy,
) -> SignalSample {
    match (reference.value, extension.value) {
        (Some(reference_value), Some(extension_value)) => {
            let agrees = (reference_value - extension_value).abs() <= policy.agreement_tolerance;
            let gaze_prefers_reference = matches!(id.get(), 37..=40);
            let choose_extension = !gaze_prefers_reference
                && !agrees
                && extension.confidence >= reference.confidence + policy.confidence_switch_margin;
            let selected = if choose_extension {
                extension
            } else {
                reference
            };
            SignalSample::available(
                selected.value.unwrap_or(reference_value),
                fused_confidence(reference.confidence, extension.confidence),
                SignalState::Fused,
                reference
                    .sample_capture_timestamp_ns
                    .min(extension.sample_capture_timestamp_ns),
                0,
            )
        }
        (Some(_), None) => reference.clone(),
        (None, Some(_)) => extension.clone(),
        (None, None) => choose_unavailable_signal(reference, extension).clone(),
    }
}

fn choose_unavailable<'a, T>(
    reference: &'a Tracked<T>,
    extension: &'a Tracked<T>,
) -> &'a Tracked<T> {
    if reference.state == SignalState::Unsupported
        || (extension.state != SignalState::Unsupported
            && extension.confidence > reference.confidence)
    {
        extension
    } else {
        reference
    }
}

fn choose_unavailable_signal<'a>(
    reference: &'a SignalSample,
    extension: &'a SignalSample,
) -> &'a SignalSample {
    if reference.state == SignalState::Unsupported
        || (extension.state != SignalState::Unsupported
            && extension.confidence > reference.confidence)
    {
        extension
    } else {
        reference
    }
}

fn fuse_tracked<T: Clone>(reference: &Tracked<T>, extension: &Tracked<T>) -> Tracked<T> {
    match (&reference.value, &extension.value) {
        (Some(value), Some(_)) => Tracked::available(
            value.clone(),
            fused_confidence(reference.confidence, extension.confidence),
            SignalState::Fused,
            reference
                .sample_capture_timestamp_ns
                .min(extension.sample_capture_timestamp_ns),
            0,
        ),
        (Some(_), None) => reference.clone(),
        (None, Some(_)) => extension.clone(),
        (None, None) => choose_unavailable(reference, extension).clone(),
    }
}

fn fuse_geometry(
    reference: &NanaGeometryResult,
    extension: &NanaGeometryResult,
) -> NanaGeometryResult {
    let reference_landmarks = reference.face_geometry_state != SignalState::Unsupported;
    NanaGeometryResult {
        head_camera_pose: fuse_tracked(&reference.head_camera_pose, &extension.head_camera_pose),
        eyes: SideMap {
            left: nana_tracking_protocol::EyeGeometry {
                origin_head: fuse_tracked(
                    &reference.eyes.left.origin_head,
                    &extension.eyes.left.origin_head,
                ),
                direction_head: fuse_tracked(
                    &reference.eyes.left.direction_head,
                    &extension.eyes.left.direction_head,
                ),
            },
            right: nana_tracking_protocol::EyeGeometry {
                origin_head: fuse_tracked(
                    &reference.eyes.right.origin_head,
                    &extension.eyes.right.origin_head,
                ),
                direction_head: fuse_tracked(
                    &reference.eyes.right.direction_head,
                    &extension.eyes.right.direction_head,
                ),
            },
        },
        look_at_camera: fuse_tracked(&reference.look_at_camera, &extension.look_at_camera),
        face_geometry_state: fuse_state(
            reference.face_geometry_state,
            extension.face_geometry_state,
        ),
        face_landmarks: if reference_landmarks {
            reference.face_landmarks.clone()
        } else {
            extension.face_landmarks.clone()
        },
    }
}

fn fuse_skeleton(
    reference: &NanaSkeletonResult,
    extension: &NanaSkeletonResult,
) -> NanaSkeletonResult {
    NanaSkeletonResult {
        torso_camera_pose: fuse_tracked(&reference.torso_camera_pose, &extension.torso_camera_pose),
        shoulder: fuse_side(&reference.shoulder, &extension.shoulder),
        elbow: fuse_side(&reference.elbow, &extension.elbow),
        wrist: fuse_side(&reference.wrist, &extension.wrist),
        upper_arm_direction_torso: fuse_side(
            &reference.upper_arm_direction_torso,
            &extension.upper_arm_direction_torso,
        ),
        forearm_direction_torso: fuse_side(
            &reference.forearm_direction_torso,
            &extension.forearm_direction_torso,
        ),
        upper_arm_twist: fuse_side(&reference.upper_arm_twist, &extension.upper_arm_twist),
        forearm_twist: fuse_side(&reference.forearm_twist, &extension.forearm_twist),
    }
}

fn fuse_side<T: Clone>(
    reference: &SideMap<Tracked<T>>,
    extension: &SideMap<Tracked<T>>,
) -> SideMap<Tracked<T>> {
    SideMap {
        left: fuse_tracked(&reference.left, &extension.left),
        right: fuse_tracked(&reference.right, &extension.right),
    }
}

fn fuse_state(reference: SignalState, extension: SignalState) -> SignalState {
    match (reference, extension) {
        (SignalState::Unsupported, state) | (state, SignalState::Unsupported) => state,
        (left, right) if left.carries_value() && right.carries_value() => SignalState::Fused,
        (state, _) => state,
    }
}

fn fuse_region(reference: RegionQuality, extension: RegionQuality) -> RegionQuality {
    RegionQuality {
        confidence: fused_confidence(reference.confidence, extension.confidence),
        state: fuse_state(reference.state, extension.state),
    }
}

fn fuse_quality(
    reference: &NanaTrackingQuality,
    extension: &NanaTrackingQuality,
) -> NanaTrackingQuality {
    NanaTrackingQuality {
        overall_confidence: fused_confidence(
            reference.overall_confidence,
            extension.overall_confidence,
        ),
        face: fuse_region(reference.face, extension.face),
        eyes: fuse_region(reference.eyes, extension.eyes),
        torso: fuse_region(reference.torso, extension.torso),
        arm: SideMap {
            left: fuse_region(reference.arm.left, extension.arm.left),
            right: fuse_region(reference.arm.right, extension.arm.right),
        },
        auricle: SideMap {
            left: fuse_region(reference.auricle.left, extension.auricle.left),
            right: fuse_region(reference.auricle.right, extension.auricle.right),
        },
        stabilization_revision: reference.stabilization_revision,
    }
}
