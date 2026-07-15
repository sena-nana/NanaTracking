use core::fmt;

use crate::{
    capability::{NanaTrackingDescriptor, StructureFeatures},
    revision::ContractRevisions,
    signal::{STABLE_SIGNAL_COUNT, SignalId, SignalMetadata},
    types::{
        Direction3, NanaGeometryResult, NanaSkeletonResult, NanaTrackingResult, Pose, Position3,
        RegionQuality, SignalSample, SignalState, Tracked, Vec3,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractError {
    IncompatibleRevision(&'static str),
    ProfileMismatch,
    ExperimentalSignal(SignalId),
    FixedSlotCount(usize),
    InvalidTimestamp(&'static str),
    InvalidConfidence(&'static str),
    InvalidStateValue(&'static str),
    InvalidSignalValue(SignalId),
    InvalidVector(&'static str),
    InvalidUnitVector(&'static str),
    InvalidQuaternion(&'static str),
    InvalidCoordinateBinding(&'static str),
    InvalidLandmarkOrder,
    UnassignedLandmark(u16),
    CapabilityMismatch(SignalId),
    StructureCapabilityMismatch(&'static str),
    FeatureDependency(&'static str),
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleRevision(name) => write!(formatter, "incompatible {name} revision"),
            Self::ProfileMismatch => {
                formatter.write_str("guaranteed profile is not capability-derived")
            }
            Self::ExperimentalSignal(id) => {
                write!(
                    formatter,
                    "experimental Signal ID {:#06x} in strict v1 descriptor",
                    id.get()
                )
            }
            Self::FixedSlotCount(count) => write!(
                formatter,
                "expected {STABLE_SIGNAL_COUNT} fixed slots, got {count}"
            ),
            Self::InvalidTimestamp(name) => write!(formatter, "invalid timestamp: {name}"),
            Self::InvalidConfidence(name) => write!(formatter, "invalid confidence: {name}"),
            Self::InvalidStateValue(name) => write!(formatter, "state/value mismatch: {name}"),
            Self::InvalidSignalValue(id) => write!(
                formatter,
                "out-of-contract value for Signal ID {}",
                id.get()
            ),
            Self::InvalidVector(name) => write!(formatter, "non-finite vector: {name}"),
            Self::InvalidUnitVector(name) => write!(formatter, "non-unit direction: {name}"),
            Self::InvalidQuaternion(name) => write!(formatter, "invalid quaternion: {name}"),
            Self::InvalidCoordinateBinding(name) => {
                write!(formatter, "invalid coordinate binding: {name}")
            }
            Self::InvalidLandmarkOrder => {
                formatter.write_str("landmark semantic IDs must be non-zero, sorted, and unique")
            }
            Self::UnassignedLandmark(id) => {
                write!(
                    formatter,
                    "landmark semantic ID {id} is not assigned by this registry"
                )
            }
            Self::CapabilityMismatch(id) => write!(
                formatter,
                "Signal ID {} disagrees with descriptor capability",
                id.get()
            ),
            Self::StructureCapabilityMismatch(name) => write!(
                formatter,
                "{name} state disagrees with descriptor capability"
            ),
            Self::FeatureDependency(name) => {
                write!(formatter, "invalid feature dependency: {name}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ContractError {}

pub trait Validate {
    /// Check all semantic invariants owned by this contract value.
    ///
    /// # Errors
    ///
    /// Returns the first contract violation found.
    fn validate(&self) -> Result<(), ContractError>;
}

impl Validate for NanaTrackingDescriptor {
    fn validate(&self) -> Result<(), ContractError> {
        let expected = ContractRevisions::NTP_V1;
        if self.revisions.protocol.major != expected.protocol.major {
            return Err(ContractError::IncompatibleRevision("protocol"));
        }
        if self.revisions.schema_revision < expected.schema_revision {
            return Err(ContractError::IncompatibleRevision("schema"));
        }
        for (actual, required, name) in [
            (
                self.revisions.signal_registry,
                expected.signal_registry,
                "Signal Registry",
            ),
            (
                self.revisions.normalization,
                expected.normalization,
                "normalization",
            ),
            (
                self.revisions.calibration,
                expected.calibration,
                "calibration",
            ),
            (self.revisions.features, expected.features, "feature"),
        ] {
            if actual.major != required.major {
                return Err(ContractError::IncompatibleRevision(name));
            }
        }
        if let Some(experimental) = self
            .supported_signals
            .iter()
            .find(|id| id.is_experimental())
        {
            return Err(ContractError::ExperimentalSignal(experimental));
        }
        let derived = Self::highest_profile(&self.supported_signals, self.supported_structures);
        if self.guaranteed_profile != derived {
            return Err(ContractError::ProfileMismatch);
        }
        if self
            .features
            .contains(crate::capability::TrackingFeatures::DENSE_FACE_MESH)
            && !self
                .supported_structures
                .contains(StructureFeatures::FACE_GEOMETRY)
        {
            return Err(ContractError::FeatureDependency(
                "dense_face_mesh requires face geometry",
            ));
        }
        if self
            .features
            .contains(crate::capability::TrackingFeatures::WRIST_POSE)
            && !self
                .supported_structures
                .contains(StructureFeatures::BODY_SKELETON)
        {
            return Err(ContractError::FeatureDependency(
                "wrist_pose requires body skeleton",
            ));
        }
        if self
            .features
            .contains(crate::capability::TrackingFeatures::METRIC_COORDINATES)
            && self.supported_structures.bits() == 0
        {
            return Err(ContractError::FeatureDependency(
                "metric_coordinates requires a structure",
            ));
        }
        Ok(())
    }
}

fn valid_confidence(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn validate_state_shell<T>(tracked: &Tracked<T>, name: &'static str) -> Result<(), ContractError> {
    if !valid_confidence(tracked.confidence) {
        return Err(ContractError::InvalidConfidence(name));
    }
    if tracked.state.carries_value() != tracked.value.is_some() {
        return Err(ContractError::InvalidStateValue(name));
    }
    if tracked.state == SignalState::Unsupported
        && (tracked.confidence != 0.0
            || tracked.sample_capture_timestamp_ns != 0
            || tracked.prediction_horizon_ns != 0)
    {
        return Err(ContractError::InvalidStateValue(name));
    }
    if tracked.state == SignalState::Predicted && tracked.prediction_horizon_ns == 0 {
        return Err(ContractError::InvalidStateValue(name));
    }
    if tracked.state != SignalState::Predicted && tracked.prediction_horizon_ns != 0 {
        return Err(ContractError::InvalidStateValue(name));
    }
    Ok(())
}

fn validate_signal_sample(
    id: SignalId,
    sample: &SignalSample,
    produced_timestamp_ns: u64,
) -> Result<(), ContractError> {
    if !valid_confidence(sample.confidence) {
        return Err(ContractError::InvalidConfidence("signal"));
    }
    if sample.state.carries_value() != sample.value.is_some() {
        return Err(ContractError::InvalidStateValue("signal"));
    }
    if sample.state == SignalState::Unsupported
        && (sample.confidence != 0.0
            || sample.sample_capture_timestamp_ns != 0
            || sample.prediction_horizon_ns != 0)
    {
        return Err(ContractError::InvalidStateValue("unsupported signal"));
    }
    if sample.state == SignalState::Predicted && sample.prediction_horizon_ns == 0 {
        return Err(ContractError::InvalidStateValue("predicted signal"));
    }
    if sample.state != SignalState::Predicted && sample.prediction_horizon_ns != 0 {
        return Err(ContractError::InvalidStateValue("non-predicted signal"));
    }
    if sample.state != SignalState::Unsupported
        && sample.sample_capture_timestamp_ns > produced_timestamp_ns
    {
        return Err(ContractError::InvalidTimestamp(
            "signal sample is from the future",
        ));
    }
    if let Some(value) = sample.value {
        let metadata = SignalMetadata::get(id).expect("fixed stable signal metadata");
        if !metadata.scalar_type.contains(value) {
            return Err(ContractError::InvalidSignalValue(id));
        }
    }
    Ok(())
}

fn validate_vec3(value: Vec3, name: &'static str) -> Result<(), ContractError> {
    if [value.x, value.y, value.z].into_iter().all(f32::is_finite) {
        Ok(())
    } else {
        Err(ContractError::InvalidVector(name))
    }
}

fn validate_position(value: &Position3, name: &'static str) -> Result<(), ContractError> {
    validate_vec3(value.value, name)
}

fn validate_direction(value: &Direction3, name: &'static str) -> Result<(), ContractError> {
    validate_vec3(value.value, name)?;
    let norm_squared = value.value.x * value.value.x
        + value.value.y * value.value.y
        + value.value.z * value.value.z;
    if (norm_squared - 1.0).abs() <= 2.0e-4 {
        Ok(())
    } else {
        Err(ContractError::InvalidUnitVector(name))
    }
}

fn validate_pose(value: &Pose, name: &'static str) -> Result<(), ContractError> {
    validate_vec3(value.position, name)?;
    let norm_squared = value.orientation_xyzw.norm_squared();
    if !norm_squared.is_finite() || (norm_squared - 1.0).abs() > 2.0e-4 {
        return Err(ContractError::InvalidQuaternion(name));
    }
    Ok(())
}

fn validate_tracked<T>(
    value: &Tracked<T>,
    name: &'static str,
    produced_timestamp_ns: u64,
    validate_value: impl FnOnce(&T, &'static str) -> Result<(), ContractError>,
) -> Result<(), ContractError> {
    validate_state_shell(value, name)?;
    if value.state != SignalState::Unsupported
        && value.sample_capture_timestamp_ns > produced_timestamp_ns
    {
        return Err(ContractError::InvalidTimestamp(name));
    }
    if let Some(inner) = &value.value {
        validate_value(inner, name)?;
    }
    Ok(())
}

fn validate_region(value: RegionQuality, name: &'static str) -> Result<(), ContractError> {
    if !valid_confidence(value.confidence) {
        return Err(ContractError::InvalidConfidence(name));
    }
    if value.state == SignalState::Unsupported && value.confidence != 0.0 {
        return Err(ContractError::InvalidStateValue(name));
    }
    Ok(())
}

impl Validate for NanaTrackingResult {
    fn validate(&self) -> Result<(), ContractError> {
        if self.produced_timestamp_ns < self.capture_timestamp_ns {
            return Err(ContractError::InvalidTimestamp("produced before capture"));
        }
        if self.rig.slots().len() != STABLE_SIGNAL_COUNT {
            return Err(ContractError::FixedSlotCount(self.rig.slots().len()));
        }
        for (id, sample) in self.rig.iter() {
            validate_signal_sample(id, sample, self.produced_timestamp_ns)?;
        }
        validate_geometry(&self.geometry, self.produced_timestamp_ns)?;
        validate_skeleton(&self.skeleton, self.produced_timestamp_ns)?;
        if !valid_confidence(self.quality.overall_confidence) {
            return Err(ContractError::InvalidConfidence("overall"));
        }
        for (region, name) in [
            (self.quality.face, "face"),
            (self.quality.eyes, "eyes"),
            (self.quality.torso, "torso"),
            (self.quality.arm.left, "left arm"),
            (self.quality.arm.right, "right arm"),
            (self.quality.auricle.left, "left auricle"),
            (self.quality.auricle.right, "right auricle"),
        ] {
            validate_region(region, name)?;
        }
        Ok(())
    }
}

fn validate_geometry(
    geometry: &NanaGeometryResult,
    produced_timestamp_ns: u64,
) -> Result<(), ContractError> {
    validate_tracked(
        &geometry.head_camera_pose,
        "head camera pose",
        produced_timestamp_ns,
        validate_pose,
    )?;
    if let Some(pose) = &geometry.head_camera_pose.value {
        if pose.parent_space != crate::types::CoordinateSpace::Camera {
            return Err(ContractError::InvalidCoordinateBinding("head camera pose"));
        }
    }
    for (eye, side) in [
        (&geometry.eyes.left, "left eye"),
        (&geometry.eyes.right, "right eye"),
    ] {
        validate_tracked(
            &eye.origin_head,
            side,
            produced_timestamp_ns,
            validate_position,
        )?;
        validate_tracked(
            &eye.direction_head,
            side,
            produced_timestamp_ns,
            validate_direction,
        )?;
        if let Some(origin) = &eye.origin_head.value {
            if origin.space != crate::types::CoordinateSpace::HeadLocal
                || origin.length_basis != crate::types::LengthBasis::HeadRelative
            {
                return Err(ContractError::InvalidCoordinateBinding(side));
            }
        }
        if let Some(direction) = &eye.direction_head.value {
            if direction.space != crate::types::CoordinateSpace::HeadLocal {
                return Err(ContractError::InvalidCoordinateBinding(side));
            }
        }
    }
    validate_tracked(
        &geometry.look_at_camera,
        "look-at point",
        produced_timestamp_ns,
        validate_position,
    )?;
    if let Some(position) = &geometry.look_at_camera.value {
        if position.space != crate::types::CoordinateSpace::Camera {
            return Err(ContractError::InvalidCoordinateBinding("look-at point"));
        }
    }
    let mut previous = 0;
    for landmark in &geometry.face_landmarks {
        if landmark.semantic_id == 0 || landmark.semantic_id <= previous {
            return Err(ContractError::InvalidLandmarkOrder);
        }
        previous = landmark.semantic_id;
        validate_tracked(
            &landmark.position_head,
            "face landmark",
            produced_timestamp_ns,
            validate_position,
        )?;
        if let Some(position) = &landmark.position_head.value {
            if position.space != crate::types::CoordinateSpace::HeadLocal
                || position.length_basis != crate::types::LengthBasis::HeadRelative
            {
                return Err(ContractError::InvalidCoordinateBinding("face landmark"));
            }
        }
    }
    Ok(())
}

fn validate_skeleton(
    skeleton: &NanaSkeletonResult,
    produced_timestamp_ns: u64,
) -> Result<(), ContractError> {
    validate_tracked(
        &skeleton.torso_camera_pose,
        "torso camera pose",
        produced_timestamp_ns,
        validate_pose,
    )?;
    if let Some(pose) = &skeleton.torso_camera_pose.value {
        if pose.parent_space != crate::types::CoordinateSpace::Camera {
            return Err(ContractError::InvalidCoordinateBinding("torso camera pose"));
        }
    }
    for (joint, name) in [
        (&skeleton.shoulder.left, "left shoulder"),
        (&skeleton.shoulder.right, "right shoulder"),
        (&skeleton.elbow.left, "left elbow"),
        (&skeleton.elbow.right, "right elbow"),
        (&skeleton.wrist.left, "left wrist"),
        (&skeleton.wrist.right, "right wrist"),
    ] {
        validate_tracked(joint, name, produced_timestamp_ns, validate_pose)?;
        if let Some(pose) = &joint.value {
            if pose.parent_space != crate::types::CoordinateSpace::TorsoLocal {
                return Err(ContractError::InvalidCoordinateBinding(name));
            }
        }
        if let (Some(torso), Some(pose)) = (&skeleton.torso_camera_pose.value, &joint.value) {
            if pose.length_basis != torso.length_basis {
                return Err(ContractError::InvalidCoordinateBinding(
                    "mixed body skeleton length bases",
                ));
            }
        }
    }
    for (direction, name) in [
        (&skeleton.upper_arm_direction_torso.left, "left upper arm"),
        (&skeleton.upper_arm_direction_torso.right, "right upper arm"),
        (&skeleton.forearm_direction_torso.left, "left forearm"),
        (&skeleton.forearm_direction_torso.right, "right forearm"),
    ] {
        validate_tracked(direction, name, produced_timestamp_ns, validate_direction)?;
        if let Some(direction) = &direction.value {
            if direction.space != crate::types::CoordinateSpace::TorsoLocal {
                return Err(ContractError::InvalidCoordinateBinding(name));
            }
        }
    }
    for (twist, name) in [
        (&skeleton.upper_arm_twist.left, "left upper-arm twist"),
        (&skeleton.upper_arm_twist.right, "right upper-arm twist"),
        (&skeleton.forearm_twist.left, "left forearm twist"),
        (&skeleton.forearm_twist.right, "right forearm twist"),
    ] {
        validate_tracked(twist, name, produced_timestamp_ns, |value, name| {
            if value.is_finite()
                && *value >= -core::f32::consts::PI
                && *value < core::f32::consts::PI
            {
                Ok(())
            } else {
                Err(ContractError::InvalidVector(name))
            }
        })?;
    }
    Ok(())
}

impl NanaTrackingDescriptor {
    /// Validate a frame against this session declaration after validating each object itself.
    ///
    /// # Errors
    ///
    /// Returns a contract or capability mismatch.
    pub fn validate_result(&self, result: &NanaTrackingResult) -> Result<(), ContractError> {
        self.validate()?;
        result.validate()?;
        if self.revisions.signal_registry == crate::revision::Revision::V1_0_0 {
            if let Some(landmark) = result.geometry.face_landmarks.first() {
                return Err(ContractError::UnassignedLandmark(landmark.semantic_id));
            }
        }
        for (id, sample) in result.rig.iter() {
            let declared = self.supported_signals.contains(id);
            if declared == (sample.state == SignalState::Unsupported) {
                return Err(ContractError::CapabilityMismatch(id));
            }
        }
        require_structure_state(
            self.supported_structures
                .contains(StructureFeatures::HEAD_GEOMETRY),
            result.geometry.head_camera_pose.state,
            "head geometry",
        )?;
        let eyes_supported = self
            .supported_structures
            .contains(StructureFeatures::EYE_GEOMETRY);
        for state in [
            result.geometry.eyes.left.origin_head.state,
            result.geometry.eyes.left.direction_head.state,
            result.geometry.eyes.right.origin_head.state,
            result.geometry.eyes.right.direction_head.state,
        ] {
            require_structure_state(eyes_supported, state, "eye geometry")?;
        }
        require_structure_state(
            self.supported_structures
                .contains(StructureFeatures::LOOK_AT_POINT),
            result.geometry.look_at_camera.state,
            "look-at point",
        )?;
        require_structure_state(
            self.supported_structures
                .contains(StructureFeatures::FACE_GEOMETRY),
            result.geometry.face_geometry_state,
            "face geometry",
        )?;
        let skeleton_supported = self
            .supported_structures
            .contains(StructureFeatures::BODY_SKELETON);
        for state in skeleton_states(&result.skeleton) {
            require_structure_state(skeleton_supported, state, "body skeleton")?;
        }
        require_structure_state(
            supports_any(self, 1..=36)
                || self
                    .supported_structures
                    .contains(StructureFeatures::HEAD_GEOMETRY),
            result.quality.face.state,
            "face region quality",
        )?;
        require_structure_state(
            supports_ids(self, &[7, 8, 9, 10, 37, 38, 39, 40]) || eyes_supported,
            result.quality.eyes.state,
            "eyes region quality",
        )?;
        require_structure_state(
            supports_any(self, 42..=53) || skeleton_supported,
            result.quality.torso.state,
            "torso region quality",
        )?;
        require_structure_state(
            supports_ids(self, &[63, 65, 67, 68, 69, 70, 71]) || skeleton_supported,
            result.quality.arm.left.state,
            "left arm region quality",
        )?;
        require_structure_state(
            supports_ids(self, &[64, 66, 72, 73, 74, 75, 76]) || skeleton_supported,
            result.quality.arm.right.state,
            "right arm region quality",
        )?;
        require_structure_state(
            supports_ids(self, &[57, 59, 61, 81]),
            result.quality.auricle.left.state,
            "left auricle region quality",
        )?;
        require_structure_state(
            supports_ids(self, &[58, 60, 62, 82]),
            result.quality.auricle.right.state,
            "right auricle region quality",
        )?;
        Ok(())
    }
}

fn supports_any(
    descriptor: &NanaTrackingDescriptor,
    mut ids: core::ops::RangeInclusive<u16>,
) -> bool {
    ids.any(|raw| {
        descriptor
            .supported_signals
            .contains(SignalId::new(raw).expect("non-zero static Signal ID"))
    })
}

fn supports_ids(descriptor: &NanaTrackingDescriptor, ids: &[u16]) -> bool {
    ids.iter().copied().any(|raw| {
        descriptor
            .supported_signals
            .contains(SignalId::new(raw).expect("non-zero static Signal ID"))
    })
}

fn require_structure_state(
    supported: bool,
    state: SignalState,
    name: &'static str,
) -> Result<(), ContractError> {
    if supported == (state == SignalState::Unsupported) {
        Err(ContractError::StructureCapabilityMismatch(name))
    } else {
        Ok(())
    }
}

fn skeleton_states(skeleton: &NanaSkeletonResult) -> [SignalState; 15] {
    [
        skeleton.torso_camera_pose.state,
        skeleton.shoulder.left.state,
        skeleton.shoulder.right.state,
        skeleton.elbow.left.state,
        skeleton.elbow.right.state,
        skeleton.wrist.left.state,
        skeleton.wrist.right.state,
        skeleton.upper_arm_direction_torso.left.state,
        skeleton.upper_arm_direction_torso.right.state,
        skeleton.forearm_direction_torso.left.state,
        skeleton.forearm_direction_torso.right.state,
        skeleton.upper_arm_twist.left.state,
        skeleton.upper_arm_twist.right.state,
        skeleton.forearm_twist.left.state,
        skeleton.forearm_twist.right.state,
    ]
}
