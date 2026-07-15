//! Fixed C ABI views. These types are conversion targets, never the canonical wire encoding.
//!
//! `NtpCoreResultC` contains every fixed v1 core field. Future semantic landmarks and other
//! variable-size extension blocks remain in the canonical codec until their own C ABI revision is
//! assigned; vendor topology pointers are deliberately not accepted here.

use alloc::boxed::Box;
use core::fmt;

use crate::{
    capability::{NanaTrackingDescriptor, StructureFeatures, TrackingFeatures, TrackingProfile},
    revision::{ContractRevisions, ProtocolVersion, Revision},
    signal::{STABLE_SIGNAL_COUNT, SignalBitSet, SignalId},
    types::{
        CoordinateSpace, Direction3, EyeGeometry, LengthBasis, NanaGeometryResult, NanaRigResult,
        NanaSkeletonResult, NanaTrackingQuality, NanaTrackingResult, Pose, Position3, Quaternion,
        RegionQuality, SessionId, SideMap, SignalSample, SignalState, Tracked, Vec3,
    },
    validate::{ContractError, Validate},
};

pub const NTP_SIGNAL_WORD_COUNT: usize = (u16::MAX as usize + 1) / 64;
pub const NTP_STATE_UNSUPPORTED: u8 = 0;
pub const NTP_STATE_OBSERVED: u8 = 1;
pub const NTP_STATE_FUSED: u8 = 2;
pub const NTP_STATE_PREDICTED: u8 = 3;
pub const NTP_STATE_OCCLUDED: u8 = 4;
pub const NTP_STATE_OUT_OF_FRAME: u8 = 5;
pub const NTP_STATE_TRACKING_LOST: u8 = 6;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FfiError {
    InvalidEnum(&'static str, u8),
    InvalidProfile(u8),
    Contract(ContractError),
}

impl fmt::Display for FfiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEnum(name, value) => {
                write!(formatter, "invalid C ABI {name} value {value}")
            }
            Self::InvalidProfile(value) => write!(formatter, "invalid C ABI profile value {value}"),
            Self::Contract(error) => write!(formatter, "invalid C ABI contract: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FfiError {}

impl From<ContractError> for FfiError {
    fn from(value: ContractError) -> Self {
        Self::Contract(value)
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpRevisionC {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl From<Revision> for NtpRevisionC {
    fn from(value: Revision) -> Self {
        Self {
            major: value.major,
            minor: value.minor,
            patch: value.patch,
        }
    }
}

impl From<NtpRevisionC> for Revision {
    fn from(value: NtpRevisionC) -> Self {
        Self {
            major: value.major,
            minor: value.minor,
            patch: value.patch,
        }
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct NtpDescriptorC {
    pub protocol_major: u16,
    pub protocol_minor: u16,
    pub schema_revision: u32,
    pub signal_registry_revision: NtpRevisionC,
    pub normalization_revision: NtpRevisionC,
    pub calibration_revision: NtpRevisionC,
    pub feature_revision: NtpRevisionC,
    pub guaranteed_profile: u8,
    pub reserved: [u8; 7],
    pub supported_signal_words: [u64; NTP_SIGNAL_WORD_COUNT],
    pub supported_structures: u64,
    pub features: u64,
}

impl From<&NanaTrackingDescriptor> for NtpDescriptorC {
    fn from(value: &NanaTrackingDescriptor) -> Self {
        let mut supported_signal_words = [0; NTP_SIGNAL_WORD_COUNT];
        for id in value.supported_signals.iter() {
            let raw = usize::from(id.get());
            supported_signal_words[raw / 64] |= 1_u64 << (raw % 64);
        }
        Self {
            protocol_major: value.revisions.protocol.major,
            protocol_minor: value.revisions.protocol.minor,
            schema_revision: value.revisions.schema_revision,
            signal_registry_revision: value.revisions.signal_registry.into(),
            normalization_revision: value.revisions.normalization.into(),
            calibration_revision: value.revisions.calibration.into(),
            feature_revision: value.revisions.features.into(),
            guaranteed_profile: value.guaranteed_profile as u8,
            reserved: [0; 7],
            supported_signal_words,
            supported_structures: value.supported_structures.bits(),
            features: value.features.bits(),
        }
    }
}

impl TryFrom<&NtpDescriptorC> for NanaTrackingDescriptor {
    type Error = FfiError;

    fn try_from(value: &NtpDescriptorC) -> Result<Self, Self::Error> {
        let mut supported_signals = SignalBitSet::new();
        for (word_index, word) in value.supported_signal_words.iter().copied().enumerate() {
            for bit in 0..64 {
                let raw = word_index * 64 + bit;
                if raw != 0 && word & (1_u64 << bit) != 0 {
                    let raw = u16::try_from(raw).expect("bitmap covers the u16 ID domain");
                    supported_signals.insert(SignalId::new(raw).expect("non-zero u16 ID"));
                }
            }
        }
        let descriptor = Self {
            revisions: ContractRevisions {
                protocol: ProtocolVersion {
                    major: value.protocol_major,
                    minor: value.protocol_minor,
                },
                schema_revision: value.schema_revision,
                signal_registry: value.signal_registry_revision.into(),
                normalization: value.normalization_revision.into(),
                calibration: value.calibration_revision.into(),
                features: value.feature_revision.into(),
            },
            guaranteed_profile: profile_from_u8(value.guaranteed_profile)?,
            supported_signals,
            supported_structures: StructureFeatures(value.supported_structures),
            features: TrackingFeatures(value.features),
        };
        descriptor.validate()?;
        Ok(descriptor)
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpVec3C {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpQuaternionC {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpPoseC {
    pub parent_space: u8,
    pub length_basis: u8,
    pub reserved: [u8; 2],
    pub position: NtpVec3C,
    pub orientation_xyzw: NtpQuaternionC,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpPosition3C {
    pub space: u8,
    pub length_basis: u8,
    pub reserved: [u8; 2],
    pub value: NtpVec3C,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpDirection3C {
    pub space: u8,
    pub reserved: [u8; 3],
    pub value: NtpVec3C,
}

macro_rules! tracked_c {
    ($name:ident, $value:ty) => {
        #[derive(Clone, Copy, Debug, Default)]
        #[repr(C)]
        pub struct $name {
            pub state: u8,
            pub has_value: u8,
            pub reserved: [u8; 2],
            pub confidence: f32,
            pub sample_capture_timestamp_ns: u64,
            pub prediction_horizon_ns: u64,
            pub value: $value,
        }
    };
}

tracked_c!(NtpTrackedPoseC, NtpPoseC);
tracked_c!(NtpTrackedPosition3C, NtpPosition3C);
tracked_c!(NtpTrackedDirection3C, NtpDirection3C);
tracked_c!(NtpTrackedScalarC, f32);

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpSignalSampleC {
    pub state: u8,
    pub has_value: u8,
    pub reserved: [u8; 2],
    pub confidence: f32,
    pub sample_capture_timestamp_ns: u64,
    pub prediction_horizon_ns: u64,
    pub value: f32,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpRegionQualityC {
    pub state: u8,
    pub reserved: [u8; 3],
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpGeometryC {
    pub head_camera_pose: NtpTrackedPoseC,
    pub left_eye_origin_head: NtpTrackedPosition3C,
    pub left_eye_direction_head: NtpTrackedDirection3C,
    pub right_eye_origin_head: NtpTrackedPosition3C,
    pub right_eye_direction_head: NtpTrackedDirection3C,
    pub look_at_camera: NtpTrackedPosition3C,
    pub face_geometry_state: u8,
    pub reserved: [u8; 7],
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpSkeletonC {
    pub torso_camera_pose: NtpTrackedPoseC,
    pub shoulder: [NtpTrackedPoseC; 2],
    pub elbow: [NtpTrackedPoseC; 2],
    pub wrist: [NtpTrackedPoseC; 2],
    pub upper_arm_direction_torso: [NtpTrackedDirection3C; 2],
    pub forearm_direction_torso: [NtpTrackedDirection3C; 2],
    pub upper_arm_twist: [NtpTrackedScalarC; 2],
    pub forearm_twist: [NtpTrackedScalarC; 2],
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NtpQualityC {
    pub overall_confidence: f32,
    pub face: NtpRegionQualityC,
    pub eyes: NtpRegionQualityC,
    pub torso: NtpRegionQualityC,
    pub arm: [NtpRegionQualityC; 2],
    pub auricle: [NtpRegionQualityC; 2],
    pub stabilization_revision: NtpRevisionC,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct NtpCoreResultC {
    pub session_id: [u8; 16],
    pub generation: u32,
    pub sequence: u64,
    pub capture_timestamp_ns: u64,
    pub produced_timestamp_ns: u64,
    pub rig: [NtpSignalSampleC; STABLE_SIGNAL_COUNT],
    pub geometry: NtpGeometryC,
    pub skeleton: NtpSkeletonC,
    pub quality: NtpQualityC,
}

impl From<&NanaTrackingResult> for NtpCoreResultC {
    fn from(value: &NanaTrackingResult) -> Self {
        Self {
            session_id: value.session_id.0,
            generation: value.generation,
            sequence: value.sequence,
            capture_timestamp_ns: value.capture_timestamp_ns,
            produced_timestamp_ns: value.produced_timestamp_ns,
            rig: core::array::from_fn(|slot| signal_to_c(&value.rig.slots()[slot])),
            geometry: geometry_to_c(&value.geometry),
            skeleton: skeleton_to_c(&value.skeleton),
            quality: quality_to_c(&value.quality),
        }
    }
}

impl TryFrom<&NtpCoreResultC> for NanaTrackingResult {
    type Error = FfiError;

    fn try_from(value: &NtpCoreResultC) -> Result<Self, Self::Error> {
        let slots = value
            .rig
            .iter()
            .map(signal_from_c)
            .collect::<Result<Box<[SignalSample]>, _>>()?;
        let result = Self {
            session_id: SessionId(value.session_id),
            generation: value.generation,
            sequence: value.sequence,
            capture_timestamp_ns: value.capture_timestamp_ns,
            produced_timestamp_ns: value.produced_timestamp_ns,
            rig: NanaRigResult::from_slots(slots),
            geometry: geometry_from_c(&value.geometry)?,
            skeleton: skeleton_from_c(&value.skeleton)?,
            quality: quality_from_c(&value.quality)?,
        };
        result.validate()?;
        Ok(result)
    }
}

fn profile_from_u8(value: u8) -> Result<TrackingProfile, FfiError> {
    match value {
        0 => Ok(TrackingProfile::Partial),
        1 => Ok(TrackingProfile::Basic),
        2 => Ok(TrackingProfile::Spatial),
        3 => Ok(TrackingProfile::Full),
        other => Err(FfiError::InvalidProfile(other)),
    }
}

fn state_from_u8(value: u8) -> Result<SignalState, FfiError> {
    match value {
        NTP_STATE_UNSUPPORTED => Ok(SignalState::Unsupported),
        NTP_STATE_OBSERVED => Ok(SignalState::Observed),
        NTP_STATE_FUSED => Ok(SignalState::Fused),
        NTP_STATE_PREDICTED => Ok(SignalState::Predicted),
        NTP_STATE_OCCLUDED => Ok(SignalState::Occluded),
        NTP_STATE_OUT_OF_FRAME => Ok(SignalState::OutOfFrame),
        NTP_STATE_TRACKING_LOST => Ok(SignalState::TrackingLost),
        other => Err(FfiError::InvalidEnum("signal state", other)),
    }
}

const fn state_to_u8(value: SignalState) -> u8 {
    match value {
        SignalState::Unsupported => NTP_STATE_UNSUPPORTED,
        SignalState::Observed => NTP_STATE_OBSERVED,
        SignalState::Fused => NTP_STATE_FUSED,
        SignalState::Predicted => NTP_STATE_PREDICTED,
        SignalState::Occluded => NTP_STATE_OCCLUDED,
        SignalState::OutOfFrame => NTP_STATE_OUT_OF_FRAME,
        SignalState::TrackingLost => NTP_STATE_TRACKING_LOST,
    }
}

fn space_from_u8(value: u8) -> Result<CoordinateSpace, FfiError> {
    match value {
        0 => Ok(CoordinateSpace::Camera),
        1 => Ok(CoordinateSpace::TorsoLocal),
        2 => Ok(CoordinateSpace::HeadLocal),
        other => Err(FfiError::InvalidEnum("coordinate space", other)),
    }
}

fn basis_from_u8(value: u8) -> Result<LengthBasis, FfiError> {
    match value {
        0 => Ok(LengthBasis::Metric),
        1 => Ok(LengthBasis::HeadRelative),
        2 => Ok(LengthBasis::TorsoRelative),
        other => Err(FfiError::InvalidEnum("length basis", other)),
    }
}

fn vec_to_c(value: Vec3) -> NtpVec3C {
    NtpVec3C {
        x: value.x,
        y: value.y,
        z: value.z,
    }
}
fn vec_from_c(value: NtpVec3C) -> Vec3 {
    Vec3 {
        x: value.x,
        y: value.y,
        z: value.z,
    }
}

fn pose_to_c(value: Pose) -> NtpPoseC {
    let q = value.orientation_xyzw.canonicalized();
    NtpPoseC {
        parent_space: value.parent_space as u8,
        length_basis: value.length_basis as u8,
        reserved: [0; 2],
        position: vec_to_c(value.position),
        orientation_xyzw: NtpQuaternionC {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        },
    }
}

fn pose_from_c(value: NtpPoseC) -> Result<Pose, FfiError> {
    Ok(Pose {
        parent_space: space_from_u8(value.parent_space)?,
        length_basis: basis_from_u8(value.length_basis)?,
        position: vec_from_c(value.position),
        orientation_xyzw: Quaternion {
            x: value.orientation_xyzw.x,
            y: value.orientation_xyzw.y,
            z: value.orientation_xyzw.z,
            w: value.orientation_xyzw.w,
        },
    })
}

fn position_to_c(value: Position3) -> NtpPosition3C {
    NtpPosition3C {
        space: value.space as u8,
        length_basis: value.length_basis as u8,
        reserved: [0; 2],
        value: vec_to_c(value.value),
    }
}

fn position_from_c(value: NtpPosition3C) -> Result<Position3, FfiError> {
    Ok(Position3 {
        space: space_from_u8(value.space)?,
        length_basis: basis_from_u8(value.length_basis)?,
        value: vec_from_c(value.value),
    })
}

fn direction_to_c(value: Direction3) -> NtpDirection3C {
    NtpDirection3C {
        space: value.space as u8,
        reserved: [0; 3],
        value: vec_to_c(value.value),
    }
}

fn direction_from_c(value: NtpDirection3C) -> Result<Direction3, FfiError> {
    Ok(Direction3 {
        space: space_from_u8(value.space)?,
        value: vec_from_c(value.value),
    })
}

macro_rules! tracked_converters {
    ($to_name:ident, $from_name:ident, $rust:ty, $c:ident, $to_value:ident, $from_value:ident) => {
        fn $to_name(value: &Tracked<$rust>) -> $c {
            $c {
                state: state_to_u8(value.state),
                has_value: u8::from(value.value.is_some()),
                reserved: [0; 2],
                confidence: value.confidence,
                sample_capture_timestamp_ns: value.sample_capture_timestamp_ns,
                prediction_horizon_ns: value.prediction_horizon_ns,
                value: value.value.map($to_value).unwrap_or_default(),
            }
        }
        fn $from_name(value: &$c) -> Result<Tracked<$rust>, FfiError> {
            let inner = match value.has_value {
                0 => None,
                1 => Some($from_value(value.value)?),
                other => return Err(FfiError::InvalidEnum("value presence", other)),
            };
            Ok(Tracked {
                value: inner,
                confidence: value.confidence,
                state: state_from_u8(value.state)?,
                sample_capture_timestamp_ns: value.sample_capture_timestamp_ns,
                prediction_horizon_ns: value.prediction_horizon_ns,
            })
        }
    };
}

tracked_converters!(
    tracked_pose_to_c,
    tracked_pose_from_c,
    Pose,
    NtpTrackedPoseC,
    pose_to_c,
    pose_from_c
);
tracked_converters!(
    tracked_position_to_c,
    tracked_position_from_c,
    Position3,
    NtpTrackedPosition3C,
    position_to_c,
    position_from_c
);
tracked_converters!(
    tracked_direction_to_c,
    tracked_direction_from_c,
    Direction3,
    NtpTrackedDirection3C,
    direction_to_c,
    direction_from_c
);

fn scalar_to_c(value: &Tracked<f32>) -> NtpTrackedScalarC {
    NtpTrackedScalarC {
        state: state_to_u8(value.state),
        has_value: u8::from(value.value.is_some()),
        reserved: [0; 2],
        confidence: value.confidence,
        sample_capture_timestamp_ns: value.sample_capture_timestamp_ns,
        prediction_horizon_ns: value.prediction_horizon_ns,
        value: value.value.unwrap_or_default(),
    }
}

fn scalar_from_c(value: &NtpTrackedScalarC) -> Result<Tracked<f32>, FfiError> {
    let inner = match value.has_value {
        0 => None,
        1 => Some(value.value),
        other => return Err(FfiError::InvalidEnum("value presence", other)),
    };
    Ok(Tracked {
        value: inner,
        confidence: value.confidence,
        state: state_from_u8(value.state)?,
        sample_capture_timestamp_ns: value.sample_capture_timestamp_ns,
        prediction_horizon_ns: value.prediction_horizon_ns,
    })
}

fn signal_to_c(value: &SignalSample) -> NtpSignalSampleC {
    NtpSignalSampleC {
        state: state_to_u8(value.state),
        has_value: u8::from(value.value.is_some()),
        reserved: [0; 2],
        confidence: value.confidence,
        sample_capture_timestamp_ns: value.sample_capture_timestamp_ns,
        prediction_horizon_ns: value.prediction_horizon_ns,
        value: value.value.unwrap_or_default(),
    }
}

fn signal_from_c(value: &NtpSignalSampleC) -> Result<SignalSample, FfiError> {
    let inner = match value.has_value {
        0 => None,
        1 => Some(value.value),
        other => return Err(FfiError::InvalidEnum("value presence", other)),
    };
    Ok(SignalSample {
        value: inner,
        confidence: value.confidence,
        state: state_from_u8(value.state)?,
        sample_capture_timestamp_ns: value.sample_capture_timestamp_ns,
        prediction_horizon_ns: value.prediction_horizon_ns,
    })
}

fn geometry_to_c(value: &NanaGeometryResult) -> NtpGeometryC {
    NtpGeometryC {
        head_camera_pose: tracked_pose_to_c(&value.head_camera_pose),
        left_eye_origin_head: tracked_position_to_c(&value.eyes.left.origin_head),
        left_eye_direction_head: tracked_direction_to_c(&value.eyes.left.direction_head),
        right_eye_origin_head: tracked_position_to_c(&value.eyes.right.origin_head),
        right_eye_direction_head: tracked_direction_to_c(&value.eyes.right.direction_head),
        look_at_camera: tracked_position_to_c(&value.look_at_camera),
        face_geometry_state: state_to_u8(value.face_geometry_state),
        reserved: [0; 7],
    }
}

fn geometry_from_c(value: &NtpGeometryC) -> Result<NanaGeometryResult, FfiError> {
    Ok(NanaGeometryResult {
        head_camera_pose: tracked_pose_from_c(&value.head_camera_pose)?,
        eyes: SideMap {
            left: EyeGeometry {
                origin_head: tracked_position_from_c(&value.left_eye_origin_head)?,
                direction_head: tracked_direction_from_c(&value.left_eye_direction_head)?,
            },
            right: EyeGeometry {
                origin_head: tracked_position_from_c(&value.right_eye_origin_head)?,
                direction_head: tracked_direction_from_c(&value.right_eye_direction_head)?,
            },
        },
        look_at_camera: tracked_position_from_c(&value.look_at_camera)?,
        face_geometry_state: state_from_u8(value.face_geometry_state)?,
        face_landmarks: alloc::vec::Vec::new(),
    })
}

fn skeleton_to_c(value: &NanaSkeletonResult) -> NtpSkeletonC {
    NtpSkeletonC {
        torso_camera_pose: tracked_pose_to_c(&value.torso_camera_pose),
        shoulder: [
            tracked_pose_to_c(&value.shoulder.left),
            tracked_pose_to_c(&value.shoulder.right),
        ],
        elbow: [
            tracked_pose_to_c(&value.elbow.left),
            tracked_pose_to_c(&value.elbow.right),
        ],
        wrist: [
            tracked_pose_to_c(&value.wrist.left),
            tracked_pose_to_c(&value.wrist.right),
        ],
        upper_arm_direction_torso: [
            tracked_direction_to_c(&value.upper_arm_direction_torso.left),
            tracked_direction_to_c(&value.upper_arm_direction_torso.right),
        ],
        forearm_direction_torso: [
            tracked_direction_to_c(&value.forearm_direction_torso.left),
            tracked_direction_to_c(&value.forearm_direction_torso.right),
        ],
        upper_arm_twist: [
            scalar_to_c(&value.upper_arm_twist.left),
            scalar_to_c(&value.upper_arm_twist.right),
        ],
        forearm_twist: [
            scalar_to_c(&value.forearm_twist.left),
            scalar_to_c(&value.forearm_twist.right),
        ],
    }
}

fn skeleton_from_c(value: &NtpSkeletonC) -> Result<NanaSkeletonResult, FfiError> {
    Ok(NanaSkeletonResult {
        torso_camera_pose: tracked_pose_from_c(&value.torso_camera_pose)?,
        shoulder: SideMap {
            left: tracked_pose_from_c(&value.shoulder[0])?,
            right: tracked_pose_from_c(&value.shoulder[1])?,
        },
        elbow: SideMap {
            left: tracked_pose_from_c(&value.elbow[0])?,
            right: tracked_pose_from_c(&value.elbow[1])?,
        },
        wrist: SideMap {
            left: tracked_pose_from_c(&value.wrist[0])?,
            right: tracked_pose_from_c(&value.wrist[1])?,
        },
        upper_arm_direction_torso: SideMap {
            left: tracked_direction_from_c(&value.upper_arm_direction_torso[0])?,
            right: tracked_direction_from_c(&value.upper_arm_direction_torso[1])?,
        },
        forearm_direction_torso: SideMap {
            left: tracked_direction_from_c(&value.forearm_direction_torso[0])?,
            right: tracked_direction_from_c(&value.forearm_direction_torso[1])?,
        },
        upper_arm_twist: SideMap {
            left: scalar_from_c(&value.upper_arm_twist[0])?,
            right: scalar_from_c(&value.upper_arm_twist[1])?,
        },
        forearm_twist: SideMap {
            left: scalar_from_c(&value.forearm_twist[0])?,
            right: scalar_from_c(&value.forearm_twist[1])?,
        },
    })
}

fn region_to_c(value: RegionQuality) -> NtpRegionQualityC {
    NtpRegionQualityC {
        state: state_to_u8(value.state),
        reserved: [0; 3],
        confidence: value.confidence,
    }
}
fn region_from_c(value: NtpRegionQualityC) -> Result<RegionQuality, FfiError> {
    Ok(RegionQuality {
        confidence: value.confidence,
        state: state_from_u8(value.state)?,
    })
}
fn quality_to_c(value: &NanaTrackingQuality) -> NtpQualityC {
    NtpQualityC {
        overall_confidence: value.overall_confidence,
        face: region_to_c(value.face),
        eyes: region_to_c(value.eyes),
        torso: region_to_c(value.torso),
        arm: [region_to_c(value.arm.left), region_to_c(value.arm.right)],
        auricle: [
            region_to_c(value.auricle.left),
            region_to_c(value.auricle.right),
        ],
        stabilization_revision: value.stabilization_revision.into(),
    }
}
fn quality_from_c(value: &NtpQualityC) -> Result<NanaTrackingQuality, FfiError> {
    Ok(NanaTrackingQuality {
        overall_confidence: value.overall_confidence,
        face: region_from_c(value.face)?,
        eyes: region_from_c(value.eyes)?,
        torso: region_from_c(value.torso)?,
        arm: SideMap {
            left: region_from_c(value.arm[0])?,
            right: region_from_c(value.arm[1])?,
        },
        auricle: SideMap {
            left: region_from_c(value.auricle[0])?,
            right: region_from_c(value.auricle[1])?,
        },
        stabilization_revision: value.stabilization_revision.into(),
    })
}
