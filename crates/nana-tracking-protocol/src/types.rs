use alloc::{boxed::Box, vec, vec::Vec};

use serde::{Deserialize, Serialize};

use crate::{
    revision::Revision,
    signal::{STABLE_SIGNAL_COUNT, SignalId},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(C)]
pub struct SessionId(pub [u8; 16]);

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quaternion {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    #[must_use]
    pub fn canonicalized(self) -> Self {
        let should_negate = self.w < 0.0
            || (self.w == 0.0
                && [self.x, self.y, self.z]
                    .into_iter()
                    .find(|component| *component != 0.0)
                    .is_some_and(|component| component < 0.0));
        if should_negate {
            Self {
                x: -self.x,
                y: -self.y,
                z: -self.z,
                w: -self.w,
            }
        } else {
            self
        }
    }

    #[must_use]
    pub fn norm_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CoordinateSpace {
    Camera = 0,
    TorsoLocal = 1,
    HeadLocal = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum LengthBasis {
    Metric = 0,
    HeadRelative = 1,
    TorsoRelative = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Position3 {
    pub space: CoordinateSpace,
    pub length_basis: LengthBasis,
    pub value: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Direction3 {
    pub space: CoordinateSpace,
    pub value: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub parent_space: CoordinateSpace,
    pub length_basis: LengthBasis,
    pub position: Vec3,
    pub orientation_xyzw: Quaternion,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SignalState {
    Observed = 0,
    Fused = 1,
    Predicted = 2,
    Occluded = 3,
    OutOfFrame = 4,
    TrackingLost = 5,
    #[default]
    Unsupported = 6,
}

impl SignalState {
    #[must_use]
    pub const fn carries_value(self) -> bool {
        matches!(self, Self::Observed | Self::Fused | Self::Predicted)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tracked<T> {
    pub value: Option<T>,
    pub confidence: f32,
    pub state: SignalState,
    pub sample_capture_timestamp_ns: u64,
    pub prediction_horizon_ns: u64,
}

impl<T> Tracked<T> {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            value: None,
            confidence: 0.0,
            state: SignalState::Unsupported,
            sample_capture_timestamp_ns: 0,
            prediction_horizon_ns: 0,
        }
    }

    #[must_use]
    pub const fn available(
        value: T,
        confidence: f32,
        state: SignalState,
        sample_capture_timestamp_ns: u64,
        prediction_horizon_ns: u64,
    ) -> Self {
        Self {
            value: Some(value),
            confidence,
            state,
            sample_capture_timestamp_ns,
            prediction_horizon_ns,
        }
    }

    #[must_use]
    pub const fn unavailable(
        confidence: f32,
        state: SignalState,
        sample_capture_timestamp_ns: u64,
    ) -> Self {
        Self {
            value: None,
            confidence,
            state,
            sample_capture_timestamp_ns,
            prediction_horizon_ns: 0,
        }
    }
}

impl<T> Default for Tracked<T> {
    fn default() -> Self {
        Self::unsupported()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignalSample {
    pub value: Option<f32>,
    pub confidence: f32,
    pub state: SignalState,
    pub sample_capture_timestamp_ns: u64,
    pub prediction_horizon_ns: u64,
}

impl SignalSample {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            value: None,
            confidence: 0.0,
            state: SignalState::Unsupported,
            sample_capture_timestamp_ns: 0,
            prediction_horizon_ns: 0,
        }
    }

    #[must_use]
    pub const fn available(
        value: f32,
        confidence: f32,
        state: SignalState,
        sample_capture_timestamp_ns: u64,
        prediction_horizon_ns: u64,
    ) -> Self {
        Self {
            value: Some(value),
            confidence,
            state,
            sample_capture_timestamp_ns,
            prediction_horizon_ns,
        }
    }

    #[must_use]
    pub const fn unavailable(
        confidence: f32,
        state: SignalState,
        sample_capture_timestamp_ns: u64,
    ) -> Self {
        Self {
            value: None,
            confidence,
            state,
            sample_capture_timestamp_ns,
            prediction_horizon_ns: 0,
        }
    }
}

impl Default for SignalSample {
    fn default() -> Self {
        Self::unsupported()
    }
}

/// Fixed stable slots. Slot `n` always represents Signal ID `n + 1`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanaRigResult {
    slots: Box<[SignalSample]>,
}

impl NanaRigResult {
    #[must_use]
    pub fn unsupported() -> Self {
        Self {
            slots: vec![SignalSample::unsupported(); STABLE_SIGNAL_COUNT].into_boxed_slice(),
        }
    }

    #[must_use]
    pub fn slots(&self) -> &[SignalSample] {
        &self.slots
    }

    pub fn iter(&self) -> impl Iterator<Item = (SignalId, &SignalSample)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, sample)| SignalId::from_stable_slot(slot).map(|id| (id, sample)))
    }

    #[must_use]
    pub fn get(&self, id: SignalId) -> Option<&SignalSample> {
        self.slots.get(id.stable_slot()?)
    }

    pub fn set(&mut self, id: SignalId, sample: SignalSample) -> Option<SignalSample> {
        let target = self.slots.get_mut(id.stable_slot()?)?;
        Some(core::mem::replace(target, sample))
    }

    pub(crate) fn from_slots(slots: Box<[SignalSample]>) -> Self {
        Self { slots }
    }
}

impl Default for NanaRigResult {
    fn default() -> Self {
        Self::unsupported()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SideMap<T> {
    pub left: T,
    pub right: T,
}

impl<T: Default> Default for SideMap<T> {
    fn default() -> Self {
        Self {
            left: T::default(),
            right: T::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyeGeometry {
    pub origin_head: Tracked<Position3>,
    pub direction_head: Tracked<Direction3>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceLandmark {
    pub semantic_id: u16,
    pub position_head: Tracked<Position3>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NanaGeometryResult {
    pub head_camera_pose: Tracked<Pose>,
    pub eyes: SideMap<EyeGeometry>,
    pub look_at_camera: Tracked<Position3>,
    /// State of the normalized semantic face-geometry block as a whole.
    pub face_geometry_state: SignalState,
    /// Only NTP-assigned semantic IDs are legal. Vendor topology indices never enter this list.
    pub face_landmarks: Vec<FaceLandmark>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NanaSkeletonResult {
    pub torso_camera_pose: Tracked<Pose>,
    pub shoulder: SideMap<Tracked<Pose>>,
    pub elbow: SideMap<Tracked<Pose>>,
    pub wrist: SideMap<Tracked<Pose>>,
    pub upper_arm_direction_torso: SideMap<Tracked<Direction3>>,
    pub forearm_direction_torso: SideMap<Tracked<Direction3>>,
    pub upper_arm_twist: SideMap<Tracked<f32>>,
    pub forearm_twist: SideMap<Tracked<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionQuality {
    pub confidence: f32,
    pub state: SignalState,
}

impl Default for RegionQuality {
    fn default() -> Self {
        Self {
            confidence: 0.0,
            state: SignalState::Unsupported,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanaTrackingQuality {
    pub overall_confidence: f32,
    pub face: RegionQuality,
    pub eyes: RegionQuality,
    pub torso: RegionQuality,
    pub arm: SideMap<RegionQuality>,
    pub auricle: SideMap<RegionQuality>,
    /// Version of producer-side temporal stabilization, not an algorithm or vendor name.
    pub stabilization_revision: Revision,
}

impl Default for NanaTrackingQuality {
    fn default() -> Self {
        Self {
            overall_confidence: 0.0,
            face: RegionQuality::default(),
            eyes: RegionQuality::default(),
            torso: RegionQuality::default(),
            arm: SideMap::default(),
            auricle: SideMap::default(),
            stabilization_revision: Revision::V1_0_0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanaTrackingResult {
    pub session_id: SessionId,
    pub generation: u32,
    pub sequence: u64,
    pub capture_timestamp_ns: u64,
    pub produced_timestamp_ns: u64,
    pub rig: NanaRigResult,
    pub geometry: NanaGeometryResult,
    pub skeleton: NanaSkeletonResult,
    pub quality: NanaTrackingQuality,
}

impl NanaTrackingResult {
    #[must_use]
    pub fn unsupported(
        session_id: SessionId,
        generation: u32,
        sequence: u64,
        capture_timestamp_ns: u64,
        produced_timestamp_ns: u64,
    ) -> Self {
        Self {
            session_id,
            generation,
            sequence,
            capture_timestamp_ns,
            produced_timestamp_ns,
            rig: NanaRigResult::unsupported(),
            geometry: NanaGeometryResult::default(),
            skeleton: NanaSkeletonResult::default(),
            quality: NanaTrackingQuality::default(),
        }
    }

    #[must_use]
    pub fn sample_age_ns(&self, now_ns: u64) -> u64 {
        now_ns.saturating_sub(self.capture_timestamp_ns)
    }

    #[must_use]
    pub fn processing_latency_ns(&self) -> u64 {
        self.produced_timestamp_ns
            .saturating_sub(self.capture_timestamp_ns)
    }
}
