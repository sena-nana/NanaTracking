use std::{fmt, time::Duration};

use nana_tracking_protocol::{
    ContractError, ContractRevisions, Direction3, NanaTrackingResult, Pose, Revision,
    STABLE_SIGNAL_COUNT, SessionId, SignalId, SignalState, Tracked, Validate, Vec3,
};
use serde::{Deserialize, Serialize};

/// Revision of the formulas and history rules implemented by this crate.
pub const SEMANTICS_REVISION: Revision = Revision::V1_0_0;
pub const SEMANTIC_VALUE_COUNT: usize = 70;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    const fn slot(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
}

/// Stable names for deterministic, non-wire semantic views.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub enum SemanticId {
    EyeBlink(Side),
    EyeWide(Side),
    JawLeft,
    JawRight,
    CheekPuff(Side),
    CheekSuck(Side),
    MouthSmile(Side),
    MouthFrown(Side),
    MouthStretch(Side),
    MouthPucker,
    MouthFunnel,
    MouthInteriorVisible,
    UpperTeethVisible,
    LowerTeethVisible,
    TongueVisible,
    BodyLeanForward,
    BodyLeanBackward,
    BodyLeanLeft,
    BodyLeanRight,
    BodyTwistLeft,
    BodyTwistRight,
    ArmRaise(Side),
    ArmRaiseAzimuth(Side),
    ArmForwardWeight(Side),
    ArmSideWeight(Side),
    ArmBackwardWeight(Side),
    ArmBend(Side),
    ArmExtension(Side),
    ArmCrossBody(Side),
    ArmReachForward(Side),
    HandNearFace(Side),
    HandNearChest(Side),
    HandAboveHead(Side),
    AuricleRaise(Side),
    AuriclePullBack(Side),
    AuriclePullForward(Side),
    AuricleFlatten(Side),
    AuricleFlare(Side),
    AuricleWiggleAmplitude(Side),
    AuricleWiggleVelocity(Side),
    AuricleWiggleEnergy(Side),
    AuricleWigglePhase(Side),
}

impl SemanticId {
    #[must_use]
    pub const fn slot(self) -> usize {
        match self {
            Self::EyeBlink(value) => side_slot(value),
            Self::EyeWide(value) => 2 + side_slot(value),
            Self::JawLeft => 4,
            Self::JawRight => 5,
            Self::CheekPuff(value) => 6 + side_slot(value),
            Self::CheekSuck(value) => 8 + side_slot(value),
            Self::MouthSmile(value) => 10 + side_slot(value),
            Self::MouthFrown(value) => 12 + side_slot(value),
            Self::MouthStretch(value) => 14 + side_slot(value),
            Self::MouthPucker => 16,
            Self::MouthFunnel => 17,
            Self::MouthInteriorVisible => 18,
            Self::UpperTeethVisible => 19,
            Self::LowerTeethVisible => 20,
            Self::TongueVisible => 21,
            Self::BodyLeanForward => 22,
            Self::BodyLeanBackward => 23,
            Self::BodyLeanLeft => 24,
            Self::BodyLeanRight => 25,
            Self::BodyTwistLeft => 26,
            Self::BodyTwistRight => 27,
            Self::ArmRaise(value) => 28 + side_slot(value),
            Self::ArmRaiseAzimuth(value) => 30 + side_slot(value),
            Self::ArmForwardWeight(value) => 32 + side_slot(value),
            Self::ArmSideWeight(value) => 34 + side_slot(value),
            Self::ArmBackwardWeight(value) => 36 + side_slot(value),
            Self::ArmBend(value) => 38 + side_slot(value),
            Self::ArmExtension(value) => 40 + side_slot(value),
            Self::ArmCrossBody(value) => 42 + side_slot(value),
            Self::ArmReachForward(value) => 44 + side_slot(value),
            Self::HandNearFace(value) => 46 + side_slot(value),
            Self::HandNearChest(value) => 48 + side_slot(value),
            Self::HandAboveHead(value) => 50 + side_slot(value),
            Self::AuricleRaise(value) => 52 + side_slot(value),
            Self::AuriclePullBack(value) => 54 + side_slot(value),
            Self::AuriclePullForward(value) => 56 + side_slot(value),
            Self::AuricleFlatten(value) => 58 + side_slot(value),
            Self::AuricleFlare(value) => 60 + side_slot(value),
            Self::AuricleWiggleAmplitude(value) => 62 + side_slot(value),
            Self::AuricleWiggleVelocity(value) => 64 + side_slot(value),
            Self::AuricleWiggleEnergy(value) => 66 + side_slot(value),
            Self::AuricleWigglePhase(value) => 68 + side_slot(value),
        }
    }

    #[must_use]
    pub const fn from_slot(slot: usize) -> Option<Self> {
        let side = if slot % 2 == 0 {
            Side::Left
        } else {
            Side::Right
        };
        Some(match slot {
            0..=1 => Self::EyeBlink(side),
            2..=3 => Self::EyeWide(side),
            4 => Self::JawLeft,
            5 => Self::JawRight,
            6..=7 => Self::CheekPuff(side),
            8..=9 => Self::CheekSuck(side),
            10..=11 => Self::MouthSmile(side),
            12..=13 => Self::MouthFrown(side),
            14..=15 => Self::MouthStretch(side),
            16 => Self::MouthPucker,
            17 => Self::MouthFunnel,
            18 => Self::MouthInteriorVisible,
            19 => Self::UpperTeethVisible,
            20 => Self::LowerTeethVisible,
            21 => Self::TongueVisible,
            22 => Self::BodyLeanForward,
            23 => Self::BodyLeanBackward,
            24 => Self::BodyLeanLeft,
            25 => Self::BodyLeanRight,
            26 => Self::BodyTwistLeft,
            27 => Self::BodyTwistRight,
            28..=29 => Self::ArmRaise(side),
            30..=31 => Self::ArmRaiseAzimuth(side),
            32..=33 => Self::ArmForwardWeight(side),
            34..=35 => Self::ArmSideWeight(side),
            36..=37 => Self::ArmBackwardWeight(side),
            38..=39 => Self::ArmBend(side),
            40..=41 => Self::ArmExtension(side),
            42..=43 => Self::ArmCrossBody(side),
            44..=45 => Self::ArmReachForward(side),
            46..=47 => Self::HandNearFace(side),
            48..=49 => Self::HandNearChest(side),
            50..=51 => Self::HandAboveHead(side),
            52..=53 => Self::AuricleRaise(side),
            54..=55 => Self::AuriclePullBack(side),
            56..=57 => Self::AuriclePullForward(side),
            58..=59 => Self::AuricleFlatten(side),
            60..=61 => Self::AuricleFlare(side),
            62..=63 => Self::AuricleWiggleAmplitude(side),
            64..=65 => Self::AuricleWiggleVelocity(side),
            66..=67 => Self::AuricleWiggleEnergy(side),
            68..=69 => Self::AuricleWigglePhase(side),
            _ => return None,
        })
    }
}

const fn side_slot(side: Side) -> usize {
    match side {
        Side::Left => 0,
        Side::Right => 1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DerivedSample {
    pub value: f32,
    pub confidence: f32,
    pub state: SignalState,
    pub sample_capture_timestamp_ns: u64,
    pub sample_age_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticFrame {
    pub revisions: ContractRevisions,
    pub semantics_revision: Revision,
    pub session_id: SessionId,
    pub generation: u32,
    pub sequence: u64,
    pub capture_timestamp_ns: u64,
    pub evaluation_timestamp_ns: u64,
    pub config: SemanticConfig,
    values: Box<[Option<DerivedSample>]>,
}

impl SemanticFrame {
    #[must_use]
    pub fn get(&self, id: SemanticId) -> Option<&DerivedSample> {
        self.values.get(id.slot())?.as_ref()
    }

    pub fn iter(&self) -> impl Iterator<Item = (SemanticId, &DerivedSample)> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(slot, sample)| Some((SemanticId::from_slot(slot)?, sample.as_ref()?)))
    }
}

trait SemanticValueStore {
    fn insert(&mut self, id: SemanticId, sample: DerivedSample);
}

impl SemanticValueStore for [Option<DerivedSample>] {
    fn insert(&mut self, id: SemanticId, sample: DerivedSample) {
        if let Some(target) = self.get_mut(id.slot()) {
            *target = Some(sample);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticConfig {
    pub predicted_confidence_scale: f32,
    pub occluded_confidence_scale: f32,
    pub arm_raise_distance: f32,
    pub arm_reach_distance: f32,
    pub cross_body_distance: f32,
    pub face_center_torso: Vec3,
    pub chest_center_torso: Vec3,
    pub proximity_radius: f32,
    pub head_top_y_torso: f32,
    pub wiggle_velocity_full_scale_per_second: f32,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            predicted_confidence_scale: 0.65,
            occluded_confidence_scale: 0.35,
            arm_raise_distance: 0.45,
            arm_reach_distance: 0.45,
            cross_body_distance: 0.30,
            face_center_torso: Vec3 {
                x: 0.0,
                y: -0.38,
                z: 0.02,
            },
            chest_center_torso: Vec3 {
                x: 0.0,
                y: -0.10,
                z: 0.08,
            },
            proximity_radius: 0.24,
            head_top_y_torso: -0.58,
            wiggle_velocity_full_scale_per_second: 4.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SemanticError {
    InvalidContract(ContractError),
    InvalidConfig,
    OutOfOrderCaptureTimestamp { previous: u64, actual: u64 },
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContract(error) => write!(formatter, "invalid NTP input: {error}"),
            Self::InvalidConfig => formatter.write_str("invalid semantic derivation configuration"),
            Self::OutOfOrderCaptureTimestamp { previous, actual } => write!(
                formatter,
                "capture timestamp moved backwards from {previous} to {actual}"
            ),
        }
    }
}

impl std::error::Error for SemanticError {}

#[derive(Clone, Copy, Debug)]
struct Input {
    value: f32,
    confidence: f32,
    state: SignalState,
    timestamp_ns: u64,
}

impl Input {
    fn derived(self, value: f32, now_ns: u64) -> DerivedSample {
        DerivedSample {
            value,
            confidence: self.confidence,
            state: self.state,
            sample_capture_timestamp_ns: self.timestamp_ns,
            sample_age_ns: now_ns.saturating_sub(self.timestamp_ns),
        }
    }

    fn combine(self, other: Self, value: f32) -> Self {
        Self {
            value,
            confidence: self.confidence.min(other.confidence),
            state: worse_state(self.state, other.state),
            timestamp_ns: self.timestamp_ns.min(other.timestamp_ns),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MotionPoint {
    axes: [f32; 3],
    timestamp_ns: u64,
}

struct ArmInputs<'a> {
    flexion: Option<Input>,
    abduction: Option<Input>,
    elbow_id: u16,
    shoulder: &'a Tracked<Pose>,
    wrist: &'a Tracked<Pose>,
    upper: &'a Tracked<Direction3>,
    forearm: &'a Tracked<Direction3>,
}

/// Stateful reference implementation. It retains values only for explicit occlusion continuity and
/// capture-time motion; it does not smooth producer output.
pub struct SemanticDeriver {
    config: SemanticConfig,
    session_id: Option<SessionId>,
    generation: Option<u32>,
    last_capture_timestamp_ns: Option<u64>,
    last_signals: [Option<Input>; STABLE_SIGNAL_COUNT],
    auricle_motion: [Option<MotionPoint>; 2],
}

impl Default for SemanticDeriver {
    fn default() -> Self {
        Self::new(SemanticConfig::default()).expect("default semantic config is valid")
    }
}

impl SemanticDeriver {
    /// Creates a derivation engine after validating all scale and threshold constants.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticError::InvalidConfig`] when a scale is non-finite or outside its valid
    /// interval.
    pub fn new(config: SemanticConfig) -> Result<Self, SemanticError> {
        let scalars = [
            config.predicted_confidence_scale,
            config.occluded_confidence_scale,
            config.arm_raise_distance,
            config.arm_reach_distance,
            config.cross_body_distance,
            config.proximity_radius,
            config.wiggle_velocity_full_scale_per_second,
        ];
        if !scalars.into_iter().all(f32::is_finite)
            || !(0.0..=1.0).contains(&config.predicted_confidence_scale)
            || !(0.0..=1.0).contains(&config.occluded_confidence_scale)
            || scalars[2..].iter().any(|value| *value <= 0.0)
            || ![
                config.face_center_torso.x,
                config.face_center_torso.y,
                config.face_center_torso.z,
                config.chest_center_torso.x,
                config.chest_center_torso.y,
                config.chest_center_torso.z,
                config.head_top_y_torso,
            ]
            .into_iter()
            .all(f32::is_finite)
        {
            return Err(SemanticError::InvalidConfig);
        }
        Ok(Self {
            config,
            session_id: None,
            generation: None,
            last_capture_timestamp_ns: None,
            last_signals: [None; STABLE_SIGNAL_COUNT],
            auricle_motion: [None; 2],
        })
    }

    pub fn reset(&mut self) {
        self.session_id = None;
        self.generation = None;
        self.last_capture_timestamp_ns = None;
        self.last_signals = [None; STABLE_SIGNAL_COUNT];
        self.auricle_motion = [None; 2];
    }

    /// Derives a frame using source capture timestamps. `now_ns` is used only for explicit age.
    ///
    /// # Errors
    ///
    /// Returns a contract error for invalid NTP input or an ordering error if capture time moves
    /// backwards inside a session generation.
    pub fn derive(
        &mut self,
        result: &NanaTrackingResult,
        now_ns: u64,
    ) -> Result<SemanticFrame, SemanticError> {
        result.validate().map_err(SemanticError::InvalidContract)?;
        if self.session_id != Some(result.session_id) || self.generation != Some(result.generation)
        {
            self.last_signals = [None; STABLE_SIGNAL_COUNT];
            self.auricle_motion = [None; 2];
            self.last_capture_timestamp_ns = None;
            self.session_id = Some(result.session_id);
            self.generation = Some(result.generation);
        }
        if let Some(previous) = self.last_capture_timestamp_ns
            && result.capture_timestamp_ns < previous
        {
            return Err(SemanticError::OutOfOrderCaptureTimestamp {
                previous,
                actual: result.capture_timestamp_ns,
            });
        }

        let mut inputs = [None; STABLE_SIGNAL_COUNT];
        for (id, sample) in result.rig.iter() {
            let Some(slot) = id.stable_slot() else {
                continue;
            };
            let resolved = if let Some(value) = sample.value {
                let confidence = sample.confidence * self.state_scale(sample.state);
                let input = Input {
                    value,
                    confidence,
                    state: sample.state,
                    timestamp_ns: sample.sample_capture_timestamp_ns,
                };
                self.last_signals[slot] = Some(input);
                Some(input)
            } else if sample.state == SignalState::Occluded {
                self.last_signals[slot].map(|previous| Input {
                    confidence: previous.confidence.min(sample.confidence)
                        * self.config.occluded_confidence_scale,
                    state: SignalState::Occluded,
                    ..previous
                })
            } else {
                None
            };
            inputs[slot] = resolved;
        }

        let mut values = vec![None; SEMANTIC_VALUE_COUNT].into_boxed_slice();
        Self::derive_face(&inputs, now_ns, &mut values);
        self.derive_body(result, &inputs, now_ns, &mut values);
        self.derive_auricles(&inputs, now_ns, &mut values);
        self.last_capture_timestamp_ns = Some(result.capture_timestamp_ns);

        Ok(SemanticFrame {
            revisions: ContractRevisions::NTP_V1,
            semantics_revision: SEMANTICS_REVISION,
            session_id: result.session_id,
            generation: result.generation,
            sequence: result.sequence,
            capture_timestamp_ns: result.capture_timestamp_ns,
            evaluation_timestamp_ns: now_ns,
            config: self.config,
            values,
        })
    }

    fn state_scale(&self, state: SignalState) -> f32 {
        match state {
            SignalState::Observed => 1.0,
            SignalState::Fused => 0.95,
            SignalState::Predicted => self.config.predicted_confidence_scale,
            SignalState::Occluded => self.config.occluded_confidence_scale,
            SignalState::OutOfFrame | SignalState::TrackingLost | SignalState::Unsupported => 0.0,
        }
    }

    fn derive_face(
        inputs: &[Option<Input>; STABLE_SIGNAL_COUNT],
        now_ns: u64,
        values: &mut [Option<DerivedSample>],
    ) {
        for (side, aperture, cheek, vertical, horizontal) in
            [(Side::Left, 7, 11, 20, 22), (Side::Right, 8, 12, 21, 23)]
        {
            split(
                inputs,
                aperture,
                SemanticId::EyeBlink(side),
                SemanticId::EyeWide(side),
                now_ns,
                values,
            );
            split(
                inputs,
                cheek,
                SemanticId::CheekSuck(side),
                SemanticId::CheekPuff(side),
                now_ns,
                values,
            );
            split(
                inputs,
                vertical,
                SemanticId::MouthFrown(side),
                SemanticId::MouthSmile(side),
                now_ns,
                values,
            );
            emit_unary(
                inputs,
                horizontal,
                SemanticId::MouthStretch(side),
                f32::max,
                0.0,
                now_ns,
                values,
            );
        }
        split(
            inputs,
            18,
            SemanticId::JawLeft,
            SemanticId::JawRight,
            now_ns,
            values,
        );

        if let (Some(protrusion), Some(roundness), Some(seal)) =
            (input(inputs, 29), input(inputs, 30), input(inputs, 28))
        {
            let protrusion_value = protrusion.value.max(0.0);
            let pucker = protrusion_value
                * (1.0 - roundness.value).clamp(0.0, 1.0)
                * (0.5 + 0.5 * seal.value.clamp(0.0, 1.0));
            let funnel = protrusion_value
                * roundness.value.clamp(0.0, 1.0)
                * (1.0 - seal.value).clamp(0.0, 1.0);
            let combined = protrusion.combine(roundness, 0.0).combine(seal, 0.0);
            values.insert(SemanticId::MouthPucker, combined.derived(pucker, now_ns));
            values.insert(SemanticId::MouthFunnel, combined.derived(funnel, now_ns));
        }

        if let (Some(jaw), Some(seal)) = (input(inputs, 17), input(inputs, 28)) {
            let interior = jaw.value * (1.0 - seal.value).clamp(0.0, 1.0);
            let combined = jaw.combine(seal, interior);
            values.insert(
                SemanticId::MouthInteriorVisible,
                combined.derived(interior, now_ns),
            );
            let upper_bite = input(inputs, 79).map_or(0.0, |sample| sample.value);
            let lower_bite = input(inputs, 80).map_or(0.0, |sample| sample.value);
            let upper_meta = input(inputs, 79).map_or(combined, |bite| combined.combine(bite, 0.0));
            values.insert(
                SemanticId::UpperTeethVisible,
                upper_meta.derived(interior * (1.0 - upper_bite), now_ns),
            );
            let lower_meta = input(inputs, 80).map_or(combined, |bite| combined.combine(bite, 0.0));
            values.insert(
                SemanticId::LowerTeethVisible,
                lower_meta.derived(interior * (1.0 - lower_bite), now_ns),
            );
            if let Some(tongue) = input(inputs, 41) {
                let visible = tongue.value * (0.5 + 0.5 * interior);
                values.insert(
                    SemanticId::TongueVisible,
                    tongue.combine(combined, visible).derived(visible, now_ns),
                );
            }
        }
    }

    fn derive_body(
        &self,
        result: &NanaTrackingResult,
        inputs: &[Option<Input>; STABLE_SIGNAL_COUNT],
        now_ns: u64,
        values: &mut [Option<DerivedSample>],
    ) {
        split(
            inputs,
            45,
            SemanticId::BodyLeanBackward,
            SemanticId::BodyLeanForward,
            now_ns,
            values,
        );
        split(
            inputs,
            47,
            SemanticId::BodyLeanLeft,
            SemanticId::BodyLeanRight,
            now_ns,
            values,
        );
        split(
            inputs,
            46,
            SemanticId::BodyTwistLeft,
            SemanticId::BodyTwistRight,
            now_ns,
            values,
        );
        self.derive_arm(Side::Left, result, inputs, now_ns, values);
        self.derive_arm(Side::Right, result, inputs, now_ns, values);
    }

    fn derive_arm(
        &self,
        side: Side,
        result: &NanaTrackingResult,
        inputs: &[Option<Input>; STABLE_SIGNAL_COUNT],
        now_ns: u64,
        values: &mut [Option<DerivedSample>],
    ) {
        let (flexion_id, abduction_id, elbow_id) = match side {
            Side::Left => (67, 68, 70),
            Side::Right => (72, 73, 75),
        };
        let flexion = input(inputs, flexion_id);
        let abduction = input(inputs, abduction_id);
        let (shoulder, wrist, upper, forearm) = match side {
            Side::Left => (
                &result.skeleton.shoulder.left,
                &result.skeleton.wrist.left,
                &result.skeleton.upper_arm_direction_torso.left,
                &result.skeleton.forearm_direction_torso.left,
            ),
            Side::Right => (
                &result.skeleton.shoulder.right,
                &result.skeleton.wrist.right,
                &result.skeleton.upper_arm_direction_torso.right,
                &result.skeleton.forearm_direction_torso.right,
            ),
        };
        emit_arm_weights(side, flexion, abduction, upper, self, now_ns, values);

        let arm = ArmInputs {
            flexion,
            abduction,
            elbow_id,
            shoulder,
            wrist,
            upper,
            forearm,
        };
        self.derive_arm_pose(side, result, &arm, inputs, now_ns, values);
    }

    fn derive_arm_pose(
        &self,
        side: Side,
        result: &NanaTrackingResult,
        arm: &ArmInputs<'_>,
        inputs: &[Option<Input>; STABLE_SIGNAL_COUNT],
        now_ns: u64,
        values: &mut [Option<DerivedSample>],
    ) {
        let scalar_raise = match (arm.flexion, arm.abduction) {
            (Some(flexion), Some(abduction)) => flexion.value.max(0.0).max(abduction.value.abs()),
            _ => 0.0,
        };
        let mut raise = arm
            .flexion
            .map(|value| value.combine(arm.abduction.unwrap_or(value), scalar_raise));
        if let Some(direction) =
            tracked_input(arm.upper, |value| (-value.value.y).clamp(0.0, 1.0), self)
        {
            raise = Some(direction);
        }
        if let (Some(shoulder), Some(wrist)) = (
            tracked_pose(arm.shoulder, self),
            tracked_pose(arm.wrist, self),
        ) {
            let height = ((shoulder.1.position.y - wrist.1.position.y)
                / self.config.arm_raise_distance)
                .clamp(0.0, 1.0);
            raise = Some(shoulder.0.combine(wrist.0, height));
        }
        if let Some(raise) = raise {
            values.insert(
                SemanticId::ArmRaise(side),
                raise.derived(raise.value, now_ns),
            );
        }

        let bend = if let (Some(upper), Some(forearm)) = (
            tracked_input(arm.upper, |_| 0.0, self),
            tracked_input(arm.forearm, |_| 0.0, self),
        ) {
            let upper_value = upper_direction(result, side);
            let forearm_value = forearm_direction(result, side);
            let dot = upper_value.x * forearm_value.x
                + upper_value.y * forearm_value.y
                + upper_value.z * forearm_value.z;
            Some(upper.combine(forearm, ((1.0 - dot) * 0.5).clamp(0.0, 1.0)))
        } else {
            input(inputs, arm.elbow_id)
        };
        if let Some(bend) = bend {
            values.insert(SemanticId::ArmBend(side), bend.derived(bend.value, now_ns));
            values.insert(
                SemanticId::ArmExtension(side),
                bend.derived(1.0 - bend.value, now_ns),
            );
        }

        self.derive_hand_relations(side, arm.wrist, now_ns, values);
    }

    fn derive_hand_relations(
        &self,
        side: Side,
        wrist: &Tracked<Pose>,
        now_ns: u64,
        values: &mut [Option<DerivedSample>],
    ) {
        if let Some((meta, pose)) = tracked_pose(wrist, self) {
            let side_sign = if side == Side::Left { 1.0 } else { -1.0 };
            let cross =
                (pose.position.x * side_sign / self.config.cross_body_distance).clamp(0.0, 1.0);
            let reach = (pose.position.z / self.config.arm_reach_distance).clamp(0.0, 1.0);
            let near_face = proximity(
                pose.position,
                self.config.face_center_torso,
                self.config.proximity_radius,
            );
            let near_chest = proximity(
                pose.position,
                self.config.chest_center_torso,
                self.config.proximity_radius,
            );
            let above_head = ((self.config.head_top_y_torso - pose.position.y)
                / self.config.proximity_radius)
                .clamp(0.0, 1.0);
            for (id, value) in [
                (SemanticId::ArmCrossBody(side), cross),
                (SemanticId::ArmReachForward(side), reach),
                (SemanticId::HandNearFace(side), near_face),
                (SemanticId::HandNearChest(side), near_chest),
                (SemanticId::HandAboveHead(side), above_head),
            ] {
                values.insert(id, meta.derived(value, now_ns));
            }
        }
    }

    fn derive_auricles(
        &mut self,
        inputs: &[Option<Input>; STABLE_SIGNAL_COUNT],
        now_ns: u64,
        values: &mut [Option<DerivedSample>],
    ) {
        for (side, elevation, protraction, flattening) in
            [(Side::Left, 57, 59, 61), (Side::Right, 58, 60, 62)]
        {
            emit_unary(
                inputs,
                elevation,
                SemanticId::AuricleRaise(side),
                f32::max,
                0.0,
                now_ns,
                values,
            );
            split(
                inputs,
                protraction,
                SemanticId::AuriclePullBack(side),
                SemanticId::AuriclePullForward(side),
                now_ns,
                values,
            );
            split(
                inputs,
                flattening,
                SemanticId::AuricleFlare(side),
                SemanticId::AuricleFlatten(side),
                now_ns,
                values,
            );
            let (Some(elevation), Some(protraction), Some(flattening)) = (
                input(inputs, elevation),
                input(inputs, protraction),
                input(inputs, flattening),
            ) else {
                continue;
            };
            let combined = elevation.combine(protraction, 0.0).combine(flattening, 0.0);
            let axes = [elevation.value, protraction.value, flattening.value];
            let amplitude = ((axes[0] * axes[0] + axes[1] * axes[1] + axes[2] * axes[2]) / 3.0)
                .sqrt()
                .clamp(0.0, 1.0);
            let mut velocity = 0.0;
            if let Some(previous) = self.auricle_motion[side.slot()]
                && combined.timestamp_ns > previous.timestamp_ns
            {
                let dt = Duration::from_nanos(combined.timestamp_ns - previous.timestamp_ns)
                    .as_secs_f32();
                let delta = ((axes[0] - previous.axes[0]).powi(2)
                    + (axes[1] - previous.axes[1]).powi(2)
                    + (axes[2] - previous.axes[2]).powi(2))
                .sqrt();
                velocity = (delta / dt / self.config.wiggle_velocity_full_scale_per_second)
                    .clamp(0.0, 1.0);
            }
            let energy = (amplitude * velocity).clamp(0.0, 1.0);
            let phase = protraction.value.atan2(elevation.value) / std::f32::consts::PI;
            for (id, value) in [
                (SemanticId::AuricleWiggleAmplitude(side), amplitude),
                (SemanticId::AuricleWiggleVelocity(side), velocity),
                (SemanticId::AuricleWiggleEnergy(side), energy),
                (SemanticId::AuricleWigglePhase(side), phase),
            ] {
                values.insert(id, combined.derived(value, now_ns));
            }
            self.auricle_motion[side.slot()] = Some(MotionPoint {
                axes,
                timestamp_ns: combined.timestamp_ns,
            });
        }
    }
}

fn emit_arm_weights(
    side: Side,
    mut flexion: Option<Input>,
    mut abduction: Option<Input>,
    upper: &Tracked<Direction3>,
    deriver: &SemanticDeriver,
    now_ns: u64,
    values: &mut [Option<DerivedSample>],
) {
    if let (Some(direction), Some(meta)) =
        (upper.value.as_ref(), tracked_input(upper, |_| 0.0, deriver))
    {
        let side_sign = if side == Side::Left { -1.0 } else { 1.0 };
        flexion = Some(Input {
            value: direction.value.z,
            ..meta
        });
        abduction = Some(Input {
            value: direction.value.x * side_sign,
            ..meta
        });
    }
    let (Some(flexion), Some(abduction)) = (flexion, abduction) else {
        return;
    };
    let forward = flexion.value.max(0.0);
    let lateral = abduction.value.abs();
    let backward = (-flexion.value).max(0.0);
    let total = forward + lateral + backward;
    let base = flexion.combine(abduction, 0.0);
    let (forward, lateral, backward) = if total > f32::EPSILON {
        (forward / total, lateral / total, backward / total)
    } else {
        (0.0, 0.0, 0.0)
    };
    let azimuth = abduction.value.atan2(flexion.value) / std::f32::consts::PI;
    for (id, value) in [
        (SemanticId::ArmRaiseAzimuth(side), azimuth),
        (SemanticId::ArmForwardWeight(side), forward),
        (SemanticId::ArmSideWeight(side), lateral),
        (SemanticId::ArmBackwardWeight(side), backward),
    ] {
        values.insert(id, base.derived(value, now_ns));
    }
}

fn input(inputs: &[Option<Input>; STABLE_SIGNAL_COUNT], raw_id: u16) -> Option<Input> {
    inputs[usize::from(raw_id - 1)]
}

fn split(
    inputs: &[Option<Input>; STABLE_SIGNAL_COUNT],
    raw_id: u16,
    negative: SemanticId,
    positive: SemanticId,
    now_ns: u64,
    values: &mut [Option<DerivedSample>],
) {
    if let Some(input) = input(inputs, raw_id) {
        values.insert(negative, input.derived((-input.value).max(0.0), now_ns));
        values.insert(positive, input.derived(input.value.max(0.0), now_ns));
    }
}

fn emit_unary(
    inputs: &[Option<Input>; STABLE_SIGNAL_COUNT],
    raw_id: u16,
    id: SemanticId,
    function: fn(f32, f32) -> f32,
    rhs: f32,
    now_ns: u64,
    values: &mut [Option<DerivedSample>],
) {
    if let Some(input) = input(inputs, raw_id) {
        values.insert(id, input.derived(function(input.value, rhs), now_ns));
    }
}

fn worse_state(left: SignalState, right: SignalState) -> SignalState {
    if state_rank(left) >= state_rank(right) {
        left
    } else {
        right
    }
}

const fn state_rank(state: SignalState) -> u8 {
    match state {
        SignalState::Observed => 0,
        SignalState::Fused => 1,
        SignalState::Predicted => 2,
        SignalState::Occluded => 3,
        SignalState::OutOfFrame => 4,
        SignalState::TrackingLost => 5,
        SignalState::Unsupported => 6,
    }
}

fn tracked_input<T>(
    tracked: &Tracked<T>,
    map: impl FnOnce(&T) -> f32,
    deriver: &SemanticDeriver,
) -> Option<Input> {
    tracked.value.as_ref().map(|value| Input {
        value: map(value),
        confidence: tracked.confidence * deriver.state_scale(tracked.state),
        state: tracked.state,
        timestamp_ns: tracked.sample_capture_timestamp_ns,
    })
}

fn tracked_pose<'a>(
    tracked: &'a Tracked<Pose>,
    deriver: &SemanticDeriver,
) -> Option<(Input, &'a Pose)> {
    tracked.value.as_ref().map(|pose| {
        (
            Input {
                value: 0.0,
                confidence: tracked.confidence * deriver.state_scale(tracked.state),
                state: tracked.state,
                timestamp_ns: tracked.sample_capture_timestamp_ns,
            },
            pose,
        )
    })
}

fn upper_direction(result: &NanaTrackingResult, side: Side) -> Vec3 {
    match side {
        Side::Left => result
            .skeleton
            .upper_arm_direction_torso
            .left
            .value
            .as_ref(),
        Side::Right => result
            .skeleton
            .upper_arm_direction_torso
            .right
            .value
            .as_ref(),
    }
    .expect("direction was checked")
    .value
}

fn forearm_direction(result: &NanaTrackingResult, side: Side) -> Vec3 {
    match side {
        Side::Left => result.skeleton.forearm_direction_torso.left.value.as_ref(),
        Side::Right => result.skeleton.forearm_direction_torso.right.value.as_ref(),
    }
    .expect("direction was checked")
    .value
}

fn proximity(value: Vec3, center: Vec3, radius: f32) -> f32 {
    let distance = ((value.x - center.x).powi(2)
        + (value.y - center.y).powi(2)
        + (value.z - center.z).powi(2))
    .sqrt();
    (1.0 - distance / radius).clamp(0.0, 1.0)
}

#[must_use]
/// Constructs a non-zero NTP signal ID for built-in reference tables.
///
/// # Panics
///
/// Panics when `raw` is zero, which is reserved by NTP.
pub const fn signal_id(raw: u16) -> SignalId {
    match SignalId::new(raw) {
        Some(id) => id,
        None => panic!("Signal ID zero is reserved"),
    }
}
