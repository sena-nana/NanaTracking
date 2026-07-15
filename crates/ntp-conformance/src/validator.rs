use nana_tracking_protocol::{
    ContractError, Direction3, LengthBasis, NanaTrackingDescriptor, NanaTrackingResult, Pose,
    Quaternion, SignalId, SignalState, Tracked, Validate, Vec3,
};
use nana_tracking_semantics::{SemanticDeriver, SemanticFrame, SemanticId, Side};
use serde::{Deserialize, Serialize};

use crate::{CertificationReport, FailureCode, Finding, ProfileAssessment, assess_profile};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformanceOptions {
    /// Product policy limit; protocol legality still independently requires produced >= capture.
    pub max_capture_to_result_ns: u64,
    pub max_sample_age_ns: u64,
    pub max_prediction_horizon_ns: u64,
    pub semantic_tolerance: f32,
    pub skeleton_scalar_tolerance: f32,
    pub direction_tolerance: f32,
}

impl Default for ConformanceOptions {
    fn default() -> Self {
        Self {
            max_capture_to_result_ns: u64::MAX,
            max_sample_age_ns: u64::MAX,
            max_prediction_horizon_ns: u64::MAX,
            semantic_tolerance: 1.0e-5,
            skeleton_scalar_tolerance: 0.05,
            direction_tolerance: 0.02,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TemporalSample {
    state: SignalState,
    confidence: f32,
    sample_capture_timestamp_ns: u64,
    prediction_horizon_ns: u64,
}

/// Stateful validator for one descriptor and one session. Generation advances are accepted only in
/// the forward direction and reset all sequence, timestamp, and semantic history.
pub struct ConformanceValidator {
    descriptor: NanaTrackingDescriptor,
    options: ConformanceOptions,
    report: CertificationReport,
    descriptor_valid: bool,
    session_id: Option<nana_tracking_protocol::SessionId>,
    generation: Option<u32>,
    last_sequence: Option<u64>,
    last_capture_timestamp_ns: Option<u64>,
    last_samples: [Option<TemporalSample>; nana_tracking_protocol::STABLE_SIGNAL_COUNT],
    deriver: SemanticDeriver,
}

impl ConformanceValidator {
    #[must_use]
    pub fn new(descriptor: NanaTrackingDescriptor, options: ConformanceOptions) -> Self {
        let profile = assess_profile(&descriptor);
        let mut report = CertificationReport::new(descriptor.revisions, profile);
        let descriptor_valid = match descriptor.validate() {
            Ok(()) => true,
            Err(error) => {
                report.push(contract_finding(error, None));
                false
            }
        };
        if !options.semantic_tolerance.is_finite()
            || options.semantic_tolerance < 0.0
            || !options.skeleton_scalar_tolerance.is_finite()
            || options.skeleton_scalar_tolerance < 0.0
            || !options.direction_tolerance.is_finite()
            || !(0.0..=2.0).contains(&options.direction_tolerance)
        {
            report.push(Finding::error(
                FailureCode::InvalidConfiguration,
                None,
                None,
                "tolerances must be finite and non-negative; direction tolerance must be <= 2",
            ));
        }
        Self {
            descriptor,
            options,
            report,
            descriptor_valid,
            session_id: None,
            generation: None,
            last_sequence: None,
            last_capture_timestamp_ns: None,
            last_samples: [None; nana_tracking_protocol::STABLE_SIGNAL_COUNT],
            deriver: SemanticDeriver::default(),
        }
    }

    #[must_use]
    pub fn profile(&self) -> &ProfileAssessment {
        &self.report.profile
    }

    pub fn validate_frame(&mut self, result: &NanaTrackingResult) {
        let frame_index = self.report.frames_seen;
        self.report.frames_seen += 1;

        let contract_valid = match result.validate() {
            Ok(()) => true,
            Err(error) => {
                self.report.push(contract_finding(error, Some(frame_index)));
                false
            }
        };
        let stream_valid = contract_valid && self.validate_stream_order(result, frame_index);
        self.validate_latency(result, frame_index);
        self.validate_capabilities(result, frame_index, contract_valid);

        if stream_valid {
            self.validate_sample_time_and_confidence(result, frame_index);
            self.validate_skeleton(result, frame_index);
            match self.deriver.derive(result, result.produced_timestamp_ns) {
                Ok(semantic) => self.validate_derived(result, &semantic, frame_index),
                Err(error) => self.report.push(Finding::error(
                    FailureCode::DerivedOrthogonality,
                    Some(frame_index),
                    None,
                    format!("semantic derivation rejected the frame: {error}"),
                )),
            }
        }
    }

    #[must_use]
    pub fn finish(mut self) -> CertificationReport {
        self.report.finalize();
        self.report
    }

    fn validate_stream_order(&mut self, result: &NanaTrackingResult, frame_index: u64) -> bool {
        let generation_changed = match self.session_id {
            None => true,
            Some(session) if session != result.session_id => {
                self.report.push(Finding::error(
                    FailureCode::WrongSession,
                    Some(frame_index),
                    None,
                    "session changed without starting a new certification stream",
                ));
                return false;
            }
            Some(_) => match self.generation {
                Some(generation) if result.generation < generation => {
                    self.report.push(Finding::error(
                        FailureCode::StaleGeneration,
                        Some(frame_index),
                        None,
                        format!(
                            "generation {} is older than active generation {generation}",
                            result.generation
                        ),
                    ));
                    return false;
                }
                Some(generation) => result.generation > generation,
                None => true,
            },
        };

        let last_sequence = (!generation_changed)
            .then_some(self.last_sequence)
            .flatten();
        let last_capture = (!generation_changed)
            .then_some(self.last_capture_timestamp_ns)
            .flatten();
        if let Some(last) = last_sequence {
            if result.sequence <= last {
                self.report.push(Finding::error(
                    FailureCode::SequenceNotMonotonic,
                    Some(frame_index),
                    None,
                    format!("sequence {} is not newer than {last}", result.sequence),
                ));
                return false;
            }
        }
        if let Some(last) = last_capture
            && result.capture_timestamp_ns < last
        {
            self.report.push(Finding::error(
                FailureCode::CaptureTimestampNotMonotonic,
                Some(frame_index),
                None,
                format!(
                    "capture timestamp {} is older than {last}",
                    result.capture_timestamp_ns
                ),
            ));
            return false;
        }

        if generation_changed {
            self.session_id = Some(result.session_id);
            self.generation = Some(result.generation);
            self.last_samples = [None; nana_tracking_protocol::STABLE_SIGNAL_COUNT];
        } else if let Some(last) = last_sequence {
            self.report.missing_sequences += result.sequence - last - 1;
        }
        self.last_sequence = Some(result.sequence);
        self.last_capture_timestamp_ns = Some(result.capture_timestamp_ns);
        true
    }

    fn validate_latency(&mut self, result: &NanaTrackingResult, frame_index: u64) {
        if result.produced_timestamp_ns >= result.capture_timestamp_ns {
            let latency = result.produced_timestamp_ns - result.capture_timestamp_ns;
            if latency > self.options.max_capture_to_result_ns {
                self.report.push(Finding::error(
                    FailureCode::CaptureToResultExceeded,
                    Some(frame_index),
                    None,
                    format!(
                        "capture-to-result {latency} ns exceeds configured {} ns",
                        self.options.max_capture_to_result_ns
                    ),
                ));
            }
        }
    }

    fn validate_capabilities(
        &mut self,
        result: &NanaTrackingResult,
        frame_index: u64,
        contract_valid: bool,
    ) {
        for (id, sample) in result.rig.iter() {
            let declared = self.descriptor.supported_signals.contains(id);
            if declared == (sample.state == SignalState::Unsupported) {
                let message = if declared {
                    "declared supported capability used Unsupported instead of a runtime state"
                } else {
                    "undeclared capability emitted a runtime state"
                };
                self.report.push(Finding::error(
                    FailureCode::CapabilityStateMismatch,
                    Some(frame_index),
                    Some(id),
                    message,
                ));
            }
        }
        if contract_valid
            && self.descriptor_valid
            && let Err(error) = self.descriptor.validate_result(result)
            && !matches!(error, ContractError::CapabilityMismatch(_))
        {
            self.report.push(contract_finding(error, Some(frame_index)));
        }
    }

    fn validate_sample_time_and_confidence(
        &mut self,
        result: &NanaTrackingResult,
        frame_index: u64,
    ) {
        for (id, sample) in result.rig.iter() {
            if sample.state == SignalState::Unsupported {
                continue;
            }
            if sample.sample_capture_timestamp_ns <= result.capture_timestamp_ns {
                let age = result.capture_timestamp_ns - sample.sample_capture_timestamp_ns;
                if age > self.options.max_sample_age_ns {
                    self.report.push(Finding::error(
                        FailureCode::SampleAgeExceeded,
                        Some(frame_index),
                        Some(id),
                        format!(
                            "sample age {age} ns exceeds configured {} ns",
                            self.options.max_sample_age_ns
                        ),
                    ));
                }
            } else {
                self.report.push(Finding::error(
                    FailureCode::SampleAgeExceeded,
                    Some(frame_index),
                    Some(id),
                    "sample capture timestamp is newer than the result capture timestamp",
                ));
            }
            if sample.prediction_horizon_ns > self.options.max_prediction_horizon_ns {
                self.report.push(Finding::error(
                    FailureCode::PredictionHorizonExceeded,
                    Some(frame_index),
                    Some(id),
                    format!(
                        "prediction horizon {} ns exceeds configured {} ns",
                        sample.prediction_horizon_ns, self.options.max_prediction_horizon_ns
                    ),
                ));
            }
            let slot = id
                .stable_slot()
                .expect("rig iterator only returns stable IDs");
            if let Some(previous) = self.last_samples[slot]
                && previous.sample_capture_timestamp_ns == sample.sample_capture_timestamp_ns
            {
                let prediction_increased = sample.state == SignalState::Predicted
                    && previous.state == SignalState::Predicted
                    && sample.prediction_horizon_ns >= previous.prediction_horizon_ns
                    && sample.confidence > previous.confidence + self.options.semantic_tolerance;
                let occlusion_increased = sample.state == SignalState::Occluded
                    && sample.confidence > previous.confidence + self.options.semantic_tolerance;
                if prediction_increased || occlusion_increased {
                    self.report.push(Finding::error(
                        FailureCode::PredictionConfidenceIncreased,
                        Some(frame_index),
                        Some(id),
                        "confidence increased for the same source sample during prediction/occlusion",
                    ));
                }
            }
            self.last_samples[slot] = Some(TemporalSample {
                state: sample.state,
                confidence: sample.confidence,
                sample_capture_timestamp_ns: sample.sample_capture_timestamp_ns,
                prediction_horizon_ns: sample.prediction_horizon_ns,
            });
        }
    }

    fn validate_skeleton(&mut self, result: &NanaTrackingResult, frame_index: u64) {
        self.validate_pose_rotations(result, frame_index);
        for (side, arm) in [
            (
                "left",
                ArmView {
                    shoulder: &result.skeleton.shoulder.left,
                    elbow: &result.skeleton.elbow.left,
                    wrist: &result.skeleton.wrist.left,
                    upper_direction: &result.skeleton.upper_arm_direction_torso.left,
                    forearm_direction: &result.skeleton.forearm_direction_torso.left,
                    flexion_id: 67,
                    abduction_id: 68,
                    elbow_id: 70,
                    abduction_sign: -1.0,
                },
            ),
            (
                "right",
                ArmView {
                    shoulder: &result.skeleton.shoulder.right,
                    elbow: &result.skeleton.elbow.right,
                    wrist: &result.skeleton.wrist.right,
                    upper_direction: &result.skeleton.upper_arm_direction_torso.right,
                    forearm_direction: &result.skeleton.forearm_direction_torso.right,
                    flexion_id: 72,
                    abduction_id: 73,
                    elbow_id: 75,
                    abduction_sign: 1.0,
                },
            ),
        ] {
            self.validate_arm(result, frame_index, side, &arm);
        }
    }

    fn validate_pose_rotations(&mut self, result: &NanaTrackingResult, frame_index: u64) {
        let torso = result.skeleton.torso_camera_pose.value.as_ref();
        if let Some(torso) = torso {
            let euler = quaternion_to_intrinsic_xyz(torso.orientation_xyzw);
            for (raw, expected, message) in [
                (45, euler.x, "torso pitch disagrees with torso pose"),
                (46, euler.y, "torso yaw disagrees with torso pose"),
                (47, euler.z, "torso roll disagrees with torso pose"),
            ] {
                compare_rig_angle(
                    result,
                    raw,
                    expected,
                    frame_index,
                    self.options.skeleton_scalar_tolerance,
                    &mut self.report,
                    message,
                );
            }
        }
        if let (Some(torso), Some(head)) = (torso, result.geometry.head_camera_pose.value.as_ref())
        {
            let inverse_torso = quaternion_conjugate(torso.orientation_xyzw);
            let head_relative = quaternion_multiply(inverse_torso, head.orientation_xyzw);
            let euler = quaternion_to_intrinsic_xyz(head_relative);
            for (raw, expected, message) in [
                (
                    51,
                    euler.x,
                    "head-relative pitch disagrees with composed poses",
                ),
                (
                    52,
                    euler.y,
                    "head-relative yaw disagrees with composed poses",
                ),
                (
                    53,
                    euler.z,
                    "head-relative roll disagrees with composed poses",
                ),
            ] {
                compare_rig_angle(
                    result,
                    raw,
                    expected,
                    frame_index,
                    self.options.skeleton_scalar_tolerance,
                    &mut self.report,
                    message,
                );
            }
            if torso.length_basis == LengthBasis::HeadRelative
                && head.length_basis == LengthBasis::HeadRelative
            {
                let translation =
                    quaternion_rotate(inverse_torso, subtract(head.position, torso.position));
                for (raw, expected, message) in [
                    (
                        48,
                        translation.x,
                        "head-relative X translation disagrees with composed poses",
                    ),
                    (
                        49,
                        translation.y,
                        "head-relative Y translation disagrees with composed poses",
                    ),
                    (
                        50,
                        translation.z,
                        "head-relative Z translation disagrees with composed poses",
                    ),
                ] {
                    compare_rig_scalar(
                        result,
                        raw,
                        expected,
                        frame_index,
                        self.options.skeleton_scalar_tolerance,
                        &mut self.report,
                        message,
                    );
                }
            }
        }
    }

    fn validate_arm(
        &mut self,
        result: &NanaTrackingResult,
        frame_index: u64,
        side: &str,
        arm: &ArmView<'_>,
    ) {
        check_bone(
            &BoneView {
                side,
                name: "upper arm",
                start: arm.shoulder,
                end: arm.elbow,
                direction: arm.upper_direction,
            },
            frame_index,
            self.options.direction_tolerance,
            &mut self.report,
        );
        check_bone(
            &BoneView {
                side,
                name: "forearm",
                start: arm.elbow,
                end: arm.wrist,
                direction: arm.forearm_direction,
            },
            frame_index,
            self.options.direction_tolerance,
            &mut self.report,
        );

        if let Some(direction) = arm.upper_direction.value.as_ref() {
            compare_rig_scalar(
                result,
                arm.flexion_id,
                direction.value.z,
                frame_index,
                self.options.skeleton_scalar_tolerance,
                &mut self.report,
                "shoulder flexion disagrees with authoritative upper-arm direction",
            );
            compare_rig_scalar(
                result,
                arm.abduction_id,
                direction.value.x * arm.abduction_sign,
                frame_index,
                self.options.skeleton_scalar_tolerance,
                &mut self.report,
                "shoulder abduction disagrees with authoritative upper-arm direction",
            );
        }
        if let (Some(upper), Some(forearm)) = (
            arm.upper_direction.value.as_ref(),
            arm.forearm_direction.value.as_ref(),
        ) {
            let dot = dot(upper.value, forearm.value);
            compare_rig_scalar(
                result,
                arm.elbow_id,
                ((1.0 - dot) * 0.5).clamp(0.0, 1.0),
                frame_index,
                self.options.skeleton_scalar_tolerance,
                &mut self.report,
                "elbow flexion disagrees with authoritative arm directions",
            );
        }
    }

    fn validate_derived(
        &mut self,
        result: &NanaTrackingResult,
        semantic: &SemanticFrame,
        frame_index: u64,
    ) {
        for (left, right, name) in [
            (SemanticId::JawLeft, SemanticId::JawRight, "jaw left/right"),
            (
                SemanticId::EyeBlink(Side::Left),
                SemanticId::EyeWide(Side::Left),
                "left blink/wide",
            ),
            (
                SemanticId::EyeBlink(Side::Right),
                SemanticId::EyeWide(Side::Right),
                "right blink/wide",
            ),
            (
                SemanticId::AuriclePullBack(Side::Left),
                SemanticId::AuriclePullForward(Side::Left),
                "left auricle back/forward",
            ),
            (
                SemanticId::AuriclePullBack(Side::Right),
                SemanticId::AuriclePullForward(Side::Right),
                "right auricle back/forward",
            ),
            (
                SemanticId::AuricleFlatten(Side::Left),
                SemanticId::AuricleFlare(Side::Left),
                "left auricle flatten/flare",
            ),
            (
                SemanticId::AuricleFlatten(Side::Right),
                SemanticId::AuricleFlare(Side::Right),
                "right auricle flatten/flare",
            ),
        ] {
            if let (Some(left), Some(right)) = (semantic.get(left), semantic.get(right))
                && left.value > self.options.semantic_tolerance
                && right.value > self.options.semantic_tolerance
            {
                self.report.push(Finding::error(
                    FailureCode::DerivedOrthogonality,
                    Some(frame_index),
                    None,
                    format!("{name} were simultaneously positive"),
                ));
            }
        }

        if let (Some(pucker), Some(funnel), Some(protrusion)) = (
            semantic.get(SemanticId::MouthPucker),
            semantic.get(SemanticId::MouthFunnel),
            signal_value(result, 29),
        ) && pucker.value + funnel.value > protrusion.max(0.0) + self.options.semantic_tolerance
        {
            self.report.push(Finding::error(
                FailureCode::DerivedOrthogonality,
                Some(frame_index),
                Some(SignalId::new(29).expect("non-zero signal")),
                "mouth pucker and funnel exceed their shared protrusion axis",
            ));
        }

        for side in [Side::Left, Side::Right] {
            let weights = [
                semantic.get(SemanticId::ArmForwardWeight(side)),
                semantic.get(SemanticId::ArmSideWeight(side)),
                semantic.get(SemanticId::ArmBackwardWeight(side)),
            ];
            if let [Some(forward), Some(lateral), Some(backward)] = weights {
                let sum = forward.value + lateral.value + backward.value;
                if sum > self.options.semantic_tolerance
                    && (sum - 1.0).abs() > self.options.semantic_tolerance
                {
                    self.report.push(Finding::error(
                        FailureCode::DerivedOrthogonality,
                        Some(frame_index),
                        None,
                        format!("{side:?} arm directional weights sum to {sum}"),
                    ));
                }
            }
        }
    }
}

#[must_use]
pub fn validate_stream(
    descriptor: &NanaTrackingDescriptor,
    frames: &[NanaTrackingResult],
    options: ConformanceOptions,
) -> CertificationReport {
    let mut validator = ConformanceValidator::new(descriptor.clone(), options);
    for frame in frames {
        validator.validate_frame(frame);
    }
    validator.finish()
}

struct ArmView<'a> {
    shoulder: &'a Tracked<Pose>,
    elbow: &'a Tracked<Pose>,
    wrist: &'a Tracked<Pose>,
    upper_direction: &'a Tracked<Direction3>,
    forearm_direction: &'a Tracked<Direction3>,
    flexion_id: u16,
    abduction_id: u16,
    elbow_id: u16,
    abduction_sign: f32,
}

struct BoneView<'a> {
    side: &'a str,
    name: &'a str,
    start: &'a Tracked<Pose>,
    end: &'a Tracked<Pose>,
    direction: &'a Tracked<Direction3>,
}

fn check_bone(
    bone: &BoneView<'_>,
    frame_index: u64,
    tolerance: f32,
    report: &mut CertificationReport,
) {
    let (Some(start), Some(end)) = (&bone.start.value, &bone.end.value) else {
        return;
    };
    let delta = subtract(end.position, start.position);
    let length = dot(delta, delta).sqrt();
    if !length.is_finite() || length <= f32::EPSILON {
        report.push(Finding::error(
            FailureCode::InvalidBoneLength,
            Some(frame_index),
            None,
            format!("{} {} has zero or non-finite length", bone.side, bone.name),
        ));
        return;
    }
    if let Some(direction) = &bone.direction.value {
        let normalized = scale(delta, 1.0 / length);
        if dot(normalized, direction.value) < 1.0 - tolerance {
            report.push(Finding::error(
                FailureCode::SkeletonScalarMismatch,
                Some(frame_index),
                None,
                format!(
                    "{} {} direction disagrees with joint positions",
                    bone.side, bone.name
                ),
            ));
        }
    }
}

fn compare_rig_scalar(
    result: &NanaTrackingResult,
    raw_id: u16,
    expected: f32,
    frame_index: u64,
    tolerance: f32,
    report: &mut CertificationReport,
    message: &str,
) {
    if let Some(actual) = signal_value(result, raw_id)
        && (actual - expected).abs() > tolerance
    {
        report.push(Finding::error(
            FailureCode::SkeletonScalarMismatch,
            Some(frame_index),
            SignalId::new(raw_id),
            format!("{message}: expected {expected}, got {actual}"),
        ));
    }
}

fn compare_rig_angle(
    result: &NanaTrackingResult,
    raw_id: u16,
    expected: f32,
    frame_index: u64,
    tolerance: f32,
    report: &mut CertificationReport,
    message: &str,
) {
    if let Some(actual) = signal_value(result, raw_id)
        && angle_distance(actual, expected) > tolerance
    {
        report.push(Finding::error(
            FailureCode::SkeletonScalarMismatch,
            Some(frame_index),
            SignalId::new(raw_id),
            format!("{message}: expected {expected}, got {actual}"),
        ));
    }
}

fn signal_value(result: &NanaTrackingResult, raw_id: u16) -> Option<f32> {
    result
        .rig
        .get(SignalId::new(raw_id)?)
        .and_then(|sample| sample.value)
}

const fn subtract(left: Vec3, right: Vec3) -> Vec3 {
    Vec3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

const fn scale(value: Vec3, factor: f32) -> Vec3 {
    Vec3 {
        x: value.x * factor,
        y: value.y * factor,
        z: value.z * factor,
    }
}

const fn dot(left: Vec3, right: Vec3) -> f32 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

const fn quaternion_conjugate(value: Quaternion) -> Quaternion {
    Quaternion {
        x: -value.x,
        y: -value.y,
        z: -value.z,
        w: value.w,
    }
}

const fn quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion {
    Quaternion {
        x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
    }
}

fn quaternion_rotate(rotation: Quaternion, value: Vec3) -> Vec3 {
    let vector = Quaternion {
        x: value.x,
        y: value.y,
        z: value.z,
        w: 0.0,
    };
    let rotated = quaternion_multiply(
        quaternion_multiply(rotation, vector),
        quaternion_conjugate(rotation),
    );
    Vec3 {
        x: rotated.x,
        y: rotated.y,
        z: rotated.z,
    }
}

fn quaternion_to_intrinsic_xyz(value: Quaternion) -> Vec3 {
    let r02 = 2.0 * (value.x * value.z + value.w * value.y);
    let r12 = 2.0 * (value.y * value.z - value.w * value.x);
    let r22 = 1.0 - 2.0 * (value.x * value.x + value.y * value.y);
    let r01 = 2.0 * (value.x * value.y - value.w * value.z);
    let r00 = 1.0 - 2.0 * (value.y * value.y + value.z * value.z);
    Vec3 {
        x: (-r12).atan2(r22),
        y: r02.clamp(-1.0, 1.0).asin(),
        z: (-r01).atan2(r00),
    }
}

fn angle_distance(left: f32, right: f32) -> f32 {
    let two_pi = 2.0 * core::f32::consts::PI;
    let delta = (left - right).rem_euclid(two_pi);
    delta.min(two_pi - delta)
}

fn contract_finding(error: ContractError, frame_index: Option<u64>) -> Finding {
    let (code, signal) = match error {
        ContractError::IncompatibleRevision(_) => (FailureCode::IncompatibleRevision, None),
        ContractError::ProfileMismatch => (FailureCode::ProfileMismatch, None),
        ContractError::ExperimentalSignal(id) => (FailureCode::ExperimentalSignal, Some(id)),
        ContractError::InvalidConfidence(_) => (FailureCode::InvalidConfidence, None),
        ContractError::InvalidStateValue(_) => (FailureCode::StateValueMismatch, None),
        ContractError::InvalidSignalValue(id) => (FailureCode::SignalRange, Some(id)),
        ContractError::InvalidVector(_) => (FailureCode::NonFiniteValue, None),
        ContractError::InvalidUnitVector(_) => (FailureCode::InvalidUnitVector, None),
        ContractError::InvalidQuaternion(_) => (FailureCode::InvalidQuaternion, None),
        ContractError::InvalidCoordinateBinding(_) => (FailureCode::InvalidCoordinateBinding, None),
        ContractError::CapabilityMismatch(id) => (FailureCode::CapabilityStateMismatch, Some(id)),
        ContractError::StructureCapabilityMismatch(_) => {
            (FailureCode::StructureCapabilityMismatch, None)
        }
        ContractError::FeatureDependency(_) => (FailureCode::FeatureDependency, None),
        ContractError::FixedSlotCount(_)
        | ContractError::InvalidTimestamp(_)
        | ContractError::InvalidLandmarkOrder
        | ContractError::UnassignedLandmark(_) => (FailureCode::FrameContract, None),
    };
    Finding::error(code, frame_index, signal, error.to_string())
}
