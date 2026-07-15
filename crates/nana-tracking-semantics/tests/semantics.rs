use nana_tracking_protocol::{
    CoordinateSpace, Direction3, LengthBasis, NanaTrackingDescriptor, NanaTrackingResult, Pose,
    Quaternion, SessionId, SignalBitSet, SignalSample, SignalState, StructureFeatures, Tracked,
    TrackingFeatures, Vec3,
};
use nana_tracking_semantics::{
    BindingCurve, BindingError, BindingEvaluator, BindingLayer, BindingProfile, BindingTransform,
    CombineMode, CurvePoint, LayeredBinding, ModelParameterId, RigBinding, SemanticDeriver,
    SemanticError, SemanticId, Side, SignalExpression, SignalRequirements, arkit_style_52_profile,
    live2d_common_profile, nana_native_profile,
};
use serde::Deserialize;

fn frame(generation: u32, sequence: u64, timestamp_ns: u64) -> NanaTrackingResult {
    NanaTrackingResult::unsupported(
        SessionId([7; 16]),
        generation,
        sequence,
        timestamp_ns,
        timestamp_ns + 10,
    )
}

fn set(result: &mut NanaTrackingResult, raw_id: u16, value: f32, confidence: f32) {
    result.rig.set(
        nana_tracking_semantics::signal_id(raw_id),
        SignalSample::available(
            value,
            confidence,
            SignalState::Observed,
            result.capture_timestamp_ns,
            0,
        ),
    );
}

fn value(semantics: &nana_tracking_semantics::SemanticFrame, id: SemanticId) -> f32 {
    semantics.get(id).expect("semantic value").value
}

fn assert_close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
}

#[test]
fn semantic_slots_are_dense_unique_and_reversible() {
    for slot in 0..nana_tracking_semantics::SEMANTIC_VALUE_COUNT {
        let id = SemanticId::from_slot(slot).expect("assigned semantic slot");
        assert_eq!(id.slot(), slot);
    }
    assert!(SemanticId::from_slot(nana_tracking_semantics::SEMANTIC_VALUE_COUNT).is_none());
}

#[derive(Deserialize)]
struct GoldenExpected {
    eye_blink_left: f32,
    eye_wide_right: f32,
    cheek_suck_left: f32,
    cheek_puff_right: f32,
    jaw_right: f32,
    mouth_smile_left: f32,
    mouth_frown_right: f32,
    mouth_pucker: f32,
    mouth_funnel: f32,
    mouth_interior_visible: f32,
    tongue_visible: f32,
    body_lean_forward: f32,
    body_lean_right: f32,
    body_twist_left: f32,
    auricle_raise_left: f32,
    auricle_pull_back_left: f32,
    auricle_flatten_left: f32,
    auricle_wiggle_amplitude_left: f32,
}

#[test]
fn fixed_golden_vector_is_backend_independent() {
    let mut input = frame(3, 11, 1_000);
    for (id, value) in [
        (7, -0.8),
        (8, 0.25),
        (11, -0.4),
        (12, 0.3),
        (17, 0.6),
        (18, 0.6),
        (20, 0.5),
        (21, -0.2),
        (22, 0.7),
        (23, -0.3),
        (28, 0.2),
        (29, 0.8),
        (30, 0.75),
        (41, 0.5),
        (45, 0.4),
        (46, -0.3),
        (47, 0.2),
        (57, 0.3),
        (59, -0.4),
        (61, 0.5),
    ] {
        set(&mut input, id, value, 0.9);
    }
    let first = SemanticDeriver::default().derive(&input, 1_100).unwrap();
    let second = SemanticDeriver::default().derive(&input, 1_100).unwrap();
    assert_eq!(first, second);

    let expected: GoldenExpected =
        serde_json::from_str(include_str!("vectors/semantic-golden-v1.json")).unwrap();
    for (id, expected) in [
        (SemanticId::EyeBlink(Side::Left), expected.eye_blink_left),
        (SemanticId::EyeWide(Side::Right), expected.eye_wide_right),
        (SemanticId::CheekSuck(Side::Left), expected.cheek_suck_left),
        (
            SemanticId::CheekPuff(Side::Right),
            expected.cheek_puff_right,
        ),
        (SemanticId::JawRight, expected.jaw_right),
        (
            SemanticId::MouthSmile(Side::Left),
            expected.mouth_smile_left,
        ),
        (
            SemanticId::MouthFrown(Side::Right),
            expected.mouth_frown_right,
        ),
        (SemanticId::MouthPucker, expected.mouth_pucker),
        (SemanticId::MouthFunnel, expected.mouth_funnel),
        (
            SemanticId::MouthInteriorVisible,
            expected.mouth_interior_visible,
        ),
        (SemanticId::TongueVisible, expected.tongue_visible),
        (SemanticId::BodyLeanForward, expected.body_lean_forward),
        (SemanticId::BodyLeanRight, expected.body_lean_right),
        (SemanticId::BodyTwistLeft, expected.body_twist_left),
        (
            SemanticId::AuricleRaise(Side::Left),
            expected.auricle_raise_left,
        ),
        (
            SemanticId::AuriclePullBack(Side::Left),
            expected.auricle_pull_back_left,
        ),
        (
            SemanticId::AuricleFlatten(Side::Left),
            expected.auricle_flatten_left,
        ),
        (
            SemanticId::AuricleWiggleAmplitude(Side::Left),
            expected.auricle_wiggle_amplitude_left,
        ),
    ] {
        assert_close(value(&first, id), expected);
    }
}

#[test]
fn capture_time_history_degrades_confidence_and_resets_by_generation() {
    let mut deriver = SemanticDeriver::default();
    let mut observed = frame(1, 1, 100);
    set(&mut observed, 7, -0.8, 1.0);
    deriver.derive(&observed, 100).unwrap();

    let mut occluded = frame(1, 2, 200);
    occluded.rig.set(
        nana_tracking_semantics::signal_id(7),
        SignalSample::unavailable(0.5, SignalState::Occluded, 200),
    );
    let held = deriver.derive(&occluded, 300).unwrap();
    let blink = held.get(SemanticId::EyeBlink(Side::Left)).unwrap();
    assert_close(blink.value, 0.8);
    assert_close(blink.confidence, 0.175);
    assert_eq!(blink.state, SignalState::Occluded);
    assert_eq!(blink.sample_age_ns, 200);

    let mut reset = frame(2, 1, 50);
    reset.rig.set(
        nana_tracking_semantics::signal_id(7),
        SignalSample::unavailable(0.5, SignalState::Occluded, 50),
    );
    let reset_semantics = deriver.derive(&reset, 60).unwrap();
    assert!(
        reset_semantics
            .get(SemanticId::EyeBlink(Side::Left))
            .is_none()
    );

    let older = frame(2, 2, 40);
    assert_eq!(
        deriver.derive(&older, 60),
        Err(SemanticError::OutOfOrderCaptureTimestamp {
            previous: 50,
            actual: 40,
        })
    );
}

#[test]
fn predicted_values_are_not_refiltered_but_confidence_is_reduced() {
    let mut input = frame(0, 1, 100);
    input.rig.set(
        nana_tracking_semantics::signal_id(7),
        SignalSample::available(-0.75, 0.8, SignalState::Predicted, 100, 20),
    );
    let semantics = SemanticDeriver::default().derive(&input, 120).unwrap();
    let blink = semantics.get(SemanticId::EyeBlink(Side::Left)).unwrap();
    assert_close(blink.value, 0.75);
    assert_close(blink.confidence, 0.52);
    assert_eq!(blink.state, SignalState::Predicted);
}

#[test]
fn skeleton_authoritatively_drives_arm_relations() {
    let mut input = frame(0, 1, 1_000);
    set(&mut input, 67, 0.0, 0.8);
    set(&mut input, 68, 0.0, 0.8);
    set(&mut input, 70, 0.25, 0.8);
    let pose = |position| {
        Tracked::available(
            Pose {
                parent_space: CoordinateSpace::TorsoLocal,
                length_basis: LengthBasis::TorsoRelative,
                position,
                orientation_xyzw: Quaternion::IDENTITY,
            },
            0.9,
            SignalState::Observed,
            1_000,
            0,
        )
    };
    input.skeleton.shoulder.left = pose(Vec3 {
        x: -0.2,
        y: 0.0,
        z: 0.0,
    });
    input.skeleton.wrist.left = pose(Vec3 {
        x: 0.3,
        y: -0.82,
        z: 0.45,
    });
    input.skeleton.upper_arm_direction_torso.left = Tracked::available(
        Direction3 {
            space: CoordinateSpace::TorsoLocal,
            value: Vec3 {
                x: -0.6,
                y: -0.8,
                z: 0.0,
            },
        },
        0.9,
        SignalState::Observed,
        1_000,
        0,
    );
    let semantics = SemanticDeriver::default().derive(&input, 1_010).unwrap();
    assert_close(value(&semantics, SemanticId::ArmRaise(Side::Left)), 1.0);
    assert_close(
        value(&semantics, SemanticId::ArmSideWeight(Side::Left)),
        1.0,
    );
    assert_close(value(&semantics, SemanticId::ArmCrossBody(Side::Left)), 1.0);
    assert_close(
        value(&semantics, SemanticId::ArmReachForward(Side::Left)),
        1.0,
    );
    assert_close(
        value(&semantics, SemanticId::HandAboveHead(Side::Left)),
        1.0,
    );
    assert_close(value(&semantics, SemanticId::ArmBend(Side::Left)), 0.25);
    assert_close(
        value(&semantics, SemanticId::ArmExtension(Side::Left)),
        0.75,
    );
}

#[test]
fn auricle_velocity_uses_capture_interval_not_receive_interval() {
    fn velocity(second_timestamp: u64) -> f32 {
        let mut deriver = SemanticDeriver::default();
        let mut first = frame(0, 1, 0);
        for id in [57, 59, 61] {
            set(&mut first, id, 0.0, 1.0);
        }
        deriver.derive(&first, 99_000_000_000).unwrap();
        let mut second = frame(0, 2, second_timestamp);
        set(&mut second, 57, 0.4, 1.0);
        set(&mut second, 59, 0.0, 1.0);
        set(&mut second, 61, 0.0, 1.0);
        let result = deriver.derive(&second, 99_000_000_001).unwrap();
        value(&result, SemanticId::AuricleWiggleVelocity(Side::Left))
    }
    assert_close(velocity(100_000_000), 1.0);
    assert_close(velocity(1_000_000_000), 0.1);
}

#[test]
fn arkit_profile_binds_all_52_targets_without_protocol_aliases() {
    let mut input = frame(0, 1, 100);
    for id in 1..=41 {
        set(&mut input, id, 0.0, 1.0);
    }
    set(&mut input, 7, -0.8, 1.0);
    set(&mut input, 18, 0.6, 1.0);
    set(&mut input, 20, 0.5, 1.0);
    let semantics = SemanticDeriver::default().derive(&input, 100).unwrap();
    let evaluator = BindingEvaluator::new(arkit_style_52_profile()).unwrap();
    let output = evaluator.evaluate(&input, &semantics).unwrap();
    assert_eq!(output.len(), 52);
    assert_close(
        output[&nana_tracking_semantics::ModelParameterId::new("eyeBlinkLeft").unwrap()].value,
        0.8,
    );
    assert_close(
        output[&nana_tracking_semantics::ModelParameterId::new("jawRight").unwrap()].value,
        0.6,
    );
    assert_close(
        output[&nana_tracking_semantics::ModelParameterId::new("mouthSmileLeft").unwrap()].value,
        0.5,
    );
}

#[test]
fn low_dof_profile_merges_bilateral_values_and_conflicts_are_rejected() {
    let mut input = frame(0, 1, 100);
    set(&mut input, 20, 0.8, 0.9);
    set(&mut input, 21, 0.2, 0.7);
    let semantics = SemanticDeriver::default().derive(&input, 100).unwrap();
    let evaluator = BindingEvaluator::new(live2d_common_profile()).unwrap();
    let output = evaluator.evaluate(&input, &semantics).unwrap();
    let mouth = nana_tracking_semantics::ModelParameterId::new("ParamMouthForm").unwrap();
    assert_close(output[&mouth].value, 0.5);
    assert_close(output[&mouth].confidence, 0.7);

    let mut conflict = live2d_common_profile();
    let mut duplicate = conflict.bindings[0].clone();
    duplicate.layer = BindingLayer::Orthogonal;
    conflict.bindings.push(duplicate);
    assert!(matches!(
        BindingEvaluator::new(conflict),
        Err(BindingError::LayerConflict(_))
    ));
}

#[test]
fn declarative_transforms_and_multi_binding_combine_are_executable() {
    let mut input = frame(0, 1, 100);
    set(&mut input, 1, 0.6, 0.9);
    set(&mut input, 2, 0.2, 0.7);
    let transformed = ModelParameterId::new("transformed").unwrap();
    let combined = ModelParameterId::new("combined").unwrap();
    let piecewise = ModelParameterId::new("piecewise").unwrap();
    let binding = |source, target: &ModelParameterId, transform, combine| LayeredBinding {
        layer: BindingLayer::Compatibility,
        binding: RigBinding {
            source,
            target: target.clone(),
            transform,
            combine,
        },
    };
    let profile = BindingProfile {
        name: "functional transform fixture".into(),
        requirements: SignalRequirements::default(),
        bindings: vec![
            binding(
                SignalExpression::Ntp(nana_tracking_semantics::signal_id(1)),
                &transformed,
                BindingTransform {
                    curve: BindingCurve::Power(2.0),
                    deadzone: 0.2,
                    clamp_min: 0.3,
                    clamp_max: 1.0,
                    scale: 2.0,
                    offset: 0.75,
                    invert: true,
                },
                CombineMode::Replace,
            ),
            binding(
                SignalExpression::Ntp(nana_tracking_semantics::signal_id(1)),
                &piecewise,
                BindingTransform {
                    curve: BindingCurve::PiecewiseLinear(vec![
                        CurvePoint {
                            input: 0.0,
                            output: 0.0,
                        },
                        CurvePoint {
                            input: 0.5,
                            output: 1.0,
                        },
                        CurvePoint {
                            input: 1.0,
                            output: 1.0,
                        },
                    ]),
                    ..BindingTransform::default()
                },
                CombineMode::Replace,
            ),
            binding(
                SignalExpression::Ntp(nana_tracking_semantics::signal_id(1)),
                &combined,
                BindingTransform::default(),
                CombineMode::Average,
            ),
            binding(
                SignalExpression::Ntp(nana_tracking_semantics::signal_id(2)),
                &combined,
                BindingTransform::default(),
                CombineMode::Average,
            ),
        ],
    };
    let semantics = SemanticDeriver::default().derive(&input, 100).unwrap();
    let output = BindingEvaluator::new(profile)
        .unwrap()
        .evaluate(&input, &semantics)
        .unwrap();
    assert_close(output[&transformed].value, 0.3);
    assert_close(output[&piecewise].value, 1.0);
    assert_close(output[&combined].value, 0.4);
    assert_close(output[&combined].confidence, 0.7);
}

#[test]
fn evaluator_rejects_mixed_ntp_and_semantic_frames() {
    let first = frame(0, 1, 100);
    let second = frame(0, 2, 200);
    let semantics = SemanticDeriver::default().derive(&first, 100).unwrap();
    let evaluator = BindingEvaluator::new(nana_native_profile()).unwrap();
    assert_eq!(
        evaluator.evaluate(&second, &semantics),
        Err(BindingError::FrameMismatch)
    );
}

#[test]
fn required_and_preferred_signals_resolve_against_capabilities() {
    let descriptor = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(36),
        StructureFeatures::HEAD_GEOMETRY,
        TrackingFeatures::empty(),
    );
    let arkit = arkit_style_52_profile().requirements.resolve(&descriptor);
    assert!(arkit.is_compatible());
    assert_eq!(arkit.missing_preferred.len(), 5);

    let native = nana_native_profile().requirements.resolve(&descriptor);
    assert!(native.is_compatible());
    assert_eq!(native.missing_preferred.len(), 52);
}

#[test]
fn raw_native_bindings_preserve_explicit_receiver_sample_age() {
    let mut input = frame(0, 1, 100);
    set(&mut input, 1, -0.4, 0.8);
    let semantics = SemanticDeriver::default().derive(&input, 250).unwrap();
    let evaluator = BindingEvaluator::new(nana_native_profile()).unwrap();
    let output = evaluator.evaluate(&input, &semantics).unwrap();
    let target =
        nana_tracking_semantics::ModelParameterId::new("brow.left.inner_vertical").unwrap();
    assert_close(output[&target].value, -0.4);
    assert_eq!(output[&target].sample_age_ns, 150);
}

#[test]
fn direction_based_bend_is_deterministic() {
    let mut input = frame(0, 1, 100);
    let direction = |value| {
        Tracked::available(
            Direction3 {
                space: CoordinateSpace::TorsoLocal,
                value,
            },
            1.0,
            SignalState::Observed,
            100,
            0,
        )
    };
    input.skeleton.upper_arm_direction_torso.right = direction(Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    });
    input.skeleton.forearm_direction_torso.right = direction(Vec3 {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    });
    let semantics = SemanticDeriver::default().derive(&input, 100).unwrap();
    assert_close(value(&semantics, SemanticId::ArmBend(Side::Right)), 1.0);
    assert_close(
        value(&semantics, SemanticId::ArmExtension(Side::Right)),
        0.0,
    );
}
