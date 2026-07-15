mod support;

use nana_tracking_protocol::{SignalId, SignalSample, SignalState, StructureFeatures};
use ntp_conformance::{ConformanceOptions, ConformanceValidator, FailureCode, validate_stream};
use support::{descriptor, frame, set};

#[test]
fn supported_capability_cannot_masquerade_as_unsupported_zero() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let mut result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    result
        .rig
        .set(SignalId::new(7).unwrap(), SignalSample::unsupported());
    let report = validate_stream(&descriptor, &[result], ConformanceOptions::default());
    assert!(!report.passed);
    assert!(report.findings.iter().any(|finding| {
        finding.code == FailureCode::CapabilityStateMismatch && finding.signal_id == Some(7)
    }));
}

#[test]
fn rejected_late_generation_does_not_poison_active_history() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let mut validator =
        ConformanceValidator::new(descriptor.clone(), ConformanceOptions::default());
    validator.validate_frame(&frame(&descriptor, 1, 0, 10, 1_000_000_000));
    validator.validate_frame(&frame(&descriptor, 1, 1, 1, 2_000_000_000));
    validator.validate_frame(&frame(&descriptor, 1, 0, 99, 3_000_000_000));
    validator.validate_frame(&frame(&descriptor, 1, 1, 2, 2_100_000_000));
    let report = validator.finish();
    let stale = report
        .findings
        .iter()
        .filter(|finding| finding.code == FailureCode::StaleGeneration)
        .count();
    assert_eq!(stale, 1, "{report}");
    assert!(
        !report
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::SequenceNotMonotonic)
    );
}

#[test]
fn duplicate_and_backward_capture_are_rejected() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let frames = [
        frame(&descriptor, 1, 0, 1, 1_000_000_000),
        frame(&descriptor, 1, 0, 1, 1_100_000_000),
        frame(&descriptor, 1, 0, 2, 900_000_000),
    ];
    let report = validate_stream(&descriptor, &frames, ConformanceOptions::default());
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::SequenceNotMonotonic)
    );
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::CaptureTimestampNotMonotonic)
    );
}

#[test]
fn prediction_confidence_must_fall_for_the_same_source_sample() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let source_timestamp = 1_000_000_000;
    let mut first = frame(&descriptor, 1, 0, 1, 1_100_000_000);
    first.rig.set(
        SignalId::new(7).unwrap(),
        SignalSample::available(
            -0.5,
            0.5,
            SignalState::Predicted,
            source_timestamp,
            20_000_000,
        ),
    );
    let mut second = frame(&descriptor, 1, 0, 2, 1_200_000_000);
    second.rig.set(
        SignalId::new(7).unwrap(),
        SignalSample::available(
            -0.5,
            0.6,
            SignalState::Predicted,
            source_timestamp,
            40_000_000,
        ),
    );
    let report = validate_stream(&descriptor, &[first, second], ConformanceOptions::default());
    assert!(report.findings.iter().any(|finding| {
        finding.code == FailureCode::PredictionConfidenceIncreased && finding.signal_id == Some(7)
    }));
}

#[test]
fn skeleton_and_scalar_views_must_describe_one_arm_state() {
    let descriptor = descriptor(76, StructureFeatures::FULL_REQUIRED, &[]);
    let mut result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    set(&mut result, 67, 0.8);
    let report = validate_stream(&descriptor, &[result], ConformanceOptions::default());
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::SkeletonScalarMismatch)
    );
}

#[test]
fn zero_length_bones_are_invalid_even_with_finite_coordinates() {
    let descriptor = descriptor(76, StructureFeatures::FULL_REQUIRED, &[]);
    let mut result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    result.skeleton.elbow.left.value.as_mut().unwrap().position = result
        .skeleton
        .shoulder
        .left
        .value
        .as_ref()
        .unwrap()
        .position;
    let report = validate_stream(&descriptor, &[result], ConformanceOptions::default());
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::InvalidBoneLength)
    );
}

#[test]
fn product_time_limits_are_explicit_policy_not_hidden_protocol_defaults() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    let protocol_only = validate_stream(
        &descriptor,
        std::slice::from_ref(&result),
        ConformanceOptions::default(),
    );
    assert!(protocol_only.passed, "{protocol_only}");

    let options = ConformanceOptions {
        max_capture_to_result_ns: 500_000,
        ..ConformanceOptions::default()
    };
    let policy = validate_stream(&descriptor, &[result], options);
    assert!(
        policy
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::CaptureToResultExceeded)
    );
}

#[test]
fn invalid_validator_tolerances_fail_closed() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let options = ConformanceOptions {
        semantic_tolerance: f32::NAN,
        ..ConformanceOptions::default()
    };
    let report = validate_stream(
        &descriptor,
        &[frame(&descriptor, 1, 0, 1, 1_000_000_000)],
        options,
    );
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::InvalidConfiguration)
    );
}
