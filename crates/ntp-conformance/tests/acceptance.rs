mod support;

use nana_tracking_protocol::{SignalId, SignalState, StructureFeatures, TrackingProfile};
use nana_tracking_semantics::{BindingEvaluator, SemanticDeriver, arkit_style_52_profile};
use ntp_conformance::{ConformanceOptions, validate_stream};
use support::{descriptor, frame, set, set_arm_out_of_frame};

#[test]
fn basic_producer_keeps_extra_spatial_capability() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[37]);
    assert_eq!(descriptor.guaranteed_profile, TrackingProfile::Basic);
    let mut result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    set(&mut result, 37, 0.2);
    let report = validate_stream(&descriptor, &[result], ConformanceOptions::default());
    assert!(report.passed, "{report}");
    assert_eq!(report.certified_profile, Some(TrackingProfile::Basic));
    assert!(
        descriptor
            .supported_signals
            .contains(SignalId::new(37).unwrap())
    );
}

#[test]
fn spatial_producer_keeps_extra_full_signals() {
    let descriptor = descriptor(41, StructureFeatures::SPATIAL_REQUIRED, &[57, 58]);
    assert_eq!(descriptor.guaranteed_profile, TrackingProfile::Spatial);
    let mut result = frame(&descriptor, 2, 0, 1, 1_000_000_000);
    set(&mut result, 57, 0.4);
    set(&mut result, 58, -0.2);
    let report = validate_stream(&descriptor, &[result], ConformanceOptions::default());
    assert!(report.passed, "{report}");
    assert_eq!(report.certified_profile, Some(TrackingProfile::Spatial));
}

#[test]
fn full_profile_survives_arms_out_of_frame() {
    let descriptor = descriptor(76, StructureFeatures::FULL_REQUIRED, &[]);
    let mut result = frame(&descriptor, 3, 0, 1, 1_000_000_000);
    set_arm_out_of_frame(&mut result);
    let report = validate_stream(
        &descriptor,
        &[result.clone()],
        ConformanceOptions::default(),
    );
    assert!(report.passed, "{report}");
    assert_eq!(report.certified_profile, Some(TrackingProfile::Full));
    assert_eq!(
        result.rig.get(SignalId::new(67).unwrap()).unwrap().state,
        SignalState::OutOfFrame
    );
}

#[test]
fn normalized_inputs_from_different_producers_reach_identical_model_controls() {
    let basic_descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[37, 38, 39, 40, 41]);
    let spatial_descriptor = descriptor(41, StructureFeatures::SPATIAL_REQUIRED, &[57]);
    let mut basic = frame(&basic_descriptor, 4, 0, 1, 1_000_000_000);
    let mut spatial = frame(&spatial_descriptor, 5, 0, 1, 1_000_000_000);
    for (id, value) in [
        (1, 0.25),
        (2, -0.1),
        (7, -0.8),
        (8, 0.3),
        (17, 0.5),
        (18, -0.4),
        (20, 0.7),
        (21, -0.2),
        (28, 0.1),
        (29, 0.8),
        (30, 0.25),
    ] {
        set(&mut basic, id, value);
        set(&mut spatial, id, value);
    }
    let basic_semantics = SemanticDeriver::default()
        .derive(&basic, basic.produced_timestamp_ns)
        .unwrap();
    let spatial_semantics = SemanticDeriver::default()
        .derive(&spatial, spatial.produced_timestamp_ns)
        .unwrap();
    let evaluator = BindingEvaluator::new(arkit_style_52_profile()).unwrap();
    let basic_controls = evaluator.evaluate(&basic, &basic_semantics).unwrap();
    let spatial_controls = evaluator.evaluate(&spatial, &spatial_semantics).unwrap();
    assert_eq!(basic_controls.len(), spatial_controls.len());
    for (id, basic_value) in &basic_controls {
        let spatial_value = &spatial_controls[id];
        assert!(
            (basic_value.value - spatial_value.value).abs() < 1.0e-6,
            "{id:?}"
        );
        assert!(
            (basic_value.confidence - spatial_value.confidence).abs() < 1.0e-6,
            "{id:?}"
        );
        assert_eq!(basic_value.state, spatial_value.state, "{id:?}");
    }
}
