use nana_tracking_runtime_api::{
    ActiveProvider, MAX_STABLE_SIGNALS, ModelScalar, ModelTrackingState, ModelVector3,
    TrackingModelInput, TrackingModelOutput, TrackingRuntimeError,
};

#[test]
fn borrowed_rgb_contract_rejects_truncation_without_backend_types() {
    let rgb = [0_u8; 17];
    let input = TrackingModelInput {
        rgb: &rgb,
        width: 2,
        height: 3,
        row_stride: 6,
        capture_timestamp_ns: 10,
        generation: 0,
    };
    assert!(matches!(
        input.validate(),
        Err(TrackingRuntimeError::InvalidInput)
    ));
}

#[test]
fn output_storage_is_fixed_and_reusable() {
    let mut output = TrackingModelOutput::preallocated(ActiveProvider::OnnxRuntimeCpu);
    assert_eq!(output.signals.len(), MAX_STABLE_SIGNALS);
    output.signals[0] = Some(ModelScalar::observed(0.5, 0.9, 100));
    output.geometry.upper_body_joint_positions[0] =
        nana_tracking_runtime_api::ModelTracked::observed(ModelVector3::default(), 0.8, 100);
    output.clear();
    assert!(output.signals.iter().all(Option::is_none));
    assert!(
        output
            .geometry
            .upper_body_joint_positions
            .iter()
            .all(|joint| joint.state == ModelTrackingState::Unsupported && joint.value.is_none())
    );
    assert_eq!(
        ModelScalar::default().state,
        ModelTrackingState::Unsupported
    );
    assert_eq!(ModelScalar::default().value, None);
}
