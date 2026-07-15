use nana_tracking_fusion::{
    FusionError, SpatialFusionPolicy, fuse_spatial_results, fused_descriptor,
};
use nana_tracking_protocol::{
    CoordinateSpace, Direction3, LengthBasis, NanaTrackingDescriptor, NanaTrackingResult,
    Position3, RegionQuality, SessionId, SignalBitSet, SignalId, SignalSample, SignalState,
    StructureFeatures, Tracked, TrackingFeatures, TrackingProfile, Vec3,
};
use ntp_conformance::{ConformanceOptions, validate_stream};

fn descriptor(last: u16, structures: StructureFeatures) -> NanaTrackingDescriptor {
    NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(last),
        structures,
        TrackingFeatures::empty(),
    )
}

fn observed_quality() -> RegionQuality {
    RegionQuality {
        confidence: 0.9,
        state: SignalState::Observed,
    }
}

fn fill_head(result: &mut NanaTrackingResult, timestamp: u64) {
    result.geometry.head_camera_pose = Tracked::available(
        nana_tracking_protocol::Pose {
            parent_space: CoordinateSpace::Camera,
            length_basis: LengthBasis::HeadRelative,
            position: Vec3::default(),
            orientation_xyzw: nana_tracking_protocol::Quaternion::IDENTITY,
        },
        0.9,
        SignalState::Observed,
        timestamp,
        0,
    );
    result.quality.face = observed_quality();
}

fn fill_spatial_geometry(result: &mut NanaTrackingResult, timestamp: u64) {
    for eye in [
        &mut result.geometry.eyes.left,
        &mut result.geometry.eyes.right,
    ] {
        eye.origin_head = Tracked::available(
            Position3 {
                space: CoordinateSpace::HeadLocal,
                length_basis: LengthBasis::HeadRelative,
                value: Vec3::default(),
            },
            0.9,
            SignalState::Observed,
            timestamp,
            0,
        );
        eye.direction_head = Tracked::available(
            Direction3 {
                space: CoordinateSpace::HeadLocal,
                value: Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            },
            0.9,
            SignalState::Observed,
            timestamp,
            0,
        );
    }
    result.geometry.look_at_camera = Tracked::available(
        Position3 {
            space: CoordinateSpace::Camera,
            length_basis: LengthBasis::HeadRelative,
            value: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
        },
        0.9,
        SignalState::Observed,
        timestamp,
        0,
    );
    result.geometry.face_geometry_state = SignalState::Observed;
    result.quality.eyes = observed_quality();
}

fn frame(descriptor: &NanaTrackingDescriptor, timestamp: u64) -> NanaTrackingResult {
    let mut result =
        NanaTrackingResult::unsupported(SessionId([7; 16]), 2, 11, timestamp, timestamp + 1);
    for id in descriptor.supported_signals.iter() {
        result.rig.set(
            id,
            SignalSample::available(0.0, 0.8, SignalState::Observed, timestamp, 0),
        );
    }
    if descriptor
        .supported_structures
        .contains(StructureFeatures::HEAD_GEOMETRY)
    {
        fill_head(&mut result, timestamp);
        if descriptor
            .supported_signals
            .contains(SignalId::new(7).unwrap())
        {
            result.quality.eyes = observed_quality();
        }
    }
    if descriptor
        .supported_structures
        .contains(StructureFeatures::EYE_GEOMETRY)
    {
        fill_spatial_geometry(&mut result, timestamp);
    }
    if descriptor
        .supported_signals
        .contains(SignalId::new(57).unwrap())
    {
        result.quality.auricle.left = observed_quality();
    }
    result.quality.overall_confidence = 0.85;
    result
}

#[test]
fn descriptor_union_reaches_spatial_and_preserves_full_extras() {
    let reference = descriptor(41, StructureFeatures::SPATIAL_REQUIRED);
    let mut extension = descriptor(36, StructureFeatures::BASIC_REQUIRED);
    extension
        .supported_signals
        .insert(SignalId::new(57).unwrap());
    extension.guaranteed_profile = TrackingProfile::Basic;
    let fused = fused_descriptor(&reference, &extension).unwrap();
    assert_eq!(fused.guaranteed_profile, TrackingProfile::Spatial);
    assert!(fused.supported_signals.contains(SignalId::new(57).unwrap()));

    let reference_frame = frame(&reference, 1_000);
    let mut extension_frame = frame(&extension, 1_000);
    extension_frame.rig.set(
        SignalId::new(57).unwrap(),
        SignalSample::available(0.4, 0.9, SignalState::Observed, 1_000, 0),
    );
    let (fused, fused_frame) = fuse_spatial_results(
        &reference,
        &reference_frame,
        &extension,
        &extension_frame,
        SpatialFusionPolicy::default(),
    )
    .unwrap();
    assert_eq!(
        fused_frame
            .rig
            .get(SignalId::new(57).unwrap())
            .unwrap()
            .value,
        Some(0.4)
    );
    let report = validate_stream(&fused, &[fused_frame], ConformanceOptions::default());
    assert!(report.passed, "{:?}", report.findings);
    assert_eq!(report.certified_profile, Some(TrackingProfile::Spatial));
}

#[test]
fn gaze_uses_reference_while_rgb_can_fill_visible_tongue() {
    let spatial = descriptor(41, StructureFeatures::SPATIAL_REQUIRED);
    let mut reference = frame(&spatial, 1_000);
    let mut extension = frame(&spatial, 1_000);
    reference.rig.set(
        SignalId::new(37).unwrap(),
        SignalSample::available(0.25, 0.6, SignalState::Observed, 1_000, 0),
    );
    extension.rig.set(
        SignalId::new(37).unwrap(),
        SignalSample::available(-0.4, 0.99, SignalState::Observed, 1_000, 0),
    );
    reference.rig.set(
        SignalId::new(41).unwrap(),
        SignalSample::unavailable(0.4, SignalState::Occluded, 1_000),
    );
    extension.rig.set(
        SignalId::new(41).unwrap(),
        SignalSample::available(0.7, 0.9, SignalState::Observed, 1_000, 0),
    );
    let (_, fused) = fuse_spatial_results(
        &spatial,
        &reference,
        &spatial,
        &extension,
        SpatialFusionPolicy::default(),
    )
    .unwrap();
    assert_eq!(
        fused.rig.get(SignalId::new(37).unwrap()).unwrap().value,
        Some(0.25)
    );
    assert_eq!(
        fused.rig.get(SignalId::new(41).unwrap()).unwrap().value,
        Some(0.7)
    );
    assert_eq!(
        fused.geometry.head_camera_pose.value,
        reference.geometry.head_camera_pose.value
    );
}

#[test]
fn arrival_order_cannot_join_different_captures() {
    let spatial = descriptor(41, StructureFeatures::SPATIAL_REQUIRED);
    let reference = frame(&spatial, 1_000);
    let extension = frame(&spatial, 1_001);
    assert_eq!(
        fuse_spatial_results(
            &spatial,
            &reference,
            &spatial,
            &extension,
            SpatialFusionPolicy::default(),
        ),
        Err(FusionError::CaptureTimestampMismatch)
    );
}
