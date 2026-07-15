use nana_tracking_protocol::{
    ContractError, CoordinateSpace, LengthBasis, NanaTrackingDescriptor, NanaTrackingResult, Pose,
    Quaternion, ResultStreamGuard, STABLE_SIGNAL_COUNT, SessionId, SignalBitSet, SignalId,
    SignalMetadata, SignalSample, SignalState, StreamError, StructureFeatures, Tracked,
    TrackingFeatures, TrackingProfile, Validate, Vec3,
};

fn session(value: u8) -> SessionId {
    SessionId([value; 16])
}

fn observed(value: f32, timestamp: u64) -> SignalSample {
    SignalSample::available(value, 0.9, SignalState::Observed, timestamp, 0)
}

fn head_pose(timestamp: u64) -> Tracked<Pose> {
    Tracked::available(
        Pose {
            parent_space: CoordinateSpace::Camera,
            length_basis: LengthBasis::HeadRelative,
            position: Vec3::default(),
            orientation_xyzw: Quaternion::IDENTITY,
        },
        0.9,
        SignalState::Observed,
        timestamp,
        0,
    )
}

fn basic_descriptor() -> NanaTrackingDescriptor {
    NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(36),
        StructureFeatures::HEAD_GEOMETRY,
        TrackingFeatures::empty(),
    )
}

fn basic_result(sequence: u64) -> NanaTrackingResult {
    let mut result = NanaTrackingResult::unsupported(session(1), 0, sequence, 100, 110);
    for raw in 1..=36 {
        result
            .rig
            .set(SignalId::new(raw).unwrap(), observed(0.0, 100));
    }
    result.geometry.head_camera_pose = head_pose(100);
    result.quality.face = nana_tracking_protocol::RegionQuality {
        confidence: 0.9,
        state: SignalState::Observed,
    };
    result.quality.eyes = nana_tracking_protocol::RegionQuality {
        confidence: 0.9,
        state: SignalState::Observed,
    };
    result
}

#[test]
fn registry_has_fixed_unique_contiguous_slots_and_normative_types() {
    let metadata = SignalMetadata::all().collect::<Vec<_>>();
    assert_eq!(metadata.len(), STABLE_SIGNAL_COUNT);
    for (slot, signal) in metadata.iter().enumerate() {
        assert_eq!(signal.id.stable_slot(), Some(slot));
        assert_eq!(SignalId::from_stable_slot(slot), Some(signal.id));
    }
    assert_eq!(metadata[0].stable_name, "brow.left.inner_vertical");
    assert_eq!(metadata[87].stable_name, "wrist.right.twist");
    assert!(
        SignalMetadata::get(SignalId::new(17).unwrap())
            .unwrap()
            .scalar_type
            .contains(1.0)
    );
    assert!(
        !SignalMetadata::get(SignalId::new(17).unwrap())
            .unwrap()
            .scalar_type
            .contains(-0.01)
    );
}

#[test]
fn profile_is_derived_independently_from_extra_capabilities() {
    let basic = basic_descriptor();
    assert_eq!(basic.guaranteed_profile, TrackingProfile::Basic);

    let mut signals = SignalBitSet::stable_through(36);
    signals.insert(SignalId::new(37).unwrap());
    let basic_plus_gaze = NanaTrackingDescriptor::from_capabilities(
        signals,
        StructureFeatures::HEAD_GEOMETRY,
        TrackingFeatures::empty(),
    );
    assert_eq!(basic_plus_gaze.guaranteed_profile, TrackingProfile::Basic);
    assert!(
        basic_plus_gaze
            .supported_signals
            .contains(SignalId::new(37).unwrap())
    );

    let spatial = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(41),
        StructureFeatures::SPATIAL_REQUIRED,
        TrackingFeatures::METRIC_COORDINATES,
    );
    assert_eq!(spatial.guaranteed_profile, TrackingProfile::Spatial);

    let full = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(76),
        StructureFeatures::FULL_REQUIRED,
        TrackingFeatures::WRIST_POSE,
    );
    assert_eq!(full.guaranteed_profile, TrackingProfile::Full);
}

#[test]
fn descriptor_rejects_claimed_profile_and_experimental_ids() {
    let mut descriptor = basic_descriptor();
    descriptor.guaranteed_profile = TrackingProfile::Full;
    assert_eq!(descriptor.validate(), Err(ContractError::ProfileMismatch));

    let mut signals = SignalBitSet::stable_through(36);
    let experimental = SignalId::new(0x8000).unwrap();
    signals.insert(experimental);
    let descriptor = NanaTrackingDescriptor::from_capabilities(
        signals,
        StructureFeatures::HEAD_GEOMETRY,
        TrackingFeatures::empty(),
    );
    assert_eq!(
        descriptor.validate(),
        Err(ContractError::ExperimentalSignal(experimental))
    );
}

#[test]
fn feature_bits_require_their_semantic_structure_dependencies() {
    let descriptor = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::new(),
        StructureFeatures::empty(),
        TrackingFeatures::DENSE_FACE_MESH,
    );
    assert_eq!(
        descriptor.validate(),
        Err(ContractError::FeatureDependency(
            "dense_face_mesh requires face geometry"
        ))
    );

    let descriptor = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::new(),
        StructureFeatures::empty(),
        TrackingFeatures::WRIST_POSE,
    );
    assert_eq!(
        descriptor.validate(),
        Err(ContractError::FeatureDependency(
            "wrist_pose requires body skeleton"
        ))
    );
}

#[test]
fn fixed_slots_preserve_value_confidence_and_state_independently() {
    let mut result = NanaTrackingResult::unsupported(session(1), 0, 1, 100, 110);
    let id = SignalId::new(1).unwrap();
    result.rig.set(id, observed(0.25, 100));
    assert_eq!(result.rig.slots().len(), STABLE_SIGNAL_COUNT);
    assert_eq!(result.rig.get(id).unwrap().value, Some(0.25));
    assert_eq!(
        result.rig.get(SignalId::new(2).unwrap()).unwrap().state,
        SignalState::Unsupported
    );
    result.validate().unwrap();

    result.rig.set(
        id,
        SignalSample::available(1.1, 0.9, SignalState::Observed, 100, 0),
    );
    assert_eq!(
        result.validate(),
        Err(ContractError::InvalidSignalValue(id))
    );
}

#[test]
fn descriptor_and_frame_capabilities_must_agree_without_backend_branches() {
    let descriptor = basic_descriptor();
    let result = basic_result(1);
    descriptor.validate_result(&result).unwrap();

    let mut invalid = result.clone();
    invalid
        .rig
        .set(SignalId::new(36).unwrap(), SignalSample::unsupported());
    assert_eq!(
        descriptor.validate_result(&invalid),
        Err(ContractError::CapabilityMismatch(
            SignalId::new(36).unwrap()
        ))
    );

    let descriptor = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(36),
        StructureFeatures::HEAD_GEOMETRY | StructureFeatures::FACE_GEOMETRY,
        TrackingFeatures::empty(),
    );
    assert_eq!(
        descriptor.validate_result(&result),
        Err(ContractError::StructureCapabilityMismatch("face geometry"))
    );
}

#[test]
fn extra_capability_and_current_frame_state_remain_independent() {
    let mut signals = SignalBitSet::stable_through(36);
    let gaze = SignalId::new(37).unwrap();
    signals.insert(gaze);
    let descriptor = NanaTrackingDescriptor::from_capabilities(
        signals,
        StructureFeatures::HEAD_GEOMETRY,
        TrackingFeatures::empty(),
    );
    assert_eq!(descriptor.guaranteed_profile, TrackingProfile::Basic);

    let mut result = basic_result(1);
    result.rig.set(
        gaze,
        SignalSample::unavailable(0.2, SignalState::Occluded, 100),
    );
    descriptor.validate_result(&result).unwrap();
}

#[test]
fn current_registry_rejects_unassigned_landmark_indices() {
    let descriptor = basic_descriptor();
    let mut result = basic_result(1);
    result
        .geometry
        .face_landmarks
        .push(nana_tracking_protocol::FaceLandmark {
            semantic_id: 1,
            position_head: Tracked::available(
                nana_tracking_protocol::Position3 {
                    space: CoordinateSpace::HeadLocal,
                    length_basis: LengthBasis::HeadRelative,
                    value: Vec3::default(),
                },
                0.5,
                SignalState::Observed,
                100,
                0,
            ),
        });
    assert_eq!(
        descriptor.validate_result(&result),
        Err(ContractError::UnassignedLandmark(1))
    );
}

#[test]
fn stream_guard_handles_gaps_reconnects_switches_and_late_frames() {
    let mut guard = ResultStreamGuard::new(session(1), 4);
    let first = NanaTrackingResult::unsupported(session(1), 4, 10, 100, 100);
    assert_eq!(guard.accept(&first).unwrap().missing_sequences, 0);

    let after_gap = NanaTrackingResult::unsupported(session(1), 4, 13, 130, 130);
    assert_eq!(guard.accept(&after_gap).unwrap().missing_sequences, 2);
    assert!(matches!(
        guard.accept(&first),
        Err(StreamError::DuplicateOrOutOfOrder { .. })
    ));

    guard.advance_generation(5).unwrap();
    let switched = NanaTrackingResult::unsupported(session(1), 5, 1, 200, 200);
    guard.accept(&switched).unwrap();
    assert_eq!(
        guard.accept(&after_gap),
        Err(StreamError::WrongGeneration {
            expected: 5,
            actual: 4
        })
    );

    let reconnect = NanaTrackingResult::unsupported(session(2), 0, 1, 300, 300);
    assert_eq!(guard.accept(&reconnect), Err(StreamError::WrongSession));
    assert_eq!(
        guard.replace_session(session(1), 0),
        Err(StreamError::SessionDidNotChange)
    );
    guard.replace_session(session(2), 0).unwrap();
    guard.accept(&reconnect).unwrap();
}

#[test]
fn invalid_frame_never_advances_stream_state() {
    let mut guard = ResultStreamGuard::new(session(1), 0);
    let invalid = NanaTrackingResult::unsupported(session(1), 0, 1, 200, 100);
    assert!(matches!(
        guard.accept(&invalid),
        Err(StreamError::InvalidContract(
            ContractError::InvalidTimestamp("produced before capture")
        ))
    ));
    let valid = NanaTrackingResult::unsupported(session(1), 0, 1, 200, 200);
    guard.accept(&valid).unwrap();
}

#[test]
fn sample_age_and_processing_latency_use_capture_time() {
    let result = NanaTrackingResult::unsupported(session(1), 0, 1, 100, 125);
    assert_eq!(result.sample_age_ns(160), 60);
    assert_eq!(result.sample_age_ns(90), 0);
    assert_eq!(result.processing_latency_ns(), 25);
}
