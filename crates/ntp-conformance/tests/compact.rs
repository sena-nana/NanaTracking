use nana_tracking_protocol::{
    ActiveLayout, CompactFrameCodec, CompactFrameError, CompactFrameInput, CompactRecorder,
    CompactRecordingSink, CompactSample, CompactStreamError, CompactStreamGuard,
    CompactStreamPolicy, ContractRevisions, HandshakeError, HandshakeLimits, LatestFrame,
    LayoutError, LayoutLimits, LayoutNegotiator, LayoutProposal, LayoutRecord,
    NanaTrackingDescriptor, QualityEncoding, RecordingError, ScalarType, SessionId, SignalBitSet,
    SignalId, SignalMetadata, SignalState, StructureFeatures, TrackingFeatures, TrackingProfile,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct CompactLayoutVector {
    schema: String,
    layout_id: u32,
    target_fps: u16,
    extra_signal_ids: Vec<u16>,
    ordered_signal_ids: Vec<u16>,
    parameter_count: u16,
    expected_payload_len: u32,
    layout_hash_hex: String,
}

#[derive(Default)]
struct MemoryRecording {
    layouts: Vec<LayoutRecord>,
    frames: Vec<Vec<u8>>,
}

impl CompactRecordingSink for MemoryRecording {
    type Error = ();

    fn record_layout(&mut self, layout: &LayoutRecord) -> Result<(), Self::Error> {
        self.layouts.push(layout.clone());
        Ok(())
    }

    fn record_frame(&mut self, compact_frame: &[u8]) -> Result<(), Self::Error> {
        self.frames.push(compact_frame.to_vec());
        Ok(())
    }
}

fn session(value: u8) -> SessionId {
    SessionId([value; 16])
}

fn basic_descriptor(last_signal: u16) -> NanaTrackingDescriptor {
    NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(last_signal),
        StructureFeatures::HEAD_GEOMETRY,
        TrackingFeatures::empty(),
    )
}

fn full_descriptor() -> NanaTrackingDescriptor {
    NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(88),
        StructureFeatures::FULL_REQUIRED,
        TrackingFeatures::empty(),
    )
}

fn basic_layout(layout_id: u32) -> ActiveLayout {
    let descriptor = basic_descriptor(41);
    let mut proposal = LayoutProposal::for_profile(TrackingProfile::Basic, 120);
    proposal.extra_signals = vec![SignalId::new(37).unwrap(), SignalId::new(41).unwrap()];
    ActiveLayout::negotiate(layout_id, proposal, &descriptor, LayoutLimits::default()).unwrap()
}

fn midpoint_samples(layout: &ActiveLayout) -> Vec<CompactSample> {
    layout
        .signals()
        .iter()
        .map(|&signal| {
            let scalar = SignalMetadata::get(signal).unwrap().scalar_type;
            let (minimum, maximum) = scalar.valid_range();
            CompactSample::available((minimum + maximum) * 0.5, 0.8, SignalState::Observed)
        })
        .collect()
}

fn encode(
    layout: &ActiveLayout,
    session_id: SessionId,
    generation: u32,
    sequence: u64,
    capture_timestamp_ns: u64,
) -> Vec<u8> {
    let samples = midpoint_samples(layout);
    CompactFrameCodec::encode(
        layout,
        &CompactFrameInput {
            session_id,
            generation,
            sequence,
            capture_timestamp_ns,
            produced_timestamp_ns: capture_timestamp_ns + 1,
            samples: &samples,
        },
    )
    .unwrap()
}

#[test]
fn basic_layout_keeps_registered_extras_in_consensus_order() {
    let descriptor = basic_descriptor(41);
    let mut gaze_then_tongue = LayoutProposal::for_profile(TrackingProfile::Basic, 120);
    gaze_then_tongue.extra_signals = vec![SignalId::new(37).unwrap(), SignalId::new(41).unwrap()];
    let first = ActiveLayout::negotiate(
        7,
        gaze_then_tongue.clone(),
        &descriptor,
        LayoutLimits::default(),
    )
    .unwrap();
    assert_eq!(first.signals().len(), 38);
    assert_eq!(first.signals()[36].get(), 37);
    assert_eq!(first.signals()[37].get(), 41);
    assert_eq!(first.frame_len(), 56 + 38 * 4);

    gaze_then_tongue.extra_signals.reverse();
    let reordered =
        ActiveLayout::negotiate(7, gaze_then_tongue, &descriptor, LayoutLimits::default()).unwrap();
    assert_ne!(first.hash(), reordered.hash());

    let repeated = basic_layout(7);
    assert_eq!(first.hash(), repeated.hash());
    assert_eq!(first.confirmation(), repeated.confirmation());
}

#[test]
fn basic_producer_can_append_registered_full_signals_without_profile_clipping() {
    let descriptor = basic_descriptor(76);
    assert_eq!(descriptor.guaranteed_profile, TrackingProfile::Basic);
    let mut proposal = LayoutProposal::for_profile(TrackingProfile::Basic, 120);
    proposal.extra_signals = vec![SignalId::new(42).unwrap(), SignalId::new(76).unwrap()];
    let layout =
        ActiveLayout::negotiate(11, proposal, &descriptor, LayoutLimits::default()).unwrap();
    assert_eq!(
        layout.signals()[36..],
        [SignalId::new(42).unwrap(), SignalId::new(76).unwrap()]
    );
}

#[test]
fn canonical_layout_hash_matches_language_neutral_golden_vector() {
    let vector: CompactLayoutVector =
        serde_json::from_str(include_str!("vectors/compact-basic-extras-v1.json")).unwrap();
    assert_eq!(vector.schema, "ntp.compact-layout-vector/1");
    let descriptor = basic_descriptor(41);
    let mut proposal = LayoutProposal::for_profile(TrackingProfile::Basic, vector.target_fps);
    proposal.extra_signals = vector
        .extra_signal_ids
        .iter()
        .map(|&raw| SignalId::new(raw).unwrap())
        .collect();
    let layout = ActiveLayout::negotiate(
        vector.layout_id,
        proposal,
        &descriptor,
        LayoutLimits::default(),
    )
    .unwrap();
    assert_eq!(
        layout
            .signals()
            .iter()
            .map(|signal| signal.get())
            .collect::<Vec<_>>(),
        vector.ordered_signal_ids
    );
    assert_eq!(layout.parameter_count(), vector.parameter_count);
    assert_eq!(layout.expected_payload_len(), vector.expected_payload_len);
    assert_eq!(hex(&layout.hash()), vector.layout_hash_hex);
}

#[test]
fn handshake_is_rate_limited_bounded_and_requires_hash_confirmation() {
    let descriptor = basic_descriptor(41);
    let mut negotiator = LayoutNegotiator::new(HandshakeLimits {
        min_proposal_interval_ns: 10,
        ..HandshakeLimits::default()
    })
    .unwrap();
    let proposal = LayoutProposal::for_profile(TrackingProfile::Basic, 120);
    let first = negotiator
        .receive_proposal(1, proposal.clone(), &descriptor, 100)
        .unwrap();
    assert_eq!(first.parameter_count, 36);
    assert_eq!(negotiator.pending_count(), 1);
    assert_eq!(
        negotiator.receive_proposal(2, proposal.clone(), &descriptor, 105),
        Err(HandshakeError::RateLimited)
    );
    let second = negotiator
        .receive_proposal(2, proposal, &descriptor, 110)
        .unwrap();
    assert_eq!(negotiator.pending_count(), 2);
    assert_eq!(
        negotiator.receive_proposal(
            3,
            LayoutProposal::for_profile(TrackingProfile::Basic, 120),
            &descriptor,
            120
        ),
        Err(HandshakeError::TooManyPendingLayouts)
    );

    let mut wrong = first.confirmation();
    wrong.layout_hash[0] ^= 1;
    assert_eq!(
        negotiator.confirm(wrong),
        Err(HandshakeError::ConfirmationMismatch)
    );
    let active = negotiator.confirm(first.confirmation()).unwrap();
    assert_eq!(active.layout_id(), 1);
    assert_eq!(negotiator.pending_count(), 1);
    assert_eq!(
        negotiator
            .confirm(second.confirmation())
            .unwrap()
            .layout_id(),
        2
    );
}

#[test]
fn layout_negotiation_fails_closed_on_contract_capability_and_resource_errors() {
    let descriptor = basic_descriptor(41);
    let limits = LayoutLimits::default();

    let mut proposal = LayoutProposal::for_profile(TrackingProfile::Spatial, 120);
    assert_eq!(
        ActiveLayout::negotiate(1, proposal.clone(), &descriptor, limits),
        Err(LayoutError::ProfileMismatch)
    );

    proposal.profile = TrackingProfile::Basic;
    proposal.revisions.schema_revision += 1;
    assert_eq!(
        ActiveLayout::negotiate(1, proposal.clone(), &descriptor, limits),
        Err(LayoutError::IncompatibleRevisions)
    );
    proposal.revisions = ContractRevisions::NTP_V1;

    proposal.extra_signals = vec![SignalId::new(36).unwrap()];
    assert_eq!(
        ActiveLayout::negotiate(1, proposal.clone(), &descriptor, limits),
        Err(LayoutError::BaseSignalListedAsExtra(
            SignalId::new(36).unwrap()
        ))
    );
    proposal.extra_signals = vec![SignalId::new(37).unwrap(), SignalId::new(37).unwrap()];
    assert_eq!(
        ActiveLayout::negotiate(1, proposal.clone(), &descriptor, limits),
        Err(LayoutError::DuplicateSignal(SignalId::new(37).unwrap()))
    );
    proposal.extra_signals = vec![SignalId::new(42).unwrap()];
    assert_eq!(
        ActiveLayout::negotiate(1, proposal.clone(), &descriptor, limits),
        Err(LayoutError::UnsupportedSignal(SignalId::new(42).unwrap()))
    );
    proposal.extra_signals = vec![SignalId::new(89).unwrap()];
    assert_eq!(
        ActiveLayout::negotiate(1, proposal.clone(), &descriptor, limits),
        Err(LayoutError::UnregisteredSignal(SignalId::new(89).unwrap()))
    );

    proposal.extra_signals.clear();
    proposal.target_fps = 241;
    assert_eq!(
        ActiveLayout::negotiate(1, proposal.clone(), &descriptor, limits),
        Err(LayoutError::InvalidTargetFps)
    );
    proposal.target_fps = 120;
    proposal.extra_signals = vec![SignalId::new(37).unwrap()];
    assert_eq!(
        ActiveLayout::negotiate(
            1,
            proposal.clone(),
            &descriptor,
            LayoutLimits {
                max_layout_bytes: 64,
                ..limits
            }
        ),
        Err(LayoutError::LayoutTooLarge)
    );
    proposal.extra_signals.clear();
    assert_eq!(
        ActiveLayout::negotiate(
            1,
            proposal.clone(),
            &descriptor,
            LayoutLimits {
                max_signals: 35,
                ..limits
            }
        ),
        Err(LayoutError::TooManySignals)
    );
    assert_eq!(
        ActiveLayout::negotiate(
            1,
            proposal,
            &descriptor,
            LayoutLimits {
                max_frame_bytes: 100,
                ..limits
            }
        ),
        Err(LayoutError::FrameTooLarge)
    );

    let partial = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::new(),
        StructureFeatures::empty(),
        TrackingFeatures::empty(),
    );
    assert_eq!(
        ActiveLayout::negotiate(
            1,
            LayoutProposal::for_profile(TrackingProfile::Partial, 60),
            &partial,
            limits
        ),
        Err(LayoutError::UnsupportedProfile)
    );
}

#[test]
fn all_registered_scalar_types_round_trip_with_quantization_bound() {
    let descriptor = full_descriptor();
    let mut proposal = LayoutProposal::for_profile(TrackingProfile::Full, 120);
    proposal.extra_signals = (77..=88).map(|raw| SignalId::new(raw).unwrap()).collect();
    let layout =
        ActiveLayout::negotiate(9, proposal, &descriptor, LayoutLimits::default()).unwrap();

    for fraction in [0.1_f32, 0.5, 0.9] {
        let samples = layout
            .signals()
            .iter()
            .map(|&signal| {
                let scalar = SignalMetadata::get(signal).unwrap().scalar_type;
                let (minimum, maximum) = scalar.valid_range();
                CompactSample::available(
                    fraction.mul_add(maximum - minimum, minimum),
                    fraction,
                    SignalState::Fused,
                )
            })
            .collect::<Vec<_>>();
        let bytes = CompactFrameCodec::encode(
            &layout,
            &CompactFrameInput {
                session_id: session(1),
                generation: 2,
                sequence: 3,
                capture_timestamp_ns: 10,
                produced_timestamp_ns: 11,
                samples: &samples,
            },
        )
        .unwrap();
        let decoded = CompactFrameCodec::decode(&layout, &bytes).unwrap();
        for (expected, actual) in samples.iter().zip(decoded.samples()) {
            let actual = actual.unwrap();
            let scalar = SignalMetadata::get(actual.signal).unwrap().scalar_type;
            let (minimum, maximum) = scalar.valid_range();
            let tolerance = (maximum - minimum) / 65_534.0 + f32::EPSILON * 8.0;
            assert!((actual.value.unwrap() - expected.value.unwrap()).abs() <= tolerance);
            assert!((actual.confidence - expected.confidence).abs() <= 1.0 / 255.0);
            assert_eq!(actual.state, SignalState::Fused);
        }
    }
}

#[test]
fn unavailable_values_require_negotiated_quality_and_consistent_state() {
    let layout = basic_layout(1);
    let mut samples = midpoint_samples(&layout);
    samples[0] = CompactSample::unavailable(0.2, SignalState::Occluded);
    let bytes = CompactFrameCodec::encode(
        &layout,
        &CompactFrameInput {
            session_id: session(1),
            generation: 0,
            sequence: 1,
            capture_timestamp_ns: 1,
            produced_timestamp_ns: 1,
            samples: &samples,
        },
    )
    .unwrap();
    let decoded = CompactFrameCodec::decode(&layout, &bytes).unwrap();
    let first = decoded.sample(0).unwrap().unwrap();
    assert_eq!(first.value, None);
    assert_eq!(first.state, SignalState::Occluded);

    let descriptor = basic_descriptor(36);
    let mut proposal = LayoutProposal::for_profile(TrackingProfile::Basic, 60);
    proposal.quality_encoding = QualityEncoding::None;
    let no_quality =
        ActiveLayout::negotiate(2, proposal, &descriptor, LayoutLimits::default()).unwrap();
    let mut samples = midpoint_samples(&no_quality);
    samples[0] = CompactSample::unavailable(0.0, SignalState::TrackingLost);
    assert_eq!(
        CompactFrameCodec::encode(
            &no_quality,
            &CompactFrameInput {
                session_id: session(1),
                generation: 0,
                sequence: 1,
                capture_timestamp_ns: 1,
                produced_timestamp_ns: 1,
                samples: &samples,
            }
        ),
        Err(CompactFrameError::InvalidStateValue {
            signal: SignalId::new(1).unwrap()
        })
    );
}

#[test]
fn malformed_frames_are_rejected_by_exact_structure_and_semantics() {
    let layout = basic_layout(3);
    let valid = encode(&layout, session(1), 4, 10, 100);
    assert!(CompactFrameCodec::decode(&layout, &valid).is_ok());
    assert!(matches!(
        CompactFrameCodec::decode(&layout, &valid[..valid.len() - 1]),
        Err(CompactFrameError::WrongLength { .. })
    ));
    let mut trailing = valid.clone();
    trailing.push(0);
    assert!(matches!(
        CompactFrameCodec::decode(&layout, &trailing),
        Err(CompactFrameError::WrongLength { .. })
    ));

    let mut malformed = valid.clone();
    malformed[0] ^= 1;
    assert_eq!(
        CompactFrameCodec::decode(&layout, &malformed).unwrap_err(),
        CompactFrameError::InvalidMagic
    );
    malformed = valid.clone();
    malformed[6] = 1;
    assert_eq!(
        CompactFrameCodec::decode(&layout, &malformed).unwrap_err(),
        CompactFrameError::NonZeroReserved
    );
    malformed = valid.clone();
    malformed[28..32].copy_from_slice(&99_u32.to_le_bytes());
    assert_eq!(
        CompactFrameCodec::decode(&layout, &malformed).unwrap_err(),
        CompactFrameError::WrongLayout {
            expected: 3,
            actual: 99
        }
    );

    let quality_offset = 56 + layout.signals().len() * 2;
    malformed = valid.clone();
    malformed[quality_offset] = 7;
    assert_eq!(
        CompactFrameCodec::decode(&layout, &malformed).unwrap_err(),
        CompactFrameError::InvalidState(7)
    );
    malformed = valid.clone();
    malformed[56..58].copy_from_slice(&i16::MIN.to_le_bytes());
    assert_eq!(
        CompactFrameCodec::decode(&layout, &malformed).unwrap_err(),
        CompactFrameError::InvalidStateValue {
            signal: SignalId::new(1).unwrap()
        }
    );
    malformed = valid;
    malformed[quality_offset] = SignalState::Unsupported as u8;
    assert_eq!(
        CompactFrameCodec::decode(&layout, &malformed).unwrap_err(),
        CompactFrameError::UnsupportedState {
            signal: SignalId::new(1).unwrap()
        }
    );
}

#[test]
fn stream_guard_requires_confirmation_and_rejects_replay_time_and_layout_switch_errors() {
    let layout = basic_layout(3);
    let mut wrong_confirmation = layout.confirmation();
    wrong_confirmation.layout_hash[0] ^= 1;
    assert!(matches!(
        CompactStreamGuard::confirmed(
            session(1),
            4,
            layout.clone(),
            wrong_confirmation,
            CompactStreamPolicy::default()
        ),
        Err(CompactStreamError::ConfirmationMismatch)
    ));

    let policy = CompactStreamPolicy {
        max_frame_age_ns: 20,
        max_future_skew_ns: 5,
        max_sequence_gap: 2,
        max_capture_jump_ns: 10,
    };
    let mut guard =
        CompactStreamGuard::confirmed(session(1), 4, layout.clone(), layout.confirmation(), policy)
            .unwrap();
    let first = encode(&layout, session(1), 4, 10, 100);
    assert_eq!(guard.accept(&first, 101).unwrap().sequence, 10);
    assert!(matches!(
        guard.accept(&first, 101),
        Err(CompactStreamError::DuplicateOrReplay { .. })
    ));

    let gap = encode(&layout, session(1), 4, 14, 101);
    assert_eq!(
        guard.accept(&gap, 102).unwrap_err(),
        CompactStreamError::SequenceGapTooLarge
    );
    let next = encode(&layout, session(1), 4, 11, 102);
    assert_eq!(guard.accept(&next, 103).unwrap().sequence, 11);

    let regressed = encode(&layout, session(1), 4, 12, 101);
    assert_eq!(
        guard.accept(&regressed, 103).unwrap_err(),
        CompactStreamError::CaptureTimestampRegressed
    );
    let jumped = encode(&layout, session(1), 4, 12, 120);
    assert_eq!(
        guard.accept(&jumped, 120).unwrap_err(),
        CompactStreamError::CaptureTimestampJump
    );

    let stale = encode(&layout, session(1), 4, 12, 50);
    assert_eq!(
        guard.accept(&stale, 100).unwrap_err(),
        CompactStreamError::StaleFrame
    );
    let future = encode(&layout, session(1), 4, 12, 110);
    assert_eq!(
        guard.accept(&future, 100).unwrap_err(),
        CompactStreamError::FutureFrame
    );

    let next_layout = basic_layout(4);
    assert_eq!(
        guard.switch_layout(4, next_layout.clone(), next_layout.confirmation()),
        Err(CompactStreamError::GenerationDidNotAdvance)
    );
    guard
        .switch_layout(5, next_layout.clone(), next_layout.confirmation())
        .unwrap();
    let switched = encode(&next_layout, session(1), 5, 1, 120);
    assert_eq!(guard.accept(&switched, 121).unwrap().sequence, 1);
    let wrong_session = encode(&next_layout, session(2), 5, 2, 121);
    assert_eq!(
        guard.accept(&wrong_session, 122).unwrap_err(),
        CompactStreamError::Frame(CompactFrameError::WrongSession)
    );
    let wrong_generation = encode(&next_layout, session(1), 4, 2, 121);
    assert_eq!(
        guard.accept(&wrong_generation, 122).unwrap_err(),
        CompactStreamError::Frame(CompactFrameError::WrongGeneration {
            expected: 5,
            actual: 4
        })
    );
    assert!(matches!(
        guard.accept(&first, 121),
        Err(CompactStreamError::Frame(
            CompactFrameError::WrongLayout { .. }
        ))
    ));
}

#[test]
fn latest_frame_handoff_is_bounded_to_one_unread_frame() {
    let mut slot = LatestFrame::default();
    slot.publish(1_u64);
    slot.publish(2);
    slot.publish(3);
    assert_eq!(slot.dropped(), 2);
    assert_eq!(slot.take(), Some(3));
    assert_eq!(slot.take(), None);
}

#[test]
fn recorder_requires_layout_record_before_referenced_frames() {
    let layout = basic_layout(3);
    let bytes = encode(&layout, session(1), 4, 1, 100);
    let mut recorder = CompactRecorder::new(MemoryRecording::default());
    assert_eq!(
        recorder.record_frame(&bytes),
        Err(RecordingError::NoActiveLayout)
    );
    recorder
        .begin_layout(session(1), 4, layout.clone())
        .unwrap();
    recorder.record_frame(&bytes).unwrap();

    let wrong_generation = encode(&layout, session(1), 5, 2, 101);
    assert_eq!(
        recorder.record_frame(&wrong_generation),
        Err(RecordingError::WrongGeneration)
    );
    let recording = recorder.into_inner();
    assert_eq!(recording.layouts.len(), 1);
    assert_eq!(recording.layouts[0].layout_hash, layout.hash());
    assert_eq!(recording.frames, vec![bytes]);
}

#[test]
fn angle_upper_endpoint_is_not_a_legal_wire_value() {
    let descriptor = full_descriptor();
    let layout = ActiveLayout::negotiate(
        10,
        LayoutProposal::for_profile(TrackingProfile::Full, 60),
        &descriptor,
        LayoutLimits::default(),
    )
    .unwrap();
    let mut bytes = encode(&layout, session(1), 0, 1, 1);
    let angle_index = 44;
    assert_eq!(
        SignalMetadata::get(layout.signals()[angle_index])
            .unwrap()
            .scalar_type,
        ScalarType::Angle
    );
    let offset = 56 + angle_index * 2;
    bytes[offset..offset + 2].copy_from_slice(&i16::MAX.to_le_bytes());
    assert_eq!(
        CompactFrameCodec::decode(&layout, &bytes).unwrap_err(),
        CompactFrameError::InvalidValue {
            signal: SignalId::new(45).unwrap()
        }
    );

    let mut samples = midpoint_samples(&layout);
    samples[angle_index] = CompactSample::available(
        core::f32::consts::PI - f32::EPSILON,
        1.0,
        SignalState::Observed,
    );
    let near_upper = CompactFrameCodec::encode(
        &layout,
        &CompactFrameInput {
            session_id: session(1),
            generation: 0,
            sequence: 2,
            capture_timestamp_ns: 2,
            produced_timestamp_ns: 2,
            samples: &samples,
        },
    )
    .unwrap();
    assert!(CompactFrameCodec::decode(&layout, &near_upper).is_ok());
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, byte| {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").unwrap();
        output
    })
}
