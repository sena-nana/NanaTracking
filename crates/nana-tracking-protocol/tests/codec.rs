use nana_tracking_protocol::{
    CanonicalCodec, CoordinateSpace, Direction3, LengthBasis, NanaTrackingDescriptor,
    NanaTrackingResult, Pose, Position3, Quaternion, RegionQuality, SessionId, SideMap,
    SignalBitSet, SignalId, SignalSample, SignalState, StructureFeatures, Tracked,
    TrackingFeatures, Validate, Vec3, WireDecode,
    ffi::{NTP_STATE_UNSUPPORTED, NtpCoreResultC, NtpDescriptorC, NtpSignalSampleC},
};
use std::fmt::Write as _;
use std::mem::size_of;

fn descriptor() -> NanaTrackingDescriptor {
    NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(36),
        StructureFeatures::HEAD_GEOMETRY,
        TrackingFeatures::METRIC_COORDINATES,
    )
}

fn result_with_quaternion(quaternion: Quaternion) -> NanaTrackingResult {
    let mut result = NanaTrackingResult::unsupported(SessionId([7; 16]), 2, 9, 1_000, 1_025);
    result.rig.set(
        SignalId::new(1).unwrap(),
        SignalSample::available(0.25, 0.8, SignalState::Fused, 1_000, 0),
    );
    result.geometry.head_camera_pose = Tracked::available(
        Pose {
            parent_space: CoordinateSpace::Camera,
            length_basis: LengthBasis::HeadRelative,
            position: Vec3 {
                x: 0.1,
                y: -0.2,
                z: 0.3,
            },
            orientation_xyzw: quaternion,
        },
        0.85,
        SignalState::Fused,
        1_000,
        0,
    );
    result
}

fn full_result() -> (NanaTrackingDescriptor, NanaTrackingResult) {
    let descriptor = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(76),
        StructureFeatures::FULL_REQUIRED,
        TrackingFeatures::WRIST_POSE,
    );
    let mut result = NanaTrackingResult::unsupported(SessionId([9; 16]), 3, 21, 2_000, 2_020);
    for raw in 1..=76 {
        result.rig.set(
            SignalId::new(raw).unwrap(),
            SignalSample::available(0.0, 0.7, SignalState::Observed, 2_000, 0),
        );
    }
    result.geometry.head_camera_pose =
        tracked_pose(CoordinateSpace::Camera, LengthBasis::HeadRelative);
    let eye_origin = Tracked::available(
        Position3 {
            space: CoordinateSpace::HeadLocal,
            length_basis: LengthBasis::HeadRelative,
            value: Vec3::default(),
        },
        0.8,
        SignalState::Fused,
        2_000,
        0,
    );
    let eye_direction = tracked_direction(CoordinateSpace::HeadLocal);
    result.geometry.eyes.left.origin_head = eye_origin.clone();
    result.geometry.eyes.right.origin_head = eye_origin;
    result.geometry.eyes.left.direction_head = eye_direction.clone();
    result.geometry.eyes.right.direction_head = eye_direction;
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
        0.8,
        SignalState::Fused,
        2_000,
        0,
    );
    result.geometry.face_geometry_state = SignalState::Fused;

    result.skeleton.torso_camera_pose =
        tracked_pose(CoordinateSpace::Camera, LengthBasis::TorsoRelative);
    let joint = tracked_pose(CoordinateSpace::TorsoLocal, LengthBasis::TorsoRelative);
    result.skeleton.shoulder = SideMap {
        left: joint.clone(),
        right: joint.clone(),
    };
    result.skeleton.elbow = SideMap {
        left: joint.clone(),
        right: joint.clone(),
    };
    result.skeleton.wrist = SideMap {
        left: joint.clone(),
        right: joint,
    };
    let direction = tracked_direction(CoordinateSpace::TorsoLocal);
    result.skeleton.upper_arm_direction_torso = SideMap {
        left: direction.clone(),
        right: direction.clone(),
    };
    result.skeleton.forearm_direction_torso = SideMap {
        left: direction.clone(),
        right: direction,
    };
    let twist = Tracked::available(0.0, 0.8, SignalState::Fused, 2_000, 0);
    result.skeleton.upper_arm_twist = SideMap {
        left: twist.clone(),
        right: twist.clone(),
    };
    result.skeleton.forearm_twist = SideMap {
        left: twist.clone(),
        right: twist,
    };

    let supported_but_lost = RegionQuality {
        confidence: 0.0,
        state: SignalState::TrackingLost,
    };
    result.quality.face = supported_but_lost;
    result.quality.eyes = supported_but_lost;
    result.quality.torso = supported_but_lost;
    result.quality.arm = SideMap {
        left: supported_but_lost,
        right: supported_but_lost,
    };
    result.quality.auricle = SideMap {
        left: supported_but_lost,
        right: supported_but_lost,
    };
    (descriptor, result)
}

fn tracked_pose(space: CoordinateSpace, basis: LengthBasis) -> Tracked<Pose> {
    Tracked::available(
        Pose {
            parent_space: space,
            length_basis: basis,
            position: Vec3::default(),
            orientation_xyzw: Quaternion::IDENTITY,
        },
        0.8,
        SignalState::Fused,
        2_000,
        0,
    )
}

fn tracked_direction(space: CoordinateSpace) -> Tracked<Direction3> {
    Tracked::available(
        Direction3 {
            space,
            value: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        },
        0.8,
        SignalState::Fused,
        2_000,
        0,
    )
}

#[test]
fn descriptor_round_trip_is_canonical_and_matches_fixed_vector() {
    let descriptor = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::new(),
        StructureFeatures::empty(),
        TrackingFeatures::empty(),
    );
    let encoded = CanonicalCodec::encode(&descriptor).unwrap();
    let expected_hex = include_str!("vectors/partial-descriptor-v1.hex").trim();
    assert_eq!(hex(&encoded), expected_hex);
    assert_eq!(
        NanaTrackingDescriptor::decode_wire(&encoded).unwrap(),
        descriptor
    );
}

#[test]
fn result_round_trip_preserves_fixed_schema() {
    let result = result_with_quaternion(Quaternion::IDENTITY);
    result.validate().unwrap();
    let encoded = CanonicalCodec::encode(&result).unwrap();
    assert!(encoded.len() < size_of::<NtpCoreResultC>());
    let decoded = NanaTrackingResult::decode_wire(&encoded).unwrap();
    assert_eq!(decoded, result);
}

#[test]
fn full_profile_all_structure_variants_round_trip_and_validate() {
    let (descriptor, result) = full_result();
    descriptor.validate_result(&result).unwrap();
    let encoded = CanonicalCodec::encode(&result).unwrap();
    assert!(encoded.len() < size_of::<NtpCoreResultC>());
    let decoded = NanaTrackingResult::decode_wire(&encoded).unwrap();
    assert_eq!(decoded, result);

    let result_c = NtpCoreResultC::from(&result);
    let from_c = NanaTrackingResult::try_from(&result_c).unwrap();
    assert_eq!(CanonicalCodec::encode(&from_c).unwrap(), encoded);
}

#[test]
fn quaternion_sign_has_one_canonical_byte_representation() {
    let positive = result_with_quaternion(Quaternion::IDENTITY);
    let negative = result_with_quaternion(Quaternion {
        x: -0.0,
        y: -0.0,
        z: -0.0,
        w: -1.0,
    });
    assert_eq!(
        CanonicalCodec::encode(&positive).unwrap(),
        CanonicalCodec::encode(&negative).unwrap()
    );
}

#[test]
fn signed_zero_has_one_canonical_byte_representation() {
    let mut positive = result_with_quaternion(Quaternion::IDENTITY);
    positive.rig.set(
        SignalId::new(1).unwrap(),
        SignalSample::available(0.0, 0.8, SignalState::Fused, 1_000, 0),
    );
    let mut negative = positive.clone();
    negative.rig.set(
        SignalId::new(1).unwrap(),
        SignalSample::available(-0.0, 0.8, SignalState::Fused, 1_000, 0),
    );
    assert_eq!(
        CanonicalCodec::encode(&positive).unwrap(),
        CanonicalCodec::encode(&negative).unwrap()
    );
}

#[test]
fn unknown_top_level_fields_and_signal_ids_are_skipped() {
    let result = result_with_quaternion(Quaternion::IDENTITY);
    let encoded = CanonicalCodec::encode(&result).unwrap();
    let with_unknown_field = append_unknown_tlv(encoded, 0x7000, &[1, 2, 3, 4]);
    assert_eq!(
        NanaTrackingResult::decode_wire(&with_unknown_field).unwrap(),
        result
    );

    let encoded = CanonicalCodec::encode(&result).unwrap();
    let with_unknown_signal = append_unknown_signal(encoded, 0x0100, &[0xaa, 0xbb, 0xcc]);
    assert_eq!(
        NanaTrackingResult::decode_wire(&with_unknown_signal).unwrap(),
        result
    );
}

#[test]
fn c_abi_conversion_is_separate_and_round_trips_core_contract() {
    assert_eq!(NtpSignalSampleC::default().state, NTP_STATE_UNSUPPORTED);
    let descriptor = descriptor();
    let descriptor_c = NtpDescriptorC::from(&descriptor);
    assert_eq!(
        NanaTrackingDescriptor::try_from(&descriptor_c).unwrap(),
        descriptor
    );

    let result = result_with_quaternion(Quaternion::IDENTITY);
    let result_c = NtpCoreResultC::from(&result);
    assert_eq!(NanaTrackingResult::try_from(&result_c).unwrap(), result);
}

#[cfg(feature = "diagnostic-json")]
#[test]
fn json_is_an_explicit_diagnostic_format() {
    let descriptor = descriptor();
    let json = nana_tracking_protocol::diagnostic::to_pretty_json(&descriptor).unwrap();
    let decoded: NanaTrackingDescriptor =
        nana_tracking_protocol::diagnostic::from_json(&json).unwrap();
    assert_eq!(decoded, descriptor);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").unwrap();
        output
    })
}

fn append_unknown_tlv(mut bytes: Vec<u8>, tag: u16, body: &[u8]) -> Vec<u8> {
    bytes.extend_from_slice(&tag.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(body);
    update_payload_length(&mut bytes, body.len() + 6);
    bytes
}

fn append_unknown_signal(mut bytes: Vec<u8>, id: u16, body: &[u8]) -> Vec<u8> {
    let mut offset = 12;
    while offset < bytes.len() {
        let tag = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        let length = u32::from_le_bytes(bytes[offset + 2..offset + 6].try_into().unwrap()) as usize;
        if tag == 2 {
            let field_start = offset + 6;
            let field_end = field_start + length;
            let count =
                u16::from_le_bytes(bytes[field_start + 2..field_start + 4].try_into().unwrap());
            bytes[field_start + 2..field_start + 4].copy_from_slice(&(count + 1).to_le_bytes());
            let mut entry = Vec::new();
            entry.extend_from_slice(&id.to_le_bytes());
            entry.extend_from_slice(&u16::try_from(body.len()).unwrap().to_le_bytes());
            entry.extend_from_slice(body);
            bytes.splice(field_end..field_end, entry.iter().copied());
            let added = entry.len();
            bytes[offset + 2..offset + 6]
                .copy_from_slice(&u32::try_from(length + added).unwrap().to_le_bytes());
            update_payload_length(&mut bytes, added);
            return bytes;
        }
        offset += 6 + length;
    }
    panic!("rig TLV not found")
}

fn update_payload_length(bytes: &mut [u8], added: usize) {
    let length = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    bytes[8..12].copy_from_slice(&(length + u32::try_from(added).unwrap()).to_le_bytes());
}
