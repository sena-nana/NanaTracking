#![allow(dead_code)]

use nana_tracking_protocol::{
    CoordinateSpace, Direction3, LengthBasis, NanaTrackingDescriptor, NanaTrackingResult, Pose,
    Position3, Quaternion, RegionQuality, SessionId, SignalBitSet, SignalId, SignalSample,
    SignalState, StructureFeatures, Tracked, TrackingFeatures, Vec3,
};

pub fn descriptor(
    stable_through: u16,
    structures: StructureFeatures,
    extras: &[u16],
) -> NanaTrackingDescriptor {
    let mut signals = SignalBitSet::stable_through(stable_through);
    for raw in extras {
        signals.insert(SignalId::new(*raw).unwrap());
    }
    NanaTrackingDescriptor::from_capabilities(signals, structures, TrackingFeatures::empty())
}

pub fn frame(
    descriptor: &NanaTrackingDescriptor,
    session: u8,
    generation: u32,
    sequence: u64,
    timestamp_ns: u64,
) -> NanaTrackingResult {
    let mut result = NanaTrackingResult::unsupported(
        SessionId([session; 16]),
        generation,
        sequence,
        timestamp_ns,
        timestamp_ns + 1_000_000,
    );
    for id in descriptor.supported_signals.iter() {
        if id.stable_slot().is_some() {
            result.rig.set(
                id,
                SignalSample::available(0.0, 0.9, SignalState::Observed, timestamp_ns, 0),
            );
        }
    }

    let structures = descriptor.supported_structures;
    if structures.contains(StructureFeatures::HEAD_GEOMETRY) {
        result.geometry.head_camera_pose = tracked_pose(
            CoordinateSpace::Camera,
            LengthBasis::HeadRelative,
            Vec3::default(),
            timestamp_ns,
        );
    }
    if structures.contains(StructureFeatures::EYE_GEOMETRY) {
        result.geometry.eyes.left.origin_head = tracked_position(
            CoordinateSpace::HeadLocal,
            LengthBasis::HeadRelative,
            Vec3 {
                x: -0.15,
                y: 0.0,
                z: 0.0,
            },
            timestamp_ns,
        );
        result.geometry.eyes.right.origin_head = tracked_position(
            CoordinateSpace::HeadLocal,
            LengthBasis::HeadRelative,
            Vec3 {
                x: 0.15,
                y: 0.0,
                z: 0.0,
            },
            timestamp_ns,
        );
        result.geometry.eyes.left.direction_head = tracked_direction(
            CoordinateSpace::HeadLocal,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            timestamp_ns,
        );
        result.geometry.eyes.right.direction_head =
            result.geometry.eyes.left.direction_head.clone();
    }
    if structures.contains(StructureFeatures::LOOK_AT_POINT) {
        result.geometry.look_at_camera = tracked_position(
            CoordinateSpace::Camera,
            LengthBasis::HeadRelative,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            timestamp_ns,
        );
    }
    if structures.contains(StructureFeatures::FACE_GEOMETRY) {
        result.geometry.face_geometry_state = SignalState::Observed;
    }
    if structures.contains(StructureFeatures::BODY_SKELETON) {
        fill_skeleton(&mut result, timestamp_ns);
    }
    fill_quality(descriptor, &mut result);
    result
}

pub fn set(result: &mut NanaTrackingResult, raw: u16, value: f32) {
    result.rig.set(
        SignalId::new(raw).unwrap(),
        SignalSample::available(
            value,
            0.9,
            SignalState::Observed,
            result.capture_timestamp_ns,
            0,
        ),
    );
}

pub fn set_arm_out_of_frame(result: &mut NanaTrackingResult) {
    for raw in 63..=76 {
        result.rig.set(
            SignalId::new(raw).unwrap(),
            SignalSample::unavailable(0.2, SignalState::OutOfFrame, result.capture_timestamp_ns),
        );
    }
    let unavailable_pose =
        || Tracked::unavailable(0.2, SignalState::OutOfFrame, result.capture_timestamp_ns);
    result.skeleton.shoulder.left = unavailable_pose();
    result.skeleton.shoulder.right = unavailable_pose();
    result.skeleton.elbow.left = unavailable_pose();
    result.skeleton.elbow.right = unavailable_pose();
    result.skeleton.wrist.left = unavailable_pose();
    result.skeleton.wrist.right = unavailable_pose();
    let unavailable_direction =
        || Tracked::unavailable(0.2, SignalState::OutOfFrame, result.capture_timestamp_ns);
    result.skeleton.upper_arm_direction_torso.left = unavailable_direction();
    result.skeleton.upper_arm_direction_torso.right = unavailable_direction();
    result.skeleton.forearm_direction_torso.left = unavailable_direction();
    result.skeleton.forearm_direction_torso.right = unavailable_direction();
    let unavailable_scalar =
        || Tracked::unavailable(0.2, SignalState::OutOfFrame, result.capture_timestamp_ns);
    result.skeleton.upper_arm_twist.left = unavailable_scalar();
    result.skeleton.upper_arm_twist.right = unavailable_scalar();
    result.skeleton.forearm_twist.left = unavailable_scalar();
    result.skeleton.forearm_twist.right = unavailable_scalar();
    result.quality.arm.left = RegionQuality {
        confidence: 0.2,
        state: SignalState::OutOfFrame,
    };
    result.quality.arm.right = result.quality.arm.left;
}

fn fill_skeleton(result: &mut NanaTrackingResult, timestamp_ns: u64) {
    result.skeleton.torso_camera_pose = tracked_pose(
        CoordinateSpace::Camera,
        LengthBasis::TorsoRelative,
        Vec3::default(),
        timestamp_ns,
    );
    for (target, position) in [
        (
            &mut result.skeleton.shoulder.left,
            Vec3 {
                x: -0.2,
                y: 0.0,
                z: 0.0,
            },
        ),
        (
            &mut result.skeleton.shoulder.right,
            Vec3 {
                x: 0.2,
                y: 0.0,
                z: 0.0,
            },
        ),
        (
            &mut result.skeleton.elbow.left,
            Vec3 {
                x: -0.2,
                y: 0.4,
                z: 0.0,
            },
        ),
        (
            &mut result.skeleton.elbow.right,
            Vec3 {
                x: 0.2,
                y: 0.4,
                z: 0.0,
            },
        ),
        (
            &mut result.skeleton.wrist.left,
            Vec3 {
                x: -0.2,
                y: 0.8,
                z: 0.0,
            },
        ),
        (
            &mut result.skeleton.wrist.right,
            Vec3 {
                x: 0.2,
                y: 0.8,
                z: 0.0,
            },
        ),
    ] {
        *target = tracked_pose(
            CoordinateSpace::TorsoLocal,
            LengthBasis::TorsoRelative,
            position,
            timestamp_ns,
        );
    }
    for target in [
        &mut result.skeleton.upper_arm_direction_torso.left,
        &mut result.skeleton.upper_arm_direction_torso.right,
        &mut result.skeleton.forearm_direction_torso.left,
        &mut result.skeleton.forearm_direction_torso.right,
    ] {
        *target = tracked_direction(
            CoordinateSpace::TorsoLocal,
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            timestamp_ns,
        );
    }
    for target in [
        &mut result.skeleton.upper_arm_twist.left,
        &mut result.skeleton.upper_arm_twist.right,
        &mut result.skeleton.forearm_twist.left,
        &mut result.skeleton.forearm_twist.right,
    ] {
        *target = Tracked::available(0.0, 0.9, SignalState::Observed, timestamp_ns, 0);
    }
}

fn fill_quality(descriptor: &NanaTrackingDescriptor, result: &mut NanaTrackingResult) {
    result.quality.overall_confidence = 0.9;
    let structures = descriptor.supported_structures;
    let supports_any = |first: u16, last: u16| {
        (first..=last).any(|raw| {
            descriptor
                .supported_signals
                .contains(SignalId::new(raw).unwrap())
        })
    };
    let supports = |ids: &[u16]| {
        ids.iter().any(|raw| {
            descriptor
                .supported_signals
                .contains(SignalId::new(*raw).unwrap())
        })
    };
    if supports_any(1, 36) || structures.contains(StructureFeatures::HEAD_GEOMETRY) {
        result.quality.face = observed_region();
    }
    if supports(&[7, 8, 9, 10, 37, 38, 39, 40])
        || structures.contains(StructureFeatures::EYE_GEOMETRY)
    {
        result.quality.eyes = observed_region();
    }
    if supports_any(42, 53) || structures.contains(StructureFeatures::BODY_SKELETON) {
        result.quality.torso = observed_region();
    }
    if supports(&[63, 65, 67, 68, 69, 70, 71])
        || structures.contains(StructureFeatures::BODY_SKELETON)
    {
        result.quality.arm.left = observed_region();
    }
    if supports(&[64, 66, 72, 73, 74, 75, 76])
        || structures.contains(StructureFeatures::BODY_SKELETON)
    {
        result.quality.arm.right = observed_region();
    }
    if supports(&[57, 59, 61, 81]) {
        result.quality.auricle.left = observed_region();
    }
    if supports(&[58, 60, 62, 82]) {
        result.quality.auricle.right = observed_region();
    }
}

const fn observed_region() -> RegionQuality {
    RegionQuality {
        confidence: 0.9,
        state: SignalState::Observed,
    }
}

fn tracked_pose(
    parent_space: CoordinateSpace,
    length_basis: LengthBasis,
    position: Vec3,
    timestamp_ns: u64,
) -> Tracked<Pose> {
    Tracked::available(
        Pose {
            parent_space,
            length_basis,
            position,
            orientation_xyzw: Quaternion::IDENTITY,
        },
        0.9,
        SignalState::Observed,
        timestamp_ns,
        0,
    )
}

fn tracked_position(
    space: CoordinateSpace,
    length_basis: LengthBasis,
    value: Vec3,
    timestamp_ns: u64,
) -> Tracked<Position3> {
    Tracked::available(
        Position3 {
            space,
            length_basis,
            value,
        },
        0.9,
        SignalState::Observed,
        timestamp_ns,
        0,
    )
}

fn tracked_direction(
    space: CoordinateSpace,
    value: Vec3,
    timestamp_ns: u64,
) -> Tracked<Direction3> {
    Tracked::available(
        Direction3 { space, value },
        0.9,
        SignalState::Observed,
        timestamp_ns,
        0,
    )
}
