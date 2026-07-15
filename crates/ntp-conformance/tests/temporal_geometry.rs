mod support;

use nana_tracking_protocol::{
    Direction3, Quaternion, SignalId, SignalSample, SignalState, StructureFeatures, Vec3,
};
use nana_tracking_semantics::{SemanticDeriver, SemanticId, Side};
use ntp_conformance::{ConformanceOptions, FailureCode, validate_stream};
use support::{descriptor, frame, set, set_arm_out_of_frame};

#[test]
fn torso_and_head_relative_rotations_are_one_composed_state() {
    let descriptor = descriptor(76, StructureFeatures::FULL_REQUIRED, &[]);
    let mut result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    result
        .skeleton
        .torso_camera_pose
        .value
        .as_mut()
        .unwrap()
        .orientation_xyzw = yaw_quaternion(0.4);
    result
        .geometry
        .head_camera_pose
        .value
        .as_mut()
        .unwrap()
        .orientation_xyzw = yaw_quaternion(-0.2);
    set(&mut result, 46, 0.4);
    set(&mut result, 52, -0.6);
    let report = validate_stream(
        &descriptor,
        &[result.clone()],
        ConformanceOptions::default(),
    );
    assert!(report.passed, "{report}");

    set(&mut result, 52, 0.0);
    let report = validate_stream(&descriptor, &[result], ConformanceOptions::default());
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::SkeletonScalarMismatch)
    );
}

#[test]
fn intrinsic_xyz_pose_projection_round_trips_all_three_axes() {
    let descriptor = descriptor(76, StructureFeatures::FULL_REQUIRED, &[]);
    let mut result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    let angles = Vec3 {
        x: 0.2,
        y: -0.3,
        z: 0.4,
    };
    let orientation = intrinsic_xyz_quaternion(angles);
    result
        .skeleton
        .torso_camera_pose
        .value
        .as_mut()
        .unwrap()
        .orientation_xyzw = orientation;
    result
        .geometry
        .head_camera_pose
        .value
        .as_mut()
        .unwrap()
        .orientation_xyzw = orientation;
    set(&mut result, 45, angles.x);
    set(&mut result, 46, angles.y);
    set(&mut result, 47, angles.z);
    let report = validate_stream(&descriptor, &[result], ConformanceOptions::default());
    assert!(report.passed, "{report}");
}

#[test]
fn comparable_absolute_positions_reproduce_head_relative_translation() {
    let descriptor = descriptor(76, StructureFeatures::FULL_REQUIRED, &[]);
    let mut result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    set_arm_out_of_frame(&mut result);
    let torso = result.skeleton.torso_camera_pose.value.as_mut().unwrap();
    torso.length_basis = nana_tracking_protocol::LengthBasis::HeadRelative;
    torso.position = Vec3 {
        x: 0.1,
        y: 0.2,
        z: 0.3,
    };
    result
        .geometry
        .head_camera_pose
        .value
        .as_mut()
        .unwrap()
        .position = Vec3 {
        x: 0.3,
        y: -0.1,
        z: 0.7,
    };
    set(&mut result, 48, 0.2);
    set(&mut result, 49, -0.3);
    set(&mut result, 50, 0.4);
    let report = validate_stream(
        &descriptor,
        std::slice::from_ref(&result),
        ConformanceOptions::default(),
    );
    assert!(report.passed, "{report}");

    set(&mut result, 48, -0.2);
    let report = validate_stream(&descriptor, &[result], ConformanceOptions::default());
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.code == FailureCode::SkeletonScalarMismatch)
    );
}

#[test]
fn articulated_wrist_geometry_drives_hand_near_face() {
    let descriptor = descriptor(76, StructureFeatures::FULL_REQUIRED, &[]);
    let mut result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    let shoulder = result
        .skeleton
        .shoulder
        .left
        .value
        .as_ref()
        .unwrap()
        .position;
    let wrist = Vec3 {
        x: 0.0,
        y: -0.38,
        z: 0.02,
    };
    let elbow = Vec3 {
        x: (shoulder.x + wrist.x) * 0.5,
        y: (shoulder.y + wrist.y) * 0.5,
        z: (shoulder.z + wrist.z) * 0.5,
    };
    result.skeleton.elbow.left.value.as_mut().unwrap().position = elbow;
    result.skeleton.wrist.left.value.as_mut().unwrap().position = wrist;
    let direction = normalize(Vec3 {
        x: wrist.x - shoulder.x,
        y: wrist.y - shoulder.y,
        z: wrist.z - shoulder.z,
    });
    result.skeleton.upper_arm_direction_torso.left.value = Some(Direction3 {
        space: nana_tracking_protocol::CoordinateSpace::TorsoLocal,
        value: direction,
    });
    result.skeleton.forearm_direction_torso.left.value = Some(Direction3 {
        space: nana_tracking_protocol::CoordinateSpace::TorsoLocal,
        value: direction,
    });
    set(&mut result, 67, direction.z);
    set(&mut result, 68, -direction.x);
    set(&mut result, 70, 0.0);

    let report = validate_stream(
        &descriptor,
        &[result.clone()],
        ConformanceOptions::default(),
    );
    assert!(report.passed, "{report}");
    let semantics = SemanticDeriver::default()
        .derive(&result, result.produced_timestamp_ns)
        .unwrap();
    assert!(
        (semantics
            .get(SemanticId::HandNearFace(Side::Left))
            .unwrap()
            .value
            - 1.0)
            .abs()
            < 1.0e-6
    );
}

#[test]
fn auricle_periodic_motion_uses_capture_time_for_velocity_and_energy() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[57, 59, 61]);
    let mut first = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    let mut second = frame(&descriptor, 1, 0, 2, 1_100_000_000);
    for raw in [57, 59, 61] {
        set(&mut first, raw, 0.0);
    }
    set(&mut second, 57, 0.4);
    set(&mut second, 59, 0.0);
    set(&mut second, 61, 0.0);
    let mut deriver = SemanticDeriver::default();
    deriver.derive(&first, first.produced_timestamp_ns).unwrap();
    let semantic = deriver
        .derive(&second, second.produced_timestamp_ns)
        .unwrap();
    let amplitude = semantic
        .get(SemanticId::AuricleWiggleAmplitude(Side::Left))
        .unwrap()
        .value;
    let velocity = semantic
        .get(SemanticId::AuricleWiggleVelocity(Side::Left))
        .unwrap()
        .value;
    let energy = semantic
        .get(SemanticId::AuricleWiggleEnergy(Side::Left))
        .unwrap()
        .value;
    assert!((amplitude - (0.16_f32 / 3.0).sqrt()).abs() < 1.0e-6);
    assert!((velocity - 1.0).abs() < 1.0e-6);
    assert!((energy - amplitude).abs() < 1.0e-6);
}

#[test]
fn occlusion_prediction_and_tracking_lost_keep_state_separate_from_zero() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let source_timestamp = 1_000_000_000;
    let observed = frame(&descriptor, 1, 0, 1, source_timestamp);
    let mut occluded = frame(&descriptor, 1, 0, 2, 1_100_000_000);
    occluded.rig.set(
        SignalId::new(7).unwrap(),
        SignalSample::unavailable(0.6, SignalState::Occluded, source_timestamp),
    );
    let mut predicted = frame(&descriptor, 1, 0, 3, 1_200_000_000);
    predicted.rig.set(
        SignalId::new(7).unwrap(),
        SignalSample::available(
            -0.4,
            0.4,
            SignalState::Predicted,
            source_timestamp,
            30_000_000,
        ),
    );
    let mut lost = frame(&descriptor, 1, 0, 4, 1_300_000_000);
    lost.rig.set(
        SignalId::new(7).unwrap(),
        SignalSample::unavailable(0.0, SignalState::TrackingLost, 1_300_000_000),
    );
    let report = validate_stream(
        &descriptor,
        &[observed, occluded, predicted, lost],
        ConformanceOptions::default(),
    );
    assert!(report.passed, "{report}");
}

fn yaw_quaternion(angle: f32) -> Quaternion {
    Quaternion {
        x: 0.0,
        y: (angle * 0.5).sin(),
        z: 0.0,
        w: (angle * 0.5).cos(),
    }
}

fn intrinsic_xyz_quaternion(angles: Vec3) -> Quaternion {
    let pitch = Quaternion {
        x: (angles.x * 0.5).sin(),
        y: 0.0,
        z: 0.0,
        w: (angles.x * 0.5).cos(),
    };
    let yaw = yaw_quaternion(angles.y);
    let roll = Quaternion {
        x: 0.0,
        y: 0.0,
        z: (angles.z * 0.5).sin(),
        w: (angles.z * 0.5).cos(),
    };
    multiply(multiply(pitch, yaw), roll)
}

fn multiply(left: Quaternion, right: Quaternion) -> Quaternion {
    Quaternion {
        x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
    }
}

fn normalize(value: Vec3) -> Vec3 {
    let length = (value.x * value.x + value.y * value.y + value.z * value.z).sqrt();
    Vec3 {
        x: value.x / length,
        y: value.y / length,
        z: value.z / length,
    }
}
