mod support;

use nana_tracking_protocol::{Quaternion, StructureFeatures};
use ntp_conformance::validate_mirror_pair;
use support::{descriptor, frame, set};

#[test]
fn anatomical_reflection_swaps_sides_and_negates_lateral_axes() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[37, 38, 39, 40]);
    let mut original = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    let mut reflected = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    set(&mut original, 7, -0.8);
    set(&mut original, 18, 0.4);
    set(&mut original, 37, 0.2);
    set(&mut original, 39, -0.1);
    set(&mut reflected, 8, -0.8);
    set(&mut reflected, 18, -0.4);
    set(&mut reflected, 37, 0.1);
    set(&mut reflected, 39, -0.2);
    set(&mut reflected, 7, 0.0);
    original
        .geometry
        .head_camera_pose
        .value
        .as_mut()
        .unwrap()
        .orientation_xyzw = yaw_quaternion(0.3);
    reflected
        .geometry
        .head_camera_pose
        .value
        .as_mut()
        .unwrap()
        .orientation_xyzw = yaw_quaternion(-0.3);
    assert!(validate_mirror_pair(&original, &reflected, 1.0e-6).is_empty());

    set(&mut reflected, 18, 0.4);
    let errors = validate_mirror_pair(&original, &reflected, 1.0e-6);
    assert!(errors.iter().any(|error| error.path == "rig.18"));
}

fn yaw_quaternion(angle: f32) -> Quaternion {
    Quaternion {
        x: 0.0,
        y: (angle * 0.5).sin(),
        z: 0.0,
        w: (angle * 0.5).cos(),
    }
}
