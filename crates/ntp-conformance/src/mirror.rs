use nana_tracking_protocol::{
    Direction3, NanaTrackingResult, Pose, Position3, Quaternion, SignalId, SignalSample, Tracked,
    Vec3,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MirrorError {
    pub path: String,
    pub message: String,
}

/// Validates an anatomical-reflection pair. This is augmentation/certification behavior, never a
/// display-mirroring operation.
#[must_use]
pub fn validate_mirror_pair(
    original: &NanaTrackingResult,
    reflected: &NanaTrackingResult,
    tolerance: f32,
) -> Vec<MirrorError> {
    let mut errors = Vec::new();
    for &(left, right, sign) in PAIRED_SIGNALS {
        compare_signal(
            original,
            reflected,
            left,
            right,
            sign,
            tolerance,
            &mut errors,
        );
        compare_signal(
            original,
            reflected,
            right,
            left,
            sign,
            tolerance,
            &mut errors,
        );
    }
    for &raw in SELF_SIGNALS {
        compare_signal(original, reflected, raw, raw, 1.0, tolerance, &mut errors);
    }
    for &raw in NEGATED_SIGNALS {
        compare_signal(original, reflected, raw, raw, -1.0, tolerance, &mut errors);
    }

    validate_structured_mirror(original, reflected, tolerance, &mut errors);
    errors
}

fn validate_structured_mirror(
    original: &NanaTrackingResult,
    reflected: &NanaTrackingResult,
    tolerance: f32,
    errors: &mut Vec<MirrorError>,
) {
    compare_tracked_pose(
        "geometry.head_camera_pose",
        &original.geometry.head_camera_pose,
        &reflected.geometry.head_camera_pose,
        tolerance,
        errors,
    );
    compare_tracked_position(
        "geometry.look_at_camera",
        &original.geometry.look_at_camera,
        &reflected.geometry.look_at_camera,
        tolerance,
        errors,
    );
    for (name, source, target) in [
        (
            "geometry.eyes.left.origin",
            &original.geometry.eyes.left.origin_head,
            &reflected.geometry.eyes.right.origin_head,
        ),
        (
            "geometry.eyes.right.origin",
            &original.geometry.eyes.right.origin_head,
            &reflected.geometry.eyes.left.origin_head,
        ),
    ] {
        compare_tracked_position(name, source, target, tolerance, errors);
    }
    for (name, source, target) in [
        (
            "geometry.eyes.left.direction",
            &original.geometry.eyes.left.direction_head,
            &reflected.geometry.eyes.right.direction_head,
        ),
        (
            "geometry.eyes.right.direction",
            &original.geometry.eyes.right.direction_head,
            &reflected.geometry.eyes.left.direction_head,
        ),
    ] {
        compare_tracked_direction(name, source, target, tolerance, errors);
    }
    compare_tracked_pose(
        "skeleton.torso_camera_pose",
        &original.skeleton.torso_camera_pose,
        &reflected.skeleton.torso_camera_pose,
        tolerance,
        errors,
    );
    for (name, source, target) in [
        (
            "skeleton.shoulder.left",
            &original.skeleton.shoulder.left,
            &reflected.skeleton.shoulder.right,
        ),
        (
            "skeleton.shoulder.right",
            &original.skeleton.shoulder.right,
            &reflected.skeleton.shoulder.left,
        ),
        (
            "skeleton.elbow.left",
            &original.skeleton.elbow.left,
            &reflected.skeleton.elbow.right,
        ),
        (
            "skeleton.elbow.right",
            &original.skeleton.elbow.right,
            &reflected.skeleton.elbow.left,
        ),
        (
            "skeleton.wrist.left",
            &original.skeleton.wrist.left,
            &reflected.skeleton.wrist.right,
        ),
        (
            "skeleton.wrist.right",
            &original.skeleton.wrist.right,
            &reflected.skeleton.wrist.left,
        ),
    ] {
        compare_tracked_pose(name, source, target, tolerance, errors);
    }
}

const PAIRED_SIGNALS: &[(u16, u16, f32)] = &[
    (1, 2, 1.0),
    (3, 4, 1.0),
    (5, 6, 1.0),
    (7, 8, 1.0),
    (9, 10, 1.0),
    (11, 12, 1.0),
    (13, 14, 1.0),
    (15, 16, 1.0),
    (20, 21, 1.0),
    (22, 23, 1.0),
    (24, 25, 1.0),
    (26, 27, 1.0),
    (33, 34, 1.0),
    (35, 36, 1.0),
    (37, 39, -1.0),
    (38, 40, 1.0),
    (57, 58, 1.0),
    (59, 60, 1.0),
    (61, 62, 1.0),
    (63, 64, 1.0),
    (65, 66, 1.0),
    (67, 72, 1.0),
    (68, 73, 1.0),
    (69, 74, 1.0),
    (70, 75, 1.0),
    (71, 76, 1.0),
    (77, 78, 1.0),
    (81, 82, 1.0),
    (83, 86, 1.0),
    (84, 87, 1.0),
    (85, 88, 1.0),
];

const SELF_SIGNALS: &[u16] = &[
    17, 19, 28, 29, 30, 31, 32, 41, 43, 44, 45, 49, 50, 51, 55, 56, 79, 80,
];
const NEGATED_SIGNALS: &[u16] = &[18, 42, 46, 47, 48, 52, 53, 54];

fn compare_signal(
    original: &NanaTrackingResult,
    reflected: &NanaTrackingResult,
    source: u16,
    target: u16,
    sign: f32,
    tolerance: f32,
    errors: &mut Vec<MirrorError>,
) {
    let source = original
        .rig
        .get(SignalId::new(source).expect("mirror table IDs are non-zero"))
        .expect("mirror table IDs are stable");
    let target_sample = reflected
        .rig
        .get(SignalId::new(target).expect("mirror table IDs are non-zero"))
        .expect("mirror table IDs are stable");
    let path = format!("rig.{target}");
    compare_sample(&path, source, target_sample, sign, tolerance, errors);
}

fn compare_sample(
    path: &str,
    source: &SignalSample,
    target: &SignalSample,
    sign: f32,
    tolerance: f32,
    errors: &mut Vec<MirrorError>,
) {
    if source.state != target.state {
        errors.push(MirrorError {
            path: path.into(),
            message: format!(
                "state {:?} did not reflect as {:?}",
                source.state, target.state
            ),
        });
    }
    if !close(source.confidence, target.confidence, tolerance) {
        errors.push(MirrorError {
            path: path.into(),
            message: "confidence changed under anatomical reflection".into(),
        });
    }
    match (source.value, target.value) {
        (Some(source), Some(target)) if !close(source * sign, target, tolerance) => {
            errors.push(MirrorError {
                path: path.into(),
                message: format!("expected {}, got {target}", source * sign),
            });
        }
        (Some(_), None) | (None, Some(_)) => errors.push(MirrorError {
            path: path.into(),
            message: "value availability changed under anatomical reflection".into(),
        }),
        _ => {}
    }
}

fn compare_tracked_pose(
    path: &str,
    source: &Tracked<Pose>,
    target: &Tracked<Pose>,
    tolerance: f32,
    errors: &mut Vec<MirrorError>,
) {
    compare_shell(path, source, target, tolerance, errors);
    if let (Some(source), Some(target)) = (&source.value, &target.value) {
        compare_vec(
            path,
            reflect_vec(source.position),
            target.position,
            tolerance,
            errors,
        );
        let expected = reflect_quaternion(source.orientation_xyzw).canonicalized();
        let actual = target.orientation_xyzw.canonicalized();
        if !close_quaternion(expected, actual, tolerance) {
            errors.push(MirrorError {
                path: path.into(),
                message: "orientation does not follow anatomical X reflection".into(),
            });
        }
    }
}

fn compare_tracked_position(
    path: &str,
    source: &Tracked<Position3>,
    target: &Tracked<Position3>,
    tolerance: f32,
    errors: &mut Vec<MirrorError>,
) {
    compare_shell(path, source, target, tolerance, errors);
    if let (Some(source), Some(target)) = (&source.value, &target.value) {
        compare_vec(
            path,
            reflect_vec(source.value),
            target.value,
            tolerance,
            errors,
        );
    }
}

fn compare_tracked_direction(
    path: &str,
    source: &Tracked<Direction3>,
    target: &Tracked<Direction3>,
    tolerance: f32,
    errors: &mut Vec<MirrorError>,
) {
    compare_shell(path, source, target, tolerance, errors);
    if let (Some(source), Some(target)) = (&source.value, &target.value) {
        compare_vec(
            path,
            reflect_vec(source.value),
            target.value,
            tolerance,
            errors,
        );
    }
}

fn compare_shell<T>(
    path: &str,
    source: &Tracked<T>,
    target: &Tracked<T>,
    tolerance: f32,
    errors: &mut Vec<MirrorError>,
) {
    if source.state != target.state
        || source.value.is_some() != target.value.is_some()
        || !close(source.confidence, target.confidence, tolerance)
    {
        errors.push(MirrorError {
            path: path.into(),
            message: "tracked state, value availability, or confidence changed".into(),
        });
    }
}

fn compare_vec(
    path: &str,
    expected: Vec3,
    actual: Vec3,
    tolerance: f32,
    errors: &mut Vec<MirrorError>,
) {
    if !close(expected.x, actual.x, tolerance)
        || !close(expected.y, actual.y, tolerance)
        || !close(expected.z, actual.z, tolerance)
    {
        errors.push(MirrorError {
            path: path.into(),
            message: format!("expected reflected vector {expected:?}, got {actual:?}"),
        });
    }
}

const fn reflect_vec(value: Vec3) -> Vec3 {
    Vec3 {
        x: -value.x,
        y: value.y,
        z: value.z,
    }
}

const fn reflect_quaternion(value: Quaternion) -> Quaternion {
    Quaternion {
        x: value.x,
        y: -value.y,
        z: -value.z,
        w: value.w,
    }
}

fn close(left: f32, right: f32, tolerance: f32) -> bool {
    (left - right).abs() <= tolerance
}

fn close_quaternion(left: Quaternion, right: Quaternion, tolerance: f32) -> bool {
    close(left.x, right.x, tolerance)
        && close(left.y, right.y, tolerance)
        && close(left.z, right.z, tolerance)
        && close(left.w, right.w, tolerance)
}
