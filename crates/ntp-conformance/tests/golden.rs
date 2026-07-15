mod support;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use nana_tracking_protocol::{CanonicalCodec, StructureFeatures};
use nana_tracking_semantics::{
    BindingEvaluator, ModelParameterId, SemanticDeriver, SemanticId, Side, arkit_style_52_profile,
};
use ntp_conformance::{ConformanceOptions, validate_stream};
use serde::Deserialize;
use support::{descriptor, frame, set};

#[derive(Deserialize)]
struct GoldenFile {
    schema: String,
    scope: String,
    revisions: BTreeMap<String, serde_json::Value>,
    vectors: Vec<GoldenVector>,
}

#[derive(Deserialize)]
struct GoldenVector {
    name: String,
    signals: Vec<(u16, f32)>,
    expected_semantics: BTreeMap<String, f32>,
    expected_controls: BTreeMap<String, f32>,
}

#[test]
fn cross_language_json_vectors_match_reference_semantics_and_bindings() {
    let golden: GoldenFile =
        serde_json::from_str(include_str!("vectors/semantic-golden-v1.json")).unwrap();
    assert_eq!(golden.schema, "ntp-golden/1.0.0");
    assert!(golden.scope.contains("smoke"));
    assert_eq!(golden.revisions["schema"], 1);

    let descriptor = descriptor(88, StructureFeatures::BASIC_REQUIRED, &[]);
    let evaluator = BindingEvaluator::new(arkit_style_52_profile()).unwrap();
    for (index, vector) in golden.vectors.iter().enumerate() {
        let mut result = frame(
            &descriptor,
            1,
            0,
            u64::try_from(index + 1).unwrap(),
            1_000_000_000 + u64::try_from(index).unwrap() * 10_000_000,
        );
        for &(id, value) in &vector.signals {
            set(&mut result, id, value);
        }
        let report = validate_stream(
            &descriptor,
            &[result.clone()],
            ConformanceOptions::default(),
        );
        assert!(report.passed, "{}: {report}", vector.name);

        let semantics = SemanticDeriver::default()
            .derive(&result, result.produced_timestamp_ns)
            .unwrap();
        for (name, expected) in &vector.expected_semantics {
            let id = semantic_id(name).unwrap_or_else(|| panic!("unknown semantic {name}"));
            let actual = semantics.get(id).unwrap().value;
            assert_close(&vector.name, name, actual, *expected);
        }
        let controls = evaluator.evaluate(&result, &semantics).unwrap();
        for (name, expected) in &vector.expected_controls {
            let id = ModelParameterId::new(name).unwrap();
            let actual = controls[&id].value;
            assert_close(&vector.name, name, actual, *expected);
        }
    }
}

#[test]
fn canonical_binary_vectors_match_cross_language_hex_fixture() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[37]);
    let mut result = frame(&descriptor, 9, 2, 7, 1_000_000_000);
    set(&mut result, 7, -0.75);
    set(&mut result, 18, 0.25);
    set(&mut result, 29, 0.5);
    set(&mut result, 37, -0.2);
    let expected = include_str!("vectors/protocol-golden-v1.hex");
    let expected_descriptor = fixture_hex(expected, "descriptor");
    let expected_result = fixture_hex(expected, "result");
    let actual_descriptor = hex(&CanonicalCodec::encode(&descriptor).unwrap());
    let actual_result = hex(&CanonicalCodec::encode(&result).unwrap());
    assert_eq!(
        actual_descriptor, expected_descriptor,
        "descriptor={actual_descriptor}"
    );
    assert_eq!(actual_result, expected_result, "result={actual_result}");
}

fn assert_close(vector: &str, field: &str, actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1.0e-5,
        "{vector}.{field}: {actual} != {expected}"
    );
}

fn fixture_hex(contents: &str, key: &str) -> String {
    contents
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap()
        .to_string()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").unwrap();
        output
    })
}

fn semantic_id(name: &str) -> Option<SemanticId> {
    Some(match name {
        "eye_blink_left" => SemanticId::EyeBlink(Side::Left),
        "eye_blink_right" => SemanticId::EyeBlink(Side::Right),
        "eye_wide_left" => SemanticId::EyeWide(Side::Left),
        "eye_wide_right" => SemanticId::EyeWide(Side::Right),
        "jaw_left" => SemanticId::JawLeft,
        "jaw_right" => SemanticId::JawRight,
        "mouth_smile_left" => SemanticId::MouthSmile(Side::Left),
        "mouth_frown_right" => SemanticId::MouthFrown(Side::Right),
        "mouth_pucker" => SemanticId::MouthPucker,
        "mouth_funnel" => SemanticId::MouthFunnel,
        "body_lean_forward" => SemanticId::BodyLeanForward,
        "body_twist_right" => SemanticId::BodyTwistRight,
        "arm_raise_left" => SemanticId::ArmRaise(Side::Left),
        "arm_forward_weight_left" => SemanticId::ArmForwardWeight(Side::Left),
        "arm_side_weight_left" => SemanticId::ArmSideWeight(Side::Left),
        "arm_backward_weight_left" => SemanticId::ArmBackwardWeight(Side::Left),
        "arm_bend_left" => SemanticId::ArmBend(Side::Left),
        "arm_extension_left" => SemanticId::ArmExtension(Side::Left),
        "auricle_raise_left" => SemanticId::AuricleRaise(Side::Left),
        "auricle_pull_back_left" => SemanticId::AuriclePullBack(Side::Left),
        "auricle_pull_forward_left" => SemanticId::AuriclePullForward(Side::Left),
        "auricle_flatten_left" => SemanticId::AuricleFlatten(Side::Left),
        "auricle_flare_left" => SemanticId::AuricleFlare(Side::Left),
        _ => return None,
    })
}
