use nana_tracking_protocol::{SignalId, SignalMetadata};

use crate::{
    BindingLayer, BindingProfile, BindingTransform, CombineMode, LayeredBinding, ModelParameterId,
    RigBinding, SemanticId, Side, SignalExpression, SignalRequirements, signal_id,
};

/// One-to-one orthogonal profile for precise Nana rigs. Every stable NTP scalar keeps its sign,
/// side, and range; consumers may use the derived semantic frame for additional non-wire controls.
#[must_use]
pub fn nana_native_profile() -> BindingProfile {
    let bindings = SignalMetadata::all()
        .map(|metadata| {
            let (clamp_min, clamp_max) = metadata.scalar_type.valid_range();
            layered(
                BindingLayer::Orthogonal,
                metadata.stable_name,
                SignalExpression::Ntp(metadata.id),
                BindingTransform {
                    clamp_min,
                    clamp_max,
                    ..BindingTransform::default()
                },
            )
        })
        .collect();
    BindingProfile {
        name: "Nana Native Rig 1.0".into(),
        requirements: SignalRequirements {
            required_signals: ids(1..=36),
            preferred_signals: ids(37..=88),
        },
        bindings,
    }
}

/// Complete ARKit-style 52-name model profile. Names are model targets, never protocol fields.
#[must_use]
pub fn arkit_style_52_profile() -> BindingProfile {
    use SemanticId as S;
    use SignalExpression as E;

    let ntp = |raw| E::Ntp(signal_id(raw));
    let positive = |raw| E::positive(ntp(raw));
    let negative = |raw| E::negative(ntp(raw));
    let semantic = E::Semantic;
    let average = E::Average;
    let bindings = vec![
        compatibility("browDownLeft", negative(1)),
        compatibility("browDownRight", negative(2)),
        compatibility("browInnerUp", average(vec![positive(1), positive(2)])),
        compatibility("browOuterUpLeft", positive(3)),
        compatibility("browOuterUpRight", positive(4)),
        compatibility(
            "cheekPuff",
            average(vec![
                semantic(S::CheekPuff(Side::Left)),
                semantic(S::CheekPuff(Side::Right)),
            ]),
        ),
        compatibility("cheekSquintLeft", ntp(13)),
        compatibility("cheekSquintRight", ntp(14)),
        compatibility("eyeBlinkLeft", semantic(S::EyeBlink(Side::Left))),
        compatibility("eyeBlinkRight", semantic(S::EyeBlink(Side::Right))),
        compatibility("eyeLookDownLeft", negative(38)),
        compatibility("eyeLookDownRight", negative(40)),
        compatibility("eyeLookInLeft", positive(37)),
        compatibility("eyeLookInRight", negative(39)),
        compatibility("eyeLookOutLeft", negative(37)),
        compatibility("eyeLookOutRight", positive(39)),
        compatibility("eyeLookUpLeft", positive(38)),
        compatibility("eyeLookUpRight", positive(40)),
        compatibility("eyeSquintLeft", ntp(9)),
        compatibility("eyeSquintRight", ntp(10)),
        compatibility("eyeWideLeft", semantic(S::EyeWide(Side::Left))),
        compatibility("eyeWideRight", semantic(S::EyeWide(Side::Right))),
        compatibility("jawForward", positive(19)),
        compatibility("jawLeft", semantic(S::JawLeft)),
        compatibility("jawOpen", ntp(17)),
        compatibility("jawRight", semantic(S::JawRight)),
        compatibility("mouthClose", ntp(28)),
        compatibility("mouthDimpleLeft", ntp(35)),
        compatibility("mouthDimpleRight", ntp(36)),
        compatibility("mouthFrownLeft", semantic(S::MouthFrown(Side::Left))),
        compatibility("mouthFrownRight", semantic(S::MouthFrown(Side::Right))),
        compatibility("mouthFunnel", semantic(S::MouthFunnel)),
        compatibility("mouthLeft", semantic(S::JawLeft)),
        compatibility("mouthLowerDownLeft", negative(26)),
        compatibility("mouthLowerDownRight", negative(27)),
        compatibility("mouthPressLeft", ntp(33)),
        compatibility("mouthPressRight", ntp(34)),
        compatibility("mouthPucker", semantic(S::MouthPucker)),
        compatibility("mouthRight", semantic(S::JawRight)),
        compatibility("mouthRollLower", positive(32)),
        compatibility("mouthRollUpper", positive(31)),
        compatibility(
            "mouthShrugLower",
            E::positive(average(vec![ntp(26), ntp(27)])),
        ),
        compatibility(
            "mouthShrugUpper",
            E::positive(average(vec![ntp(24), ntp(25)])),
        ),
        compatibility("mouthSmileLeft", semantic(S::MouthSmile(Side::Left))),
        compatibility("mouthSmileRight", semantic(S::MouthSmile(Side::Right))),
        compatibility("mouthStretchLeft", semantic(S::MouthStretch(Side::Left))),
        compatibility("mouthStretchRight", semantic(S::MouthStretch(Side::Right))),
        compatibility("mouthUpperUpLeft", positive(24)),
        compatibility("mouthUpperUpRight", positive(25)),
        compatibility("noseSneerLeft", ntp(15)),
        compatibility("noseSneerRight", ntp(16)),
        compatibility("tongueOut", ntp(41)),
    ];
    debug_assert_eq!(bindings.len(), 52);
    BindingProfile {
        name: "ARKit-style 52 1.0".into(),
        requirements: SignalRequirements {
            required_signals: ids(1..=36),
            preferred_signals: ids(37..=41),
        },
        bindings,
    }
}

/// Common `Live2D` parameter names. Bilateral Nana signals are explicitly merged for low-DOF rigs.
#[must_use]
pub fn live2d_common_profile() -> BindingProfile {
    use SemanticId as S;
    use SignalExpression as E;
    let ntp = |raw| E::Ntp(signal_id(raw));
    let semantic = E::Semantic;
    let average = E::Average;
    let signed = BindingTransform {
        clamp_min: -1.0,
        clamp_max: 1.0,
        ..BindingTransform::default()
    };
    let eye_open = BindingTransform {
        invert: true,
        offset: 1.0,
        ..BindingTransform::default()
    };
    let mut bindings = vec![
        layered(
            BindingLayer::Compatibility,
            "ParamEyeLOpen",
            semantic(S::EyeBlink(Side::Left)),
            eye_open.clone(),
        ),
        layered(
            BindingLayer::Compatibility,
            "ParamEyeROpen",
            semantic(S::EyeBlink(Side::Right)),
            eye_open,
        ),
        layered(
            BindingLayer::Compatibility,
            "ParamEyeBallX",
            average(vec![ntp(37), ntp(39)]),
            signed.clone(),
        ),
        layered(
            BindingLayer::Compatibility,
            "ParamEyeBallY",
            average(vec![ntp(38), ntp(40)]),
            signed.clone(),
        ),
        layered(
            BindingLayer::Compatibility,
            "ParamBrowLY",
            average(vec![ntp(1), ntp(3)]),
            signed.clone(),
        ),
        layered(
            BindingLayer::Compatibility,
            "ParamBrowRY",
            average(vec![ntp(2), ntp(4)]),
            signed.clone(),
        ),
        compatibility("ParamMouthOpenY", ntp(17)),
        layered(
            BindingLayer::Compatibility,
            "ParamMouthForm",
            average(vec![ntp(20), ntp(21)]),
            signed.clone(),
        ),
        layered(
            BindingLayer::Compatibility,
            "ParamShoulderY",
            average(vec![ntp(63), ntp(64)]),
            signed.clone(),
        ),
    ];
    bindings.extend(live2d_angle_bindings());
    BindingProfile {
        name: "Live2D Common 1.1".into(),
        requirements: SignalRequirements {
            required_signals: ids(1..=36),
            preferred_signals: ids(37..=76),
        },
        bindings,
    }
}

fn live2d_angle_bindings() -> Vec<LayeredBinding> {
    [
        ("ParamAngleX", 52, -30.0, 30.0),
        ("ParamAngleY", 51, -30.0, 30.0),
        ("ParamAngleZ", 53, -30.0, 30.0),
        ("ParamBodyAngleX", 46, -10.0, 10.0),
        ("ParamBodyAngleY", 45, -10.0, 10.0),
        ("ParamBodyAngleZ", 47, -10.0, 10.0),
    ]
    .into_iter()
    .map(|(target, raw, clamp_min, clamp_max)| {
        layered(
            BindingLayer::Compatibility,
            target,
            SignalExpression::Ntp(signal_id(raw)),
            BindingTransform {
                clamp_min,
                clamp_max,
                scale: 180.0 / core::f32::consts::PI,
                ..BindingTransform::default()
            },
        )
    })
    .collect()
}

/// `VTube Studio` common `Live2D` targets use the same declarative mappings.
#[must_use]
pub fn vtube_studio_common_profile() -> BindingProfile {
    let mut profile = live2d_common_profile();
    profile.name = "VTube Studio Common 1.1".into();
    profile
}

fn compatibility(target: &str, source: SignalExpression) -> LayeredBinding {
    layered(
        BindingLayer::Compatibility,
        target,
        source,
        BindingTransform::default(),
    )
}

fn layered(
    layer: BindingLayer,
    target: &str,
    source: SignalExpression,
    transform: BindingTransform,
) -> LayeredBinding {
    LayeredBinding {
        layer,
        binding: RigBinding {
            source,
            target: ModelParameterId::new(target).expect("built-in profile target is valid"),
            transform,
            combine: CombineMode::Replace,
        },
    }
}

fn ids(range: impl IntoIterator<Item = u16>) -> Vec<SignalId> {
    range.into_iter().map(signal_id).collect()
}
