# NTP v1 semantic derivation and rig binding reference

- NTP schema: `ntp/1.0`, schema revision `1`
- Signal Registry: `ntp-signals/1.0.0`
- Normalization: `ntp-normalization/1.0.0`
- Calibration: `ntp-calibration/1.0.0`
- Feature contract: `ntp-features/1.0.0`
- Semantic formulas and default configuration: `ntp-semantics/1.0.0`

`nana-tracking-semantics` is the executable reference for these formulas. It reads
`NanaTrackingResult`; none of the values below is an NTP Signal ID, result field, capability bit,
or extra network parameter. PyTorch, ONNX Runtime, and vendor SDK types do not enter its public
contracts.

The 70 official derived IDs use fixed contiguous slots inside each semantic frame. This keeps the
tracking hot path to one contiguous allocation with constant-time lookup; the slots are local
library representation and are not protocol IDs or wire fields.

## State, confidence, and time

Each derived sample independently carries a value, confidence, tracking state, source capture
timestamp, and sample age. The value is never re-filtered. `Observed` retains source confidence,
`Fused` multiplies it by `0.95`, and `Predicted` multiplies it by `0.65`. During `Occluded`, the
last capture-time value may be held with confidence limited by the occlusion confidence and then
multiplied by `0.35`. Other unavailable states produce no derived value.

History is scoped to `(session_id, generation)` and is cleared when either changes. Capture time,
not receive or evaluation spacing, determines auricle velocity. Frames whose capture timestamp
moves backwards within a generation are rejected. `evaluation_timestamp_ns` and the complete
`SemanticConfig` are carried in `SemanticFrame`, so age and non-default projection constants are
explicit rather than silently changing `ntp-semantics/1.0.0` behavior.

## Deterministic formulas

For a signed scalar `x`, `negative(x) = max(-x, 0)` and
`positive(x) = max(x, 0)`. This supplies eye blink/wide, jaw left/right, cheek suck/puff,
mouth frown/smile, ear pull-back/pull-forward, and ear flare/flatten. Mouth stretch is the positive
half of each mouth-corner horizontal axis. Body forward/back lean, lateral lean, and twist are the
corresponding half-axes of torso pitch, roll, and yaw.

Let `p = positive(mouth.protrusion)`, `r = clamp01(mouth.roundness)`, and
`s = clamp01(mouth.seal)`:

```text
mouthPucker = p * (1 - r) * (0.5 + 0.5 * s)
mouthFunnel = p * r * (1 - s)
mouthInteriorVisible = jaw.open * (1 - s)
upper/lowerTeethVisible = mouthInteriorVisible * (1 - correspondingLipBite)
tongueVisible = tongue.extension * (0.5 + 0.5 * mouthInteriorVisible)
```

Missing optional lip-bite values mean released lips; when present, their confidence/state
participates in the visibility result.

Arm skeleton geometry is authoritative when available. With torso-local `+Y` inferior and `+Z`
anterior, raise uses shoulder-to-wrist height and upward upper-arm direction, reach uses wrist
`+Z`, and left/right cross-body use wrist displacement toward anatomical right/left respectively.
The default full-scale torso-relative distances are `0.45` for raise/reach and `0.30` for crossing.
Elbow bend is `(1 - dot(upperArmDirection, forearmDirection)) / 2`; extension is `1 - bend`.
Raise azimuth and forward/side/back weights use the authoritative upper-arm direction when present,
otherwise shoulder flexion/abduction. Hand proximity uses explicit default torso-local face/chest
centres and a `0.24` torso-height radius from `SemanticConfig`.

Auricle amplitude is the RMS magnitude of elevation, protraction, and flattening. Velocity is the
three-axis capture-time delta per second divided by the default full-scale velocity `4.0`; energy
is `amplitude * velocity`; phase is `atan2(protraction, elevation) / pi`. No `tension` value is
defined because monocular RGB does not uniquely determine it.

## Declarative rig binding

`RigBinding` contains a `SignalExpression`, model `ModelParameterId`, `BindingTransform`, and
explicit `CombineMode`. Expressions may read an orthogonal NTP signal, read a derived semantic,
split a sign, or combine sources. Transforms apply deadzone, curve, scale, inversion, offset, and
clamp in that order. Curves include linear, smoothstep, power, and validated piecewise-linear forms.

`BindingEvaluator` validates a profile before use. The same source may drive multiple targets. A
target with multiple sources must use one non-replace combine mode, may not repeat a source, and
may never be owned by both the orthogonal and compatibility layers. Required and preferred NTP
signals resolve separately against `NanaTrackingDescriptor`, allowing a low-DOF model to accept a
Basic producer while reporting unavailable refinements.

Built-in profiles are:

- Nana Native: all 88 stable orthogonal names with their original signed/unsigned/angle ranges;
- ARKit-style 52: all 52 common model parameter names derived without protocol aliases;
- Live2D Common: bilateral and body signals explicitly merged for low-DOF parameters;
- VTube Studio Common: the same declared Live2D target family under a distinct profile name.

The fixed vector in `crates/nana-tracking-semantics/tests/vectors/semantic-golden-v1.json` and the
contract tests verify that identical NTP state produces identical semantic and model results.
