# NanaTracking Protocol v1 Signal Registry

- Status: **frozen for NTP v1**
- NTP schema revision: `ntp/1.0`
- Signal Registry revision: `ntp-signals/1.0.0`
- Normalization revision: `ntp-normalization/1.0.0`
- Calibration revision: `ntp-calibration/1.0.0`
- Feature revision: `ntp-features/1.0.0`

This document is the normative semantic source for NTP v1. Protocol implementations, training
labels, model heads, exporters, adapters, and consumers bind to the revisions above. Changing an
existing ID, meaning, range, unit, neutral value, coordinate space, or symmetry requires an NTP
major revision. A compatible additive signal requires a new Signal Registry revision and the
admission record defined below.

NTP describes final, framework-neutral tracking state. It never exposes a model tensor, a runtime
value, a vendor parameter dictionary, a landmark index owned by another API, or a transport type.

## Normative scalar types

The scalar type referenced by every registry row fixes its valid range, unit, neutral value, and
recommended soft limit. `1` means dimensionless normalized deformation. Calibration maps a
person's neutral state to zero and their reviewed comfortable motion range to the soft limit; it
must not redefine sign or meaning.

| Type | Valid range | Unit | Neutral | Recommended soft limit | Out-of-range rule |
| --- | ---: | --- | ---: | ---: | --- |
| `NS` | `[-1, 1]` | `1` | `0` | `[-0.95, 0.95]` | Invalid producer output; consumers may clamp only for final rig rendering. |
| `NU` | `[0, 1]` | `1` | `0` | `[0, 0.95]` | Invalid producer output; consumers may clamp only for final rig rendering. |
| `GY` | `[-1.20, 1.20]` | `rad` | `0` | `[-0.90, 0.90]` | Invalid producer output. Positive is the subject's anatomical right. |
| `GP` | `[-0.80, 0.80]` | `rad` | `0` | `[-0.60, 0.60]` | Invalid producer output. Positive is anatomically superior. |
| `TT` | `[-1, 1]` | `torso_height` | `0` | `[-0.50, 0.50]` | Invalid producer output; not metres. |
| `HT` | `[-1, 1]` | `head_width` | `0` | `[-0.75, 0.75]` | Invalid producer output; not metres. |
| `AR` | `[-pi, pi)` | `rad` | `0` | `[-pi/2, pi/2]` | Wrap only when deriving the rig view from a quaternion; wire values outside the interval are invalid. |

NaN and infinity are invalid for every type. A missing, unsupported, occluded, predicted, or lost
sample is represented by tracking state and confidence, never by an out-of-range number or a
fabricated zero. Values at a hard limit are valid but indicate saturation and should reduce the
relevant confidence when the true motion may extend beyond the calibrated range.

`Symmetry` describes anatomical reflection for validation and augmentation. `pair` swaps the named
left/right IDs without changing their semantic sign; `self` remains the same; `axial` follows the
coordinate reflection rules below. Display mirroring never performs this anatomical reflection.

## Stable scalar registry

The `Set` column gives the lowest guaranteed set. Because the sets are nested, every `Basic` row is
also in Spatial and Full, and every `Spatial` row is also in Full. Optional rows are stable v1
features but are not required by any guaranteed profile.

### BasicSet: 36 signals

| ID | Stable name | Type | Negative / zero / positive meaning | Symmetry | Set |
| ---: | --- | --- | --- | --- | --- |
| `1 (0x0001)` | `brow.left.inner_vertical` | `NS` | lower / neutral / raise inner brow | pair: `brow.right.inner_vertical` | Basic |
| `2 (0x0002)` | `brow.right.inner_vertical` | `NS` | lower / neutral / raise inner brow | pair: `brow.left.inner_vertical` | Basic |
| `3 (0x0003)` | `brow.left.outer_vertical` | `NS` | lower / neutral / raise outer brow | pair: `brow.right.outer_vertical` | Basic |
| `4 (0x0004)` | `brow.right.outer_vertical` | `NS` | lower / neutral / raise outer brow | pair: `brow.left.outer_vertical` | Basic |
| `5 (0x0005)` | `brow.left.medial` | `NS` | lateral / neutral / medial | pair: `brow.right.medial` | Basic |
| `6 (0x0006)` | `brow.right.medial` | `NS` | lateral / neutral / medial | pair: `brow.left.medial` | Basic |
| `7 (0x0007)` | `eye.left.aperture` | `NS` | close / neutral / widen eyelids | pair: `eye.right.aperture` | Basic |
| `8 (0x0008)` | `eye.right.aperture` | `NS` | close / neutral / widen eyelids | pair: `eye.left.aperture` | Basic |
| `9 (0x0009)` | `eye.left.squint` | `NU` | n/a / relaxed / periocular contraction | pair: `eye.right.squint` | Basic |
| `10 (0x000a)` | `eye.right.squint` | `NU` | n/a / relaxed / periocular contraction | pair: `eye.left.squint` | Basic |
| `11 (0x000b)` | `cheek.left.inflation` | `NS` | suck inward / neutral / puff outward | pair: `cheek.right.inflation` | Basic |
| `12 (0x000c)` | `cheek.right.inflation` | `NS` | suck inward / neutral / puff outward | pair: `cheek.left.inflation` | Basic |
| `13 (0x000d)` | `cheek.left.raise` | `NU` | n/a / neutral / raise | pair: `cheek.right.raise` | Basic |
| `14 (0x000e)` | `cheek.right.raise` | `NU` | n/a / neutral / raise | pair: `cheek.left.raise` | Basic |
| `15 (0x000f)` | `nose.left.sneer` | `NU` | n/a / neutral / sneer | pair: `nose.right.sneer` | Basic |
| `16 (0x0010)` | `nose.right.sneer` | `NU` | n/a / neutral / sneer | pair: `nose.left.sneer` | Basic |
| `17 (0x0011)` | `jaw.open` | `NU` | n/a / closed-neutral / open | self | Basic |
| `18 (0x0012)` | `jaw.lateral` | `NS` | subject-left / centered / subject-right | axial | Basic |
| `19 (0x0013)` | `jaw.protraction` | `NS` | retract / neutral / protrude | self | Basic |
| `20 (0x0014)` | `mouth.corner.left.vertical` | `NS` | lower/frown / neutral / raise/smile | pair: `mouth.corner.right.vertical` | Basic |
| `21 (0x0015)` | `mouth.corner.right.vertical` | `NS` | lower/frown / neutral / raise/smile | pair: `mouth.corner.left.vertical` | Basic |
| `22 (0x0016)` | `mouth.corner.left.horizontal` | `NS` | draw medial / neutral / stretch lateral | pair: `mouth.corner.right.horizontal` | Basic |
| `23 (0x0017)` | `mouth.corner.right.horizontal` | `NS` | draw medial / neutral / stretch lateral | pair: `mouth.corner.left.horizontal` | Basic |
| `24 (0x0018)` | `mouth.lip.upper_left.vertical` | `NS` | lower / neutral / raise | pair: `mouth.lip.upper_right.vertical` | Basic |
| `25 (0x0019)` | `mouth.lip.upper_right.vertical` | `NS` | lower / neutral / raise | pair: `mouth.lip.upper_left.vertical` | Basic |
| `26 (0x001a)` | `mouth.lip.lower_left.vertical` | `NS` | lower / neutral / raise | pair: `mouth.lip.lower_right.vertical` | Basic |
| `27 (0x001b)` | `mouth.lip.lower_right.vertical` | `NS` | lower / neutral / raise | pair: `mouth.lip.lower_left.vertical` | Basic |
| `28 (0x001c)` | `mouth.seal` | `NU` | n/a / calibrated neutral contact / additional seal | self | Basic |
| `29 (0x001d)` | `mouth.protrusion` | `NS` | retract / neutral / protrude | self | Basic |
| `30 (0x001e)` | `mouth.roundness` | `NU` | n/a / neutral / round | self | Basic |
| `31 (0x001f)` | `mouth.lip.upper_roll` | `NS` | roll inward / neutral / roll outward | self | Basic |
| `32 (0x0020)` | `mouth.lip.lower_roll` | `NS` | roll inward / neutral / roll outward | self | Basic |
| `33 (0x0021)` | `mouth.press.left` | `NU` | n/a / relaxed / press | pair: `mouth.press.right` | Basic |
| `34 (0x0022)` | `mouth.press.right` | `NU` | n/a / relaxed / press | pair: `mouth.press.left` | Basic |
| `35 (0x0023)` | `mouth.dimple.left` | `NU` | n/a / relaxed / dimple | pair: `mouth.dimple.right` | Basic |
| `36 (0x0024)` | `mouth.dimple.right` | `NU` | n/a / relaxed / dimple | pair: `mouth.dimple.left` | Basic |

### SpatialSet additions: 5 signals, 41 total

| ID | Stable name | Type | Negative / zero / positive meaning | Symmetry | Set |
| ---: | --- | --- | --- | --- | --- |
| `37 (0x0025)` | `gaze.left.yaw` | `GY` | subject-left / forward / subject-right | pair: `gaze.right.yaw` | Spatial |
| `38 (0x0026)` | `gaze.left.pitch` | `GP` | inferior / forward / superior | pair: `gaze.right.pitch` | Spatial |
| `39 (0x0027)` | `gaze.right.yaw` | `GY` | subject-left / forward / subject-right | pair: `gaze.left.yaw` | Spatial |
| `40 (0x0028)` | `gaze.right.pitch` | `GP` | inferior / forward / superior | pair: `gaze.left.pitch` | Spatial |
| `41 (0x0029)` | `tongue.extension` | `NU` | n/a / retracted-neutral / extend | self | Spatial |

### FullSet additions: 35 signals, 76 total

| ID | Stable name | Type | Negative / zero / positive meaning | Symmetry | Set |
| ---: | --- | --- | --- | --- | --- |
| `42 (0x002a)` | `torso.translation.x` | `TT` | camera-left / neutral / camera-right | axial | Full |
| `43 (0x002b)` | `torso.translation.y` | `TT` | camera-up / neutral / camera-down | axial | Full |
| `44 (0x002c)` | `torso.translation.z` | `TT` | toward camera / neutral / away from camera | axial | Full |
| `45 (0x002d)` | `torso.rotation.pitch` | `AR` | pitch inferior / neutral / pitch superior | axial | Full |
| `46 (0x002e)` | `torso.rotation.yaw` | `AR` | turn subject-left / neutral / turn subject-right | axial | Full |
| `47 (0x002f)` | `torso.rotation.roll` | `AR` | left side down / neutral / right side down | axial | Full |
| `48 (0x0030)` | `head.relative_translation.x` | `HT` | subject-left / neutral / subject-right | axial | Full |
| `49 (0x0031)` | `head.relative_translation.y` | `HT` | superior / neutral / inferior | axial | Full |
| `50 (0x0032)` | `head.relative_translation.z` | `HT` | posterior / neutral / anterior | axial | Full |
| `51 (0x0033)` | `head.relative_rotation.pitch` | `AR` | pitch inferior / neutral / pitch superior | axial | Full |
| `52 (0x0034)` | `head.relative_rotation.yaw` | `AR` | turn subject-left / neutral / turn subject-right | axial | Full |
| `53 (0x0035)` | `head.relative_rotation.roll` | `AR` | left side down / neutral / right side down | axial | Full |
| `54 (0x0036)` | `tongue.horizontal` | `NS` | subject-left / centered / subject-right | axial | Full |
| `55 (0x0037)` | `tongue.vertical` | `NS` | inferior / centered / superior | self | Full |
| `56 (0x0038)` | `tongue.curl` | `NS` | curl down / neutral / curl up | self | Full |
| `57 (0x0039)` | `auricle.left.elevation` | `NS` | lower / neutral / raise | pair: `auricle.right.elevation` | Full |
| `58 (0x003a)` | `auricle.right.elevation` | `NS` | lower / neutral / raise | pair: `auricle.left.elevation` | Full |
| `59 (0x003b)` | `auricle.left.protraction` | `NS` | retract / neutral / protract | pair: `auricle.right.protraction` | Full |
| `60 (0x003c)` | `auricle.right.protraction` | `NS` | retract / neutral / protract | pair: `auricle.left.protraction` | Full |
| `61 (0x003d)` | `auricle.left.flattening` | `NS` | flare away / neutral / flatten toward head | pair: `auricle.right.flattening` | Full |
| `62 (0x003e)` | `auricle.right.flattening` | `NS` | flare away / neutral / flatten toward head | pair: `auricle.left.flattening` | Full |
| `63 (0x003f)` | `shoulder_girdle.left.elevation` | `NS` | depress / neutral / elevate | pair: `shoulder_girdle.right.elevation` | Full |
| `64 (0x0040)` | `shoulder_girdle.right.elevation` | `NS` | depress / neutral / elevate | pair: `shoulder_girdle.left.elevation` | Full |
| `65 (0x0041)` | `shoulder_girdle.left.protraction` | `NS` | retract / neutral / protract | pair: `shoulder_girdle.right.protraction` | Full |
| `66 (0x0042)` | `shoulder_girdle.right.protraction` | `NS` | retract / neutral / protract | pair: `shoulder_girdle.left.protraction` | Full |
| `67 (0x0043)` | `arm.left.shoulder.flexion` | `NS` | extension / neutral / flexion | pair: `arm.right.shoulder.flexion` | Full |
| `68 (0x0044)` | `arm.left.shoulder.abduction` | `NS` | adduction / neutral / abduction | pair: `arm.right.shoulder.abduction` | Full |
| `69 (0x0045)` | `arm.left.shoulder.twist` | `NS` | external / neutral / internal rotation | pair: `arm.right.shoulder.twist` | Full |
| `70 (0x0046)` | `arm.left.elbow.flexion` | `NU` | n/a / extended-neutral / flexed | pair: `arm.right.elbow.flexion` | Full |
| `71 (0x0047)` | `arm.left.forearm.twist` | `NS` | supination / neutral / pronation | pair: `arm.right.forearm.twist` | Full |
| `72 (0x0048)` | `arm.right.shoulder.flexion` | `NS` | extension / neutral / flexion | pair: `arm.left.shoulder.flexion` | Full |
| `73 (0x0049)` | `arm.right.shoulder.abduction` | `NS` | adduction / neutral / abduction | pair: `arm.left.shoulder.abduction` | Full |
| `74 (0x004a)` | `arm.right.shoulder.twist` | `NS` | external / neutral / internal rotation | pair: `arm.left.shoulder.twist` | Full |
| `75 (0x004b)` | `arm.right.elbow.flexion` | `NU` | n/a / extended-neutral / flexed | pair: `arm.left.elbow.flexion` | Full |
| `76 (0x004c)` | `arm.right.forearm.twist` | `NS` | supination / neutral / pronation | pair: `arm.left.forearm.twist` | Full |

### Optional fine features: 12 signals

| ID | Stable name | Type | Negative / zero / positive meaning | Symmetry | Set |
| ---: | --- | --- | --- | --- | --- |
| `77 (0x004d)` | `nose.left.alar_flare` | `NS` | compress / neutral / flare | pair: `nose.right.alar_flare` | Optional |
| `78 (0x004e)` | `nose.right.alar_flare` | `NS` | compress / neutral / flare | pair: `nose.left.alar_flare` | Optional |
| `79 (0x004f)` | `mouth.bite.upper_lip` | `NU` | n/a / released / upper lip bitten | self | Optional |
| `80 (0x0050)` | `mouth.bite.lower_lip` | `NU` | n/a / released / lower lip bitten | self | Optional |
| `81 (0x0051)` | `auricle.left.twist` | `NS` | twist outward / neutral / twist inward | pair: `auricle.right.twist` | Optional |
| `82 (0x0052)` | `auricle.right.twist` | `NS` | twist outward / neutral / twist inward | pair: `auricle.left.twist` | Optional |
| `83 (0x0053)` | `wrist.left.flexion` | `NS` | extension / neutral / flexion | pair: `wrist.right.flexion` | Optional |
| `84 (0x0054)` | `wrist.left.deviation` | `NS` | ulnar / neutral / radial deviation | pair: `wrist.right.deviation` | Optional |
| `85 (0x0055)` | `wrist.left.twist` | `NS` | supination / neutral / pronation | pair: `wrist.right.twist` | Optional |
| `86 (0x0056)` | `wrist.right.flexion` | `NS` | extension / neutral / flexion | pair: `wrist.left.flexion` | Optional |
| `87 (0x0057)` | `wrist.right.deviation` | `NS` | ulnar / neutral / radial deviation | pair: `wrist.left.deviation` | Optional |
| `88 (0x0058)` | `wrist.right.twist` | `NS` | supination / neutral / pronation | pair: `wrist.left.twist` | Optional |

IDs `0x0001..0x0058` are permanently assigned as above. The stable-addition range starts at
`0x0100`; a registry revision must publish every assignment. `0x8000..0xffff` is experimental and
must never be emitted by a producer claiming strict v1 conformance. Removed stable IDs are retired,
not reused. Consumers ignore unknown additive IDs while preserving their known values and state.

### Scalar coordinate binding

Every row resolves its coordinate semantics by category:

| Stable-name category | Coordinate binding |
| --- | --- |
| Face, tongue, auricle, shoulder-girdle, arm, and wrist deformation rows | Anatomical scalar in the subject's calibrated local state; polarity and left/right side are fixed by the row. These are not spatial vector components. |
| `gaze.*` | Yaw/pitch of the eye direction in Head-local space `H`. |
| `torso.translation.*` | Components in Camera space `C`, relative to the calibrated session neutral pose. |
| `torso.rotation.*` | Euler Rig view of the active Torso-local-to-Camera quaternion. |
| `head.relative_translation.*` | Components of head origin in Torso-local space `T`, relative to calibrated neutral. |
| `head.relative_rotation.*` | Euler Rig view of the active Head-local-to-Torso-local quaternion. |

A calibration profile may scale normalized deformation response but cannot move a signal to another
space, swap anatomical sides, invert polarity, or reinterpret a unit.

## Orthogonality and deterministic derivation

Only the base state above consumes stable Rig IDs. Semantic actions, velocities, phases, regional
relations, and vendor/model bindings are deterministic views and do not receive IDs.

| Derived view or constraint | Normative relation |
| --- | --- |
| eyelid close / wide | `close = max(-aperture, 0)`, `wide = max(aperture, 0)` per eye. |
| jaw left / right | `left = max(-jaw.lateral, 0)`, `right = max(jaw.lateral, 0)`. They cannot both be positive. |
| cheek suck / puff | `suck = max(-inflation, 0)`, `puff = max(inflation, 0)` per cheek. |
| mouth frown / smile | `frown = max(-corner.vertical, 0)`, `smile = max(corner.vertical, 0)` per side. |
| ear pull back / forward | Negative and positive halves of `auricle.*.protraction`; never separate base IDs. |
| ear flare / flatten | Negative and positive halves of `auricle.*.flattening`; never separate base IDs. |
| torso Rig view | Scalar translation and Euler rows are a deterministic projection of the structured torso pose, not an independent estimate. |
| head-relative Rig view | `T_T_H = inverse(T_C_T) * T_C_H`; its six scalar rows are a deterministic projection of that transform. |
| gaze Rig view | Per-eye yaw/pitch is the deterministic angular projection of `EyeGeometry.direction_head`; look-at is a geometric view of the same rays, not an extra gaze freedom. |
| arm Rig view | Shoulder-girdle/shoulder/elbow/forearm rows and skeleton geometry are two views of one articulated state. The skeleton pose is authoritative when both are present; inverse-kinematic projection must reproduce the scalar view within the registry revision's tolerance. |
| arm raise / reach / proximity | Derived from arm state plus skeleton and timestamps; never transmitted as base signals. |
| motion speed / energy / phase | Derived from capture-time history and reset on generation change; never transmitted as base signals. |

Protrusion, roundness, and seal remain separate because equal protrusion can coexist with different
lip aperture/roundness and contact. Jaw opening/protraction describe the mandible while lip
vertical/seal/protrusion describe soft tissue, so neither determines the other. Squint remains
separate from aperture because periocular contraction is not determined by lid separation. Alar
flare is independent of a nose sneer, and tongue extension, two-axis displacement, and curl can
vary independently. These are observable independent freedoms, not compatibility aliases.

## Coordinate systems and rotations

All spaces are right-handed.

- Camera space `C`: origin at the optical centre, `+X` image-right, `+Y` image-down, `+Z` along the
  optical axis away from the camera.
- Torso-local space `T`: origin at the midpoint between the left and right shoulder pivots in the
  calibrated neutral pose, `+X` anatomical right, `+Y` inferior, `+Z` anterior.
- Head-local space `H`: origin at the midpoint between the calibrated eye centres, with the same
  anatomical axis directions as `T` in the neutral pose.
- A side name is always the subject's anatomical side. A preview may reflect camera-space `X` only
  at display time; it must not exchange IDs, change signs, alter stored geometry, or affect
  capability/profile checks.

Structured rotations are unit Hamilton quaternions ordered `(x, y, z, w)`. They are active
local-to-parent rotations, composed right-to-left. Producers normalize them; a norm outside
`1 +/- 1e-4` is invalid. Quaternions with `w < 0` are negated for canonical serialization; when
`w == 0`, the first non-zero component among `x,y,z` is non-negative.

Euler angles exist only as Rig views. They use intrinsic `X-Y-Z` composition: pitch about local
`+X`, then yaw about the updated `+Y`, then roll about the updated `+Z`. Positive pitch is superior,
positive yaw is anatomical right, and positive roll lowers the anatomical right side. At the Euler
singularity, conversion chooses roll `0` and preserves yaw; consumers needing interpolation or
composition must use the quaternion.

Anatomical reflection swaps left/right paired signals. It negates lateral scalar axes, camera/local
`X`, yaw, and roll; it preserves vertical/depth axes and pitch. This rule is for augmentation and
symmetry validation only, not display mirroring.

## Scale and depth

Every structured position carries a `LengthBasis`; the basis is never inferred from the numeric
value:

| Basis | Unit and rule |
| --- | --- |
| `Metric` | Metres in the declared coordinate space. Allowed only when the producer has a calibrated metric source. |
| `HeadRelative` | Dimensionless; divide physical displacement by calibrated head width. Used by normalized face/head geometry. |
| `TorsoRelative` | Dimensionless; divide physical displacement by calibrated torso height. Used by monocular body geometry. |

Monocular relative `Z` is not metres and must not use `Metric`. A consumer may compare or render
relative depth within the same session and basis but must not report real-world distance. A
producer changing basis increments `generation` and republishes its descriptor. Mixed bases inside
one structure block are invalid.

## Structured result schema draft

The following framework-neutral schema fixes semantics, not a language layout or wire codec. Fixed
slots remain present; capability and per-frame state say whether a value is usable.

```text
Position3 {
  space: Camera | TorsoLocal | HeadLocal
  length_basis: Metric | HeadRelative | TorsoRelative
  value: Vec3<f32>
}

Direction3 {
  space: Camera | TorsoLocal | HeadLocal
  value: UnitVec3<f32>
}

Pose {
  parent_space: Camera | TorsoLocal | HeadLocal
  length_basis: Metric | HeadRelative | TorsoRelative
  position: Vec3<f32>
  orientation_xyzw: Quat<f32>
}

Tracked<T> {
  value: T
  confidence: f32 in [0, 1]
  state: Observed | Fused | Predicted | Occluded |
         OutOfFrame | TrackingLost | Unsupported
  sample_capture_timestamp_ns: u64
  prediction_horizon_ns: u64
}

HeadGeometry { camera_pose: Tracked<Pose> }

EyeGeometry {
  side: Left | Right
  origin_head: Tracked<Position3>         // HeadRelative, HeadLocal
  direction_head: Tracked<Direction3>     // HeadLocal
}

FaceLandmark {
  semantic_id: stable NTP landmark semantic
  position_head: Tracked<Position3>       // HeadRelative, HeadLocal
}

FaceGeometry {
  eyes: [EyeGeometry; 2]
  look_at_camera: Tracked<Position3>      // Camera; explicit Metric or HeadRelative basis
  landmarks: stable semantic-keyed collection
}

BodySkeleton {
  torso_camera_pose: Tracked<Pose>
  shoulder: SideMap<Tracked<Pose>>
  elbow: SideMap<Tracked<Pose>>
  wrist: SideMap<Tracked<Pose>>
  upper_arm_direction_torso: SideMap<Tracked<UnitVec3<f32>>>
  forearm_direction_torso: SideMap<Tracked<UnitVec3<f32>>>
  upper_arm_twist: SideMap<Tracked<f32>>
  forearm_twist: SideMap<Tracked<f32>>
}

FrameQuality {
  overall_confidence: f32 in [0, 1]
  face: RegionState
  eyes: RegionState
  torso: RegionState
  arm: SideMap<RegionState>
  auricle: SideMap<RegionState>
}

FrameEnvelope {
  session_id: opaque stable identifier
  generation: u32
  sequence: u64
  capture_timestamp_ns: u64
  produced_timestamp_ns: u64
  prediction_horizon_ns: u64
  quality: FrameQuality
}
```

`UnitVec3` must be finite and have norm `1 +/- 1e-4`. `RegionState` uses the same state vocabulary
as `Tracked<T>` but never replaces per-signal state. Joint orientations and directions describe the
same pose and must agree. Stable landmark semantics will be assigned by a compatible registry
revision; a producer must not substitute indices from a third-party topology. Dense meshes and
auricle-local point sets are separate optional blocks with their own topology revision and cannot
replace the stable semantic landmarks.

### Structure requirements by profile

| Profile | Required structures |
| --- | --- |
| Basic | all 36 Basic signals; `HeadGeometry`; frame timing; overall quality; per-signal confidence and state |
| Spatial | all Basic requirements plus all 5 Spatial additions; both `EyeGeometry`; look-at point; normalized `FaceGeometry` |
| Full | all Spatial requirements plus all 35 Full additions; torso pose; shoulder/elbow/wrist skeleton; upper-arm/forearm directions and twists |

Optional structure feature bits are `metric_coordinates`, `dense_face_mesh`,
`auricle_local_geometry`, and `wrist_pose`. A feature bit advertises a stable structure capability;
it never changes a scalar ID's meaning.

`BasicSet` is a proper subset of `SpatialSet`, and `SpatialSet` is a proper subset of `FullSet`.
`guaranteed_profile` is the highest row whose required signals and structures are all supported. It
is a lower-bound guarantee, not a capability ceiling. Extra supported signals/structures remain in
`supported_signals` and `supported_structures`; they are never discarded to match the profile.
Current occlusion or out-of-frame state does not lower the profile.

## Time and quality semantics

- `session_id` identifies one clock and calibration domain; `generation` changes on reset,
  incompatible calibration, coordinate-basis change, or producer replacement.
- `sequence` strictly increases within `(session_id, generation)`. A result from an older
  generation is stale even if it arrives later.
- `capture_timestamp_ns` is the sensor monotonic timestamp at exposure midpoint.
- `produced_timestamp_ns` uses the same monotonic clock and is recorded after the result is final.
  It must be greater than or equal to capture time.
- `prediction_horizon_ns` is zero for state at capture time; otherwise the value describes
  `capture_timestamp_ns + prediction_horizon_ns`.
- At consumer time `now_ns`, `sample_age_ns = max(0, now_ns - capture_timestamp_ns)`. Processing
  latency is `produced_timestamp_ns - capture_timestamp_ns`; it is not prediction horizon.
- Per-signal `value`, `confidence`, and `state` are independent. Overall quality summarizes the
  frame but cannot overwrite a region's state or manufacture support.

## New-signal admission rule

Every proposed stable signal must include all of the following before assignment:

1. a physical or appearance freedom that cannot be deterministically recovered from existing
   scalar and structured state, with counterexamples showing equal existing state and different
   proposed values;
2. stable name, type/range, unit, neutral, polarity, coordinate space, symmetry, calibration, and
   quality-state behavior;
3. required profile or optional-feature placement and interaction with all existing IDs;
4. compatibility decision, registry revision, test vectors, and mappings for producers and
   consumers without exposing any framework or vendor type;
5. proof that it is not a semantic action, velocity, phase, regional relationship, duplicate
   positive/negative half-axis, joint-position alias, model binding, or backend-specific value.

Failure to establish item 1 means the proposal is a derived view and receives no Signal ID.
