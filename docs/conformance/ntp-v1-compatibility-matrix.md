# NTP v1 conformance and compatibility matrix

Status: executable contract baseline for `ntp-conformance/1.0.0`.

All checked-in inputs are synthetic contract-only smoke vectors. They prove protocol and semantic
behavior, not FaceBasic quality, target-device latency, CUDA performance, or production readiness.

## Revision compatibility

| Protocol | Schema | Signal Registry | Normalization | Calibration | Features | Semantics | Validator |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `1.0` | `1` | `1.0.0` | `1.0.0` | `1.0.0` | `1.0.0` | `1.0.0` | `ntp-conformance/1.0.0` |

The validator accepts compatible minor/additive revisions under the protocol crate's existing
rules and rejects a different protocol, registry, normalization, calibration, or feature major
revision. Published IDs and meanings are never silently reused.

## Producer capability and certified profile

| Producer fixture | Complete required set | Extra advertised capability | Certified profile | Executable evidence |
| --- | --- | --- | --- | --- |
| NVIDIA/MediaPipe-style monocular face | Basic 36 + head geometry | left gaze yaw | Basic | `acceptance::basic_producer_keeps_extra_spatial_capability` |
| ARKit/TrueDepth-style spatial face | Spatial 41 + head/eye/look-at/face structures | bilateral auricle elevation | Spatial | `acceptance::spatial_producer_keeps_extra_full_signals` |
| Full articulated producer | Full 76 + all required structures | none required | Full while arms are `OutOfFrame` | `acceptance::full_profile_survives_arms_out_of_frame` |

Profile is the highest complete nested set. Extra signals and structures remain advertised; a
frame's `Occluded`, `OutOfFrame`, `Predicted`, or `TrackingLost` state never lowers it. A declared
capability emitting `Unsupported` fails certification.

## Consumer model requirements

| Binding profile | Required signals | Preferred signals | Basic producer | Spatial producer | Full producer |
| --- | --- | --- | --- | --- | --- |
| Nana Native Rig 1.0 | 1-36 | 37-88 | compatible, degraded optional coverage | compatible, optional Full gaps | compatible through 76, optional 77-88 depend on capability |
| ARKit-style 52 1.0 | 1-36 | 37-41 | compatible, gaze/tongue controls depend on extras | complete preferred coverage | complete preferred coverage |
| Live2D Common 1.0 / VTube Studio Common 1.0 | 1-36 | 37-76 | compatible, body controls degrade | compatible, body controls degrade | complete preferred coverage |

Compatibility is resolved from `supported_signals`, never from a producer or backend name.

## Feature bits

| Feature bit | Required structure | Certification behavior |
| --- | --- | --- |
| `metric_coordinates` | at least one structure | rejected without a structured capability |
| `dense_face_mesh` | face geometry | rejected without face geometry |
| `auricle_local_geometry` | additive stable feature | preserved as advertised; no scalar meaning changes |
| `wrist_pose` | body skeleton | rejected without body skeleton |

## Golden and language round trips

| Representation / boundary | Status | Evidence |
| --- | --- | --- |
| Rust canonical binary encode/decode | verified | `protocol-golden-v1.hex`, protocol codec tests, conformance CLI binary test |
| Rust diagnostic JSON/JSONL | verified | conformance CLI JSONL test |
| C ABI core descriptor/result conversion | verified | `nana-tracking-protocol/tests/codec.rs` |
| Independent language implementation | fixture-ready, not claimed verified | language-neutral JSON expected values and canonical hex bytes are checked in for consumers |

The canonical binary is the protocol. JSONL is an explicit diagnostic/interchange harness and does
not become a second wire contract.

## Golden vector coverage

`semantic-golden-v1.json` covers neutral, bilateral independent blink/wide, asymmetric brows,
smile/frown, pucker/funnel, jaw lateral motion, continuous binocular gaze, torso/head counter
rotation, arm raise/bend weights, and auricle orthogonality. Rust structure/temporal vectors add
hand-near-face geometry, periodic auricle motion, occlusion, out-of-frame, prediction,
`TrackingLost`, generation switches, and late-frame rejection. Anatomical reflection has a
separate fixed-rule test for side swaps and lateral-axis negation.

