# FullSet v1 model and producer

Status: executable synthetic smoke baseline for Issue #9. It proves the Full-only training path,
named PyTorch heads, ONNX parity, torso-local skeleton construction, Full conformance, official
arm/auricle semantic compatibility, low-cadence scheduling, and unavailable-state behavior. It
does not prove real-user body or auricle quality, RTX 4060 performance, iOS performance, or
production readiness.

## Boundaries and revisions

- NTP schema: `ntp/1.0`
- Signal Registry: `ntp-signals/1.0.0`
- normalization: `ntp-normalization/1.0.0`
- calibration: `ntp-calibration/1.0.0`
- features: `ntp-features/1.0.0`

PyTorch is authoritative for the model and losses. The ONNX package contains only the Full-only
signals 42-76 and body structures, so the existing FaceSpatial package can run independently at a
higher cadence. ONNX Runtime arrays stop at `OrtFullSetBackend`; the producer emits plain NTP data.

## Model and training data

One depthwise shared encoder runs once per upper-body ROI. It feeds the 35 Full-only scalars,
camera-relative torso pose, torso-local shoulder/elbow/wrist poses, upper-arm and forearm
directions and twists, normalized bone lengths, five region-state classifiers, an identity
adversary, and per-signal confidence. The six internal states are Observed, PartiallyOccluded,
SelfOccluded, OutOfFrame, Predicted, and TrackingLost. NTP v1 represents the two occlusion subtypes
as `Occluded`; they remain separately trainable model outputs and are never converted to false
coordinates.

`FullSetDataset` admits identities from exactly one manifest split and requires available labels
42-76 when `require_complete_full` is enabled. Torso and body geometry are loaded from synchronized
teacher observations with confidence masks. Missing geometry remains masked. The label catalog
requires multiview torso/arm geometry and directly observed reviewed auricle/tongue truth; raw
captures remain outside Git.

## Geometry authority and cadence

`FullSetProducer` fuses a current complete Spatial result with a lower-rate body observation. It
keeps the body's original capture timestamp when reused and changes current observed samples to
`Fused`; it never rewrites an old sample as a new observation. A configurable maximum age changes
stale body results to `TrackingLost` and forces a refresh on the next visible frame. Cadence is
measured relative to the last body observation rather than absolute sequence divisibility, so a
non-zero starting sequence cannot cause back-to-back inference. Session or generation changes
invalidate the cached body sample. Leaving the frame produces `OutOfFrame`, and re-entry forces
fresh inference even if the normal body interval has not elapsed.

The skeleton is authoritative. Upper-arm and forearm directions are recomputed from the emitted
joint positions; shoulder flexion/abduction and elbow flexion are then derived from those same
directions. This prevents independent heads from creating geometrically inconsistent NTP output.
Head-relative translation and rotation are computed as
`inverse(torso_pose) * head_camera_pose`; the model's corresponding scalar head is not trusted at
the protocol boundary. Metric coordinates are never advertised for monocular output.

If only the face is visible, the descriptor remains Full, Spatial signals and structures remain
available, and every Full-only scalar and body structure is explicitly `OutOfFrame`. The producer
does not downgrade the profile or fabricate neutral body coordinates.

Advanced `armRaise`, raise azimuth, bend, proximity, body lean/twist, and auricle wiggle values are
not transmitted as additional model parameters. They are derived by `nana-tracking-semantics`
from the timestamped NTP result. The checked-in semantic golden vectors cover arm and capture-time
auricle motion formulas.

## Smoke workflow and performance

```bash
uv run --extra cpu nana-tracking train --config configs/full-set-smoke.yaml
uv run --extra cpu nana-tracking export --config configs/full-set-smoke.yaml \
  --checkpoint <checkpoint.pt> --output <package-directory>
uv run --extra cpu nana-tracking verify-export --package <package-directory>
uv run --extra cpu nana-tracking benchmark-full-set --package <package-directory> \
  --providers CPUExecutionProvider --output <runtime-report.json>
```

The checked-in Apple M4 macOS CPU smoke report used ONNX Runtime 1.27, 100 warmups, and 2,000
measured 96x96 upper-body inferences: 0.176 ms p50, 0.284 ms p95, and 0.343 ms p99. The runtime can
schedule this body package every second display frame while retaining sample age. These numbers
are fixed-input smoke evidence only, not a detector/capture pipeline benchmark or target-device
acceptance result.

- `artifacts/benchmarks/issue9-full-set-macos-m4-smoke.json`
