# FaceBasic v1 baseline

## Scope and artifact status

`FaceBasicModel` is the first executable single-frame model architecture for NTP Basic. It owns a
single lightweight encoder and six named heads: the 36-signal orthogonal rig, camera-space head
pose, auxiliary normalized landmarks, visibility state, an adversarial identity classifier, and
per-signal confidence. Geometry and identity are training/diagnostic outputs; they are not new NTP
signals and do not leak framework values into the protocol.

The checked-in `configs/face-basic-smoke.yaml` and every artifact produced from it are explicitly
smoke-only. They validate architecture, training, resume, export, runtime, and conformance control
flow. They do not demonstrate face quality, NanaLive A/B stability, RTX 4060 latency, or production
readiness. A candidate may set `export.smoke_only: false` only when it is trained from a reviewed
non-smoke manifest and accompanied by the complete real-data evaluation report.

## Input and output contract

- Input is one fixed-size NCHW float32 RGB face ROI in `[0, 1]`. The baseline smoke fixture uses
  `64x64`; a production configuration must pin the validated ROI size in its package metadata.
- A caller may supply an ROI on every frame. `FaceRoiTracker` can instead run a pluggable detector
  at a bounded interval, select the track-consistent face, add a margin, square the crop, and smooth
  coordinates. The detector model remains independently versioned rather than being hidden inside
  FaceBasic.
- Basic IDs 1 through 36 are always declared supported. Runtime visibility becomes `Observed`,
  `Occluded`, or `OutOfFrame`; supported signals are never rewritten to `Unsupported`.
- Head pose is position plus a normalized canonical xyzw quaternion in Camera space with
  HeadRelative length basis. Spatial, Full, and optional slots remain present and `Unsupported`.
- The producer publishes capture and completion timestamps. `LatestFrameRuntime` has exactly one
  pending slot and one result slot; replacing a stale frame increments `dropped_frames` instead of
  allowing result age to grow.

## Training contract

Manifest training loads only the requested identity-safe split. Complete Basic truth is required by
default. Missing pose or landmark supervision is masked, not replaced with certain zeros. The
training log records separate weighted rig, pose, landmark, visibility, confidence, identity
adversary, and mirror-consistency losses. The identity head uses gradient reversal so the encoder
is penalized for identity leakage while left/right outputs remain separate.

Structured auxiliary teacher names are:

- `head.pose.position.{x,y,z}`
- `head.pose.orientation.{x,y,z,w}`
- `face.landmark.<semantic-index>.{x,y}`

Only synchronized observed or fused values enter these losses. Basic rig labels continue to use the
versioned NTP label catalog.

## Calibration, export, and runtime verification

Level A calibration fits a per-signal median neutral plus robust negative/positive user ranges from
explicit high-confidence captures. Profiles bind to user slot, model family/version, feature,
Signal Registry, normalization, and calibration revisions. Incompatible profiles fail closed; the
base package is never modified.

The model package includes every file required by the ONNX package contract, six interoperable
fixed-vector outputs, SHA-256 digests, NTP revisions, supported signals/structures, input
normalization, precision declaration, and smoke status. Export compares every PyTorch output with
ONNX Runtime CPU before packaging; `verify-export` repeats digest and parity checks.

Useful commands:

```bash
uv run --extra cpu nana-tracking train --config configs/face-basic-smoke.yaml
uv run --extra cpu nana-tracking export --config configs/face-basic-smoke.yaml \
  --checkpoint <checkpoint.pt> --output <package-directory>
uv run --extra cpu nana-tracking verify-export --package <package-directory>
uv run --extra cpu nana-tracking calibrate-level-a --capture <calibration.npz> \
  --package <package-directory> --user-slot <slot> --output <profile.json>
uv run --extra cpu nana-tracking benchmark-face-basic --package <package-directory> \
  --providers CPUExecutionProvider --output <report.json>
uv run --extra cpu nana-tracking evaluation render-failures <failures.jsonl> \
  --output <report.html>
```

On a compatible RTX host, the benchmark provider list may be
`TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`. The report records active
providers and an `nvidia-smi` telemetry snapshot when available. Missing GPU/VRAM values remain
explicitly unavailable and are never inferred from CPU or MPS.

## Acceptance evidence boundary

Automated tests prove the following functional properties: one shared encoder invocation produces
all heads; Basic is complete; Level A calibration is versioned; training/export parity succeeds;
the latest-frame queue remains bounded; repeated fresh captures do not accumulate result age; and
the emitted diagnostic stream passes Rust `ntp-conformance` as Basic while later slots stay
Unsupported.

The following production acceptance still requires external evidence from reviewed real captures:
per-signal quality and dynamic correlation, neutral jitter, occlusion recovery, confidence
calibration, NanaLive A/B video, and an RTX 4060 FP16/TensorRT report. Synthetic smoke results must
not be substituted for those measurements.
