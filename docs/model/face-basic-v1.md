# FaceBasic v1 baseline

## Scope and artifact status

`FaceBasicModel` is the first executable single-frame model architecture for NTP Basic. It owns a
single lightweight encoder and six named heads: the 36-signal orthogonal rig, camera-space head
pose, auxiliary normalized landmarks, visibility state, an adversarial identity classifier, and
per-signal confidence. Geometry and identity are training/diagnostic outputs; they are not new NTP
signals and do not leak framework values into the protocol. The identity adversary is excluded
from deployable ONNX graphs so dataset identity classes cannot cross the training boundary.

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
- `RgbRoiWorkspace` is shared by the Basic, Spatial, and Full Python producers. It caches sampling
  indices between ROI refreshes and reuses one output-row scratch buffer, so a moving ROI or camera
  resolution change does not allocate an intermediate image-sized tensor on every frame.
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

The model package includes every file required by the ONNX package contract, five interoperable
deployment outputs, SHA-256 digests, NTP revisions, supported signals/structures, input
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
uv run --extra cpu nana-tracking benchmark-face-stability --package <package-directory> \
  --providers CPUExecutionProvider --duration-seconds 1800 --target-fps 60 \
  --output <stability-report.json>
uv run --extra cpu nana-tracking benchmark-roi-preprocess \
  --output artifacts/benchmarks/issue7-roi-preprocess-macos-m4-smoke.json
uv run --extra cpu nana-tracking evaluation render-failures <failures.jsonl> \
  --output <report.html>
uv run --extra cpu nana-tracking evaluation validate-face-basic-acceptance \
  <acceptance-bundle.json>
```

On a compatible RTX host, the benchmark provider list may be
`TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider` together with
`--tensorrt-fp16`. The report records active
providers and an `nvidia-smi` telemetry snapshot when available. Missing GPU/VRAM values remain
explicitly unavailable and are never inferred from CPU or MPS.

The checked-in moving-ROI preprocessing smoke uses a reused 720p RGB source, a 640-pixel square ROI
with 32 positions held for five frames each, and 5,000 measured iterations per model input size. On
the local Apple M4, the 64/96/128 inputs measured P50 0.089/0.138/0.193 ms and P99
0.118/0.183/0.255 ms. Persistent workspace was 3.2/4.8/6.4 KiB; the separately traced steady call
peak stayed below 4.7 KiB, rather than creating an ROI- or frame-sized temporary. See
`artifacts/benchmarks/issue7-roi-preprocess-macos-m4-smoke.json`. This is synthetic preprocessing
evidence only, not camera, model-quality, GPU, or 720p60 end-to-end acceptance.

The stability command runs the actual packaged ORT face backend on a 60 FPS paced latest-capture
schedule for 30 minutes by default. Overdue capture periods are skipped instead of queued. Latency
uses a fixed-capacity deterministic reservoir plus bounded first/last windows; process RSS and
thread count are sampled once per minute. The gate reports delivered cadence, capture-to-result,
result-age P50/P95/P99 and P95 drift, CPU core equivalents, RSS/thread growth, provider state, Git
state, and lock digest. Fixed test-vector input remains smoke-only and cannot replace camera or
tracking-quality acceptance. For non-CPU providers, session registration alone is not node-assignment
evidence and must be paired with that backend's fixed-vector/profile gate.

Paced tracking sessions use one intra-op thread, sequential graph execution, and disable ORT
intra/inter-op idle spinning by default. `--allow-spinning` is an explicit throughput experiment,
not the low-latency default; adopt it only when the target-workload report proves that its latency
gain justifies the measured idle CPU cost.

The local Apple M4 comparison used the same smoke-only package, ORT CPU provider, 60 FPS pacing,
and 30-minute duration. The pre-policy run at commit `0670a09` used ORT's automatic worker policy:
it delivered 59.974 FPS but retained five threads and consumed 2.115 CPU core equivalents. The
one-thread, no-spinning default at commit `1a2ec65` delivered 59.984 FPS over 107,971 frames with
0.104 CPU core equivalents and one thread, a 95.07% CPU reduction. Result-age P50/P95/P99 was
1.686/2.782/3.128 ms, first-to-last P95 drift was 0.781 ms, 29 overdue capture periods were skipped,
RSS did not grow, and all six stability gates passed. The bounded sampler observed 107,971 results
while retaining at most 73,728 values. See
`artifacts/benchmarks/issue7-ort-face-basic-30m-macos-m4-smoke.json` for the pre-policy run and
`artifacts/benchmarks/issue7-ort-face-basic-30m-low-cpu-macos-m4-smoke.json` for the adopted default.
Both use a fixed package test vector and therefore prove scheduling/resource behavior on this host,
not camera input, tracking quality, target GPU performance, or production readiness.

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

The acceptance command digest-pins the package metadata, NTP conformance, quality report, runtime
report, and baseline reports. It fails unless all required metrics are measured on the same
checkpoint/data revision, NanaLive A/B evidence exists, and the runtime report identifies an
actual RTX 4060 with TensorRT FP16 active. Maxine alone may be explicitly unavailable when its
license or runtime cannot be approved; the other named baselines must be measured.
