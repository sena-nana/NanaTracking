# FaceSpatial v1 model and producer

Status: executable synthetic smoke baseline for Issue #8. It validates the architecture, named
heads, training, ONNX parity, latest-frame runtime, NTP output, and fusion contracts. It does not
claim real RGB gaze/tongue accuracy, iOS sensor quality, or production readiness.

## Contract revisions

- NTP schema: `ntp/1.0`
- Signal Registry: `ntp-signals/1.0.0`
- normalization: `ntp-normalization/1.0.0`
- calibration: `ntp-calibration/1.0.0`
- features: `ntp-features/1.0.0`
- smoke canonical geometry topology: `ntp-face-canonical/1.0.0-smoke`

Changing any output meaning or topology requires a new revision and an explicit compatibility
decision. PyTorch owns model and loss behavior. ONNX is a deployment graph; no tensor or runtime
provider type enters NTP, C ABI, fusion, or consumer interfaces.

## Single-pass model

One shared image encoder feeds the complete 41-signal `SpatialSet`, head pose, two continuous eye
rays, a head-relative look-at vector, versioned canonical face geometry, face visibility, tongue
visibility, identity adversary, and 41 per-signal confidence values. Gaze is regressed as continuous
yaw/pitch. Tongue extension is unsigned, but the producer emits `Occluded` with no value when the
mouth interior does not support an observation.

The canonical geometry head is three-dimensional and `HeadRelative`; monocular Z is never labelled
as metres. Signal Registry 1.0 assigns no stable face-landmark semantic IDs, so model topology
points remain a versioned artifact output. The NTP producer emits the required face-geometry block
state but does not place model or vendor topology indices into `face_landmarks`. A later compatible
registry revision may assign stable semantic points.

## Runtime

`FaceSpatialProducer` maps only plain values, confidence, and tracking state to NTP. It transforms
the head-local look-at vector through the normalized head pose, emits head-relative eye origins and
unit eye directions, and declares all four Spatial structures. Slots 42 and later remain
`Unsupported` unless an upstream producer explicitly advertises them.

`LatestFrameRuntime` uses one replaceable pending slot. A stale frame is dropped before inference;
it is never queued behind newer captures. Input storage is preallocated and reused. Level A
calibration applies to the nested Basic 36 signals; gaze and geometry keep their separately
versioned normalization.

The iOS Swift boundary uses the same separation. `NTPSpatialFusionPlan` accepts only normalized NTP
results, prevalidates the union descriptor, and rejects any mismatch in session, generation,
sequence, or exact capture timestamp. Reference TrueDepth geometry/gaze priority, RGB gap filling,
confidence-margin selection, and no-averaging behavior match the Rust fusion contract.
`NTPSpatialProducer` then encodes that exact fused identity rather than assigning identity from
arrival order; raw ARKit fields stay outside the stream contract. The Swift canonical codec is
byte-compatible with the Rust golden descriptor and result vectors, and stream reconfiguration
increments generation while resetting sequence. A bounded `NTPLatestFrameWorker` gives device RGB
inference the same replaceable-slot behavior as the portable runtime.

## Smoke workflow

```bash
uv run --extra cpu nana-tracking train --config configs/face-spatial-smoke.yaml
uv run --extra cpu nana-tracking export --config configs/face-spatial-smoke.yaml \
  --checkpoint <checkpoint.pt> --output <package-directory>
uv run --extra cpu nana-tracking verify-export --package <package-directory>
uv run --extra cpu nana-tracking benchmark-face-spatial --package <package-directory> \
  --providers CPUExecutionProvider --output <runtime-report.json>
```

The checked-in configuration is synthetic and must stay `smoke_only: true`. Production evidence
requires licensed, identity-safe captures with per-head quality, confidence, occlusion, cross-device
semantics, and target-runtime measurements.

## Local smoke performance

The checked-in Issue #8 reports compare Basic and Spatial sequentially on one Apple M4 Mac mini,
using ONNX Runtime 1.27 CPU, 100 warmup iterations, and 2,000 measured fixed-ROI iterations. Spatial
capture-to-result was 0.431 ms p50, 0.941 ms p95, and 1.058 ms p99. The mean overhead versus Basic
was 22.57%. This is evidence that all extra heads remain in one low-latency pass on this host; it is
not a 720p detector benchmark, iOS result, model-quality measurement, or CUDA claim.

- `artifacts/benchmarks/issue8-face-basic-macos-m4-smoke.json`
- `artifacts/benchmarks/issue8-face-spatial-macos-m4-smoke.json`
- `artifacts/benchmarks/issue8-face-spatial-comparison-macos-m4-smoke.json`

The separate release Swift fusion microbenchmark measured a 5.431 microsecond median for one
validated 41+42-signal fusion on the same Apple M4 host. It isolates normalized-result fusion and
does not include capture, preprocessing, inference, transport, or iOS device work.

- `artifacts/benchmarks/issue8-swift-spatial-fusion-macos-m4-smoke.json`
