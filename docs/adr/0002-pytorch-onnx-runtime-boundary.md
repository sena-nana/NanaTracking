# ADR 0002: PyTorch authority and portable runtime boundary

Status: accepted

## Decision

PyTorch is the only authoritative implementation of NanaTracking model definitions, training
losses, optimizer behavior, checkpoints, resume, and offline adapter training. ONNX is a generated
deployment graph, never a protocol or training source. ONNX Runtime is the correctness baseline;
platform backends may replace execution only after fixed-vector parity and target-device latency
and resource evidence.

NTP, Signal Registry, conformance, semantics, fusion, and the stable runtime API expose no
PyTorch tensor, ONNX Runtime value, TensorRT binding, Core ML object, Burn tensor, or Candle type.
`nana-tracking-runtime-api` owns borrowed byte inputs, caller-owned numeric output storage,
capture/generation state, actual provider identity, stage timing, model metadata, and structured
backend-neutral errors. Backend crates alone own sessions, devices, streams, command queues,
bindings, engine caches, and small-result readback.

`nana-tracking-runtime-ort` is the first real backend implementation. It verifies the portable
package before loading, supports the FaceBasic CPU baseline, reuses one fixed NCHW preprocessing
buffer, compares every named output against the package NPZ vector, and copies only plain values
into caller-owned output storage. The application supplies and initializes its packaged ONNX
Runtime dynamic library; the backend never discovers a Python environment or downloads a runtime
at application startup. The runtime API represents value, confidence, state, capture timestamp,
and prediction horizon independently, so unavailable observations never require a fabricated zero.

Portable packages use the versioned `nana-model-package/2.0.0` contract and contain a
digest-verified `model.onnx`, schema and normalization contracts,
runtime metadata, calibration and adapter contracts, and fixed vectors. Export rejects custom ONNX
operator domains. Metadata records every required standard operator, fixed/dynamic dimensions,
profile, signals, structures, features, temporal compatibility, precision, allowed runtime family,
and mode behavior. A consumer can validate a package without Python or PyTorch before selecting a
backend.

Level A is numerical and independently resettable. Level B freezes the base model and exports a
user-bound residual ONNX adapter with separate data/base digests. Optional Level C may use Burn
only behind a non-default independent adapter feature; it may not train or represent the main
model and never creates a Burn-to-ONNX build dependency. Candle remains outside production and
release gates.

## Backend policy

- NVIDIA Windows/Linux: ORT TensorRT FP16, then ORT CUDA, then measured CPU fallback.
- General Windows GPU: ORT DirectML, with actual provider reported.
- macOS/iOS: Core ML with actual compute units reported.
- CPU: enabled only for explicitly measured model/input/cadence combinations.

Unavailable providers fail structurally or use an explicitly reported compatible fallback.
Fallback never changes signal normalization, state, confidence, or revisions. TensorRT engine
caches must bind model digest, runtime/provider version, device compatibility, precision, and
shape contract before reuse.

## Consequences

Consumer applications do not install Python, PyTorch, or training dependencies. Base model and
user profiles remain separately versioned and resettable. A feature contract change invalidates
learned adapters by default; a patch-compatible base model may reuse them only after validation.
Platform optimization work stays local to backend crates and cannot make protocol or consumer
interfaces backend-specific.

Real TensorRT, DirectML, Core ML, Metal, and optional Burn acceptance remains device-specific.
CPU or synthetic evidence cannot certify those paths.

The checked-in macOS CPU measurement is
`artifacts/benchmarks/issue11-rust-ort-face-basic-macos-m4-smoke.json`. It is synthetic fixed-input
smoke evidence for the Rust adapter and is not tracking-quality or target-device acceptance.
