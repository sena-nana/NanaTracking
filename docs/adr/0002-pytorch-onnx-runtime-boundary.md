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
capture/processing-start/generation state, actual provider identity, stage timing, model metadata,
and structured backend-neutral errors. Capture and processing-start timestamps share one monotonic
clock domain, so produced time includes scheduler/mailbox wait rather than only backend execution.
Backend crates alone own sessions, devices, streams, command queues,
bindings, engine caches, and small-result readback.

`nana-tracking-runtime-ort` is the first real backend implementation. It verifies the portable
package before loading, supports the FaceBasic/Spatial/Full CPU baseline and an explicit Apple
Core ML execution-provider path, reuses one fixed NCHW preprocessing buffer, compares every named
output against the package NPZ vector, and copies only plain values into caller-owned output
storage. A Core ML session is returned only after profiling proves that Core ML executed graph
nodes; CPU fallback nodes remain visible in the framework-neutral provider status. The application
supplies and initializes its packaged ONNX
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

Real TensorRT, DirectML, native Core ML conversion, specific Apple compute-unit assignment, iOS,
and optional Burn acceptance remains device-specific. ORT does not reveal whether Core ML chose
CPU, GPU, or ANE internally, so the macOS adapter records the requested policy without claiming an
unobservable unit. CPU or synthetic evidence cannot certify those paths.

The checked-in macOS CPU measurements are
`artifacts/benchmarks/issue11-rust-ort-face-basic-macos-m4-smoke.json` and
`artifacts/benchmarks/issue11-rust-ort-spatial-full-macos-m4-smoke.json`. Their v1.1 result-age
fields include an explicit synthetic 50 ms pre-backend wait; Spatial plus Full uses sequential
processing-start timestamps so the second stage is not hidden or counted twice. These are synthetic
fixed-input smoke evidence for the Rust adapter and are not tracking-quality or target-device
acceptance.

The Rust CPU consumer long-run smoke is
`artifacts/benchmarks/issue11-rust-ort-face-basic-30m-macos-m4-smoke.json`. On Apple M4 it paced
107,997 release-mode inferences over 1,800.003 seconds at 59.998 FPS, with result-age
P50/P95/P99 0.539/0.870/0.987 ms and 0.0023 ms first-to-last P95 drift. Process CPU was 0.0338 core
equivalents, the sampled process stayed at one thread, RSS did not grow, and CPU measured zero
after dropping the session. This closes the local Rust CPU scheduling/resource evidence gap only;
the fixed input remains smoke-only and does not certify real tracking quality or another backend.

The opt-in macOS Core ML execution-provider smoke is
`artifacts/benchmarks/issue11-rust-ort-coreml-macos-m4-smoke.json`. It verifies actual profiled Core
ML nodes plus fixed-vector parity for Basic, Spatial, and Full, and reports CPU fallback. These tiny
smoke models were slower than the ORT CPU baseline, so CPU remains the measured default rather than
enabling Core ML solely because it is available.
