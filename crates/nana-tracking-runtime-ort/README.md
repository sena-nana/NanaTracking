# `NanaTracking` ONNX Runtime backend

This crate is the real ORT CPU baseline and Apple Core ML execution-provider implementation for
verified `FaceBasic`, `FaceSpatial`, and Full-only model packages. Face packages implement
`TrackingModelSession`; the Full-only package is
fused into a same-capture Spatial output because its head-relative and tongue fields are not
semantically complete in isolation. The crate keeps ORT tensors and session values private, reuses
one NCHW input buffer per session, writes only framework-neutral values into caller-owned output
storage, and reports the actual provider and stage timings.

CPU and Core ML session options default to one sequential intra/inter-op thread with ORT idle
spinning disabled. Callers may opt into spinning explicitly only when target-workload evidence
justifies the extra between-frame CPU residency.

The benchmark-face-basic-stability example is the release-mode, paced Rust consumer stability
harness. It runs the verified CPU session for 30 minutes at 60 FPS by default, skips overdue
captures instead of queueing them, bounds latency samples, records process and hottest-thread
CPU/RSS/thread state, and observes resource use after dropping the session. Its JSON gates
duration, cadence, result-age drift, RSS/thread growth, active CPU, and stopped CPU. Fixed package
input remains smoke-only.

Run it from the repository root after building the example in release mode:

```text
cargo run --release -p nana-tracking-runtime-ort \
  --example benchmark-face-basic-stability -- \
  <libonnxruntime> <model-package> <output-json>
```

Optional trailing arguments override duration seconds, target FPS, and resource-sample seconds.
Resource sampling currently supports macOS and Linux; Windows validation remains a separate target.

The checked-in Apple M4 CPU smoke ran the release harness for 1,800.003 seconds and completed
107,997 frames at 59.998 FPS. Capture-to-result P50/P95/P99 was 0.533/0.862/0.976 ms; result-age
P50/P95/P99 was 0.539/0.870/0.987 ms and first-to-last P95 drift was 0.0023 ms. The process used
0.0338 CPU core equivalents, stayed at one thread, had no RSS growth, and measured zero CPU after
session drop. All eight gates passed. See
artifacts/benchmarks/issue11-rust-ort-face-basic-30m-macos-m4-smoke.json. This fixed-input result is
runtime smoke evidence, not camera, model-quality, GPU, cross-platform, or production acceptance.

The caller supplies capture and processing-start timestamps from one monotonic clock domain. The
backend includes pre-inference wait in result age while continuing to report preprocess, inference,
and readback separately. A Full extension reports the later of the Spatial result and its own
processing completion without counting the Spatial stages twice.

The application must initialize the process-wide ONNX Runtime library before constructing a
session. `initialize_from_dylib` supports application-packaged dynamic libraries without making a
Python installation part of the consumer contract.

Core ML is explicit opt-in through `load_core_ml`. Startup runs every fixed-vector output through
the requested provider, checks the configured per-output tolerance, ends an ORT profile, and rejects
the session unless at least one graph node actually ran on `CoreMLExecutionProvider`. Telemetry
distinguishes a Core ML graph with CPU fallback nodes. The temporary validation profile is removed
before the session is returned. ORT does not expose the internal CPU/GPU/ANE choice made by Core ML,
so the options make the requested compute policy explicit without claiming a specific compute unit.
The examples select Core ML when a writable profile directory is supplied as their final argument.
