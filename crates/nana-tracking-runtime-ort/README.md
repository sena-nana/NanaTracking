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
