# `NanaTracking` ONNX Runtime backend

This crate is the real CPU baseline implementation of `TrackingModelSession` for verified
`FaceBasic` model packages. It keeps ORT tensors and session values private, reuses one NCHW input
buffer, writes only framework-neutral values into caller-owned output storage, and reports the
actual provider and stage timings.

The application must initialize the process-wide ONNX Runtime library before constructing a
session. `initialize_from_dylib` supports application-packaged dynamic libraries without making a
Python installation part of the consumer contract.
