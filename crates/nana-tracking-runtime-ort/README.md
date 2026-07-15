# `NanaTracking` ONNX Runtime backend

This crate is the real CPU baseline implementation for verified `FaceBasic`, `FaceSpatial`, and
Full-only model packages. Face packages implement `TrackingModelSession`; the Full-only package is
fused into a same-capture Spatial output because its head-relative and tongue fields are not
semantically complete in isolation. The crate keeps ORT tensors and session values private, reuses
one NCHW input buffer per session, writes only framework-neutral values into caller-owned output
storage, and reports the actual provider and stage timings.

The caller supplies capture and processing-start timestamps from one monotonic clock domain. The
backend includes pre-inference wait in result age while continuing to report preprocess, inference,
and readback separately. A Full extension reports the later of the Spatial result and its own
processing completion without counting the Spatial stages twice.

The application must initialize the process-wide ONNX Runtime library before constructing a
session. `initialize_from_dylib` supports application-packaged dynamic libraries without making a
Python installation part of the consumer contract.
