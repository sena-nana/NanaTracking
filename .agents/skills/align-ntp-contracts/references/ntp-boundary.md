# NTP boundary

- PyTorch defines training behavior; ONNX is a deployable graph, not a protocol schema.
- NTP, Signal Registry, semantic derivation, network, and FFI contracts must not depend on ML
  framework types.
- Artifacts bind explicitly to NTP schema, Signal Registry, normalization, calibration, adapter,
  and feature revisions.
- Optimized backends must preserve the same per-output meaning and quality state as the baseline.
- Unknown future signals remain ignorable; published IDs and meanings are never silently reused.
- The ordered 16 MediaPipe anchors, HeadRelative canonical geometry, OpenCV pose, quality
  residuals, and overlay-review state are internal training contracts. They never allocate or
  reuse an NTP Signal ID, and unavailable values remain separate from neutral numeric zero.
