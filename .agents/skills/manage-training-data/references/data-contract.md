# Data contract

- Each record must retain capture timestamp, sequence, camera metadata, teacher labels, per-label
  confidence, occlusion/lighting metadata, and identity/session/device grouping when available.
- Required manifest revisions: dataset schema, data revision and digest, NTP schema, Signal Registry.
- Split by identity first; sessions and devices may then be stratified inside an identity-safe split.
- TrueDepth/ARKit may provide face, pose, gaze, and geometry teachers. Other teachers need explicit
  provenance and license review.
- Mark tongue, auricle, depth, occlusion, and out-of-frame truth unavailable when it cannot be
  observed reliably.
