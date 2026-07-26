# Data contract

- Each record must retain capture timestamp, sequence, camera metadata, teacher labels, per-label
  confidence, occlusion/lighting metadata, and identity/session/device grouping when available.
- Required manifest revisions: dataset schema, data revision and digest, NTP schema, Signal Registry.
- Split by identity first; sessions and devices may then be stratified inside an identity-safe split.
- MediaPipe may provide only pinned 16-point semantic 2D pseudo-labels. OpenCV may derive geometry
  only from synchronized, calibrated first-party views. ARKit/TrueDepth is evaluation-only and
  cannot provide training labels, calibration, thresholds, or recipe-selection evidence.
- Mark tongue, auricle, depth, occlusion, and out-of-frame truth unavailable when it cannot be
  observed reliably.
