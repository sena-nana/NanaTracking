# Data contract

- Each record must retain capture timestamp, sequence, camera metadata, teacher labels, per-label
  confidence, occlusion/lighting metadata, and identity/session/device grouping when available.
- Required manifest revisions: dataset schema, data revision and digest, NTP schema, Signal Registry.
- Split by identity first; sessions and devices may then be stratified inside an identity-safe split.
- MediaPipe may provide only pinned 16-point semantic 2D pseudo-labels. OpenCV may derive geometry
  only from synchronized, calibrated first-party views. ARKit/TrueDepth is evaluation-only and
  cannot provide training labels, calibration, thresholds, or recipe-selection evidence.
- First-party Stage A manifests pin the MediaPipe descriptor/bundle, ordered mapping, reviewed
  calibration and quality profile, approved aggregate overlay review, independent per-record human
  review, recipe, split plan, and materialized shard. Identity, session, device, and camera IDs are
  split-owned. Split counts are recipe policy, not schema invariants.
- Missing views, skew above 5 ms, invalid depth/angle/reprojection/PnP, and unreviewed or tampered
  inputs fail closed. Geometry and pose then have null values, zero weight, and a stable reason.
- Mark tongue, auricle, depth, occlusion, and out-of-frame truth unavailable when it cannot be
  observed reliably.
