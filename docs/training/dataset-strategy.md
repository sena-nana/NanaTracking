# First-party commercial training strategy

## Fixed data boundary

NanaTracking production models use only explicitly consented, project-owned first-party captures.
Existing face datasets, their annotations, derived caches, checkpoints, statistics, and
test-selected thresholds are excluded from training, evaluation, recipe selection, and release
evidence. Repository-generated arrays remain synthetic smoke fixtures only.

The approved training teachers are:

| Source | Training role | Prohibited role |
| --- | --- | --- |
| MediaPipe Face Landmarker 0.10.35, pinned model bundle | 16 semantic 2D landmark pseudo-labels with teacher confidence and provenance | NTP Basic numeric truth, metric 3D/pose truth, identity labels, confidence truth |
| OpenCV contrib-python 5.0.0.93, sole `cv2` provider | Calibrated undistortion, multiview triangulation, solvePnP, and reprojection residuals | Creating truth from an uncalibrated single view or converting residuals directly to model confidence |
| Apple ARKit/TrueDepth | Isolated first-party validation/test comparison only | Training, pseudo-labeling, calibration, threshold selection, recipe selection, or checkpoint lineage |

MediaPipe and OpenCV licenses do not grant rights to participant recordings. Every real capture
batch still requires reviewed consent for commercial training, derived labels, model-weight
distribution, retention, and withdrawal.

## Stage A teacher flow

1. Collect synchronized first-party RGB from front and approximately left/right 30-degree cameras.
   Preserve intrinsics, extrinsics, timestamps, identity/session/device IDs, and continuous clips.
2. Run the digest-pinned MediaPipe bundle independently on each view. Retain only the reviewed
   16-point semantic mapping and mark every output `pseudo_label`.
3. Use OpenCV with the recorded calibration to undistort points, triangulate anchors, transform them
   to a head-relative frame, solve pose, and calculate per-view reprojection residuals.
4. Reject unsynchronized groups, missing views, failed detections, invalid depth, and groups above
   the frozen reprojection threshold. Human review samples overlays before a capture revision is
   approved.
5. Train encoder, 2D anchors, canonical geometry, pose, visibility, and identity adversary.
   Rig/confidence heads remain outside the Stage A optimizer.

Training may sample approximately 5 FPS. Validation and test preserve continuous 15-30 FPS clips.
Identity splits are fixed first; devices and sessions cannot leak across the split boundary.
Camera IDs are also split-owned. The production skew ceiling is 5 ms. No production calibration or
quality threshold exists until a named reviewer approves its digest; repository defaults never
silently supply these values.

`data materialize-stage-a-labels` emits candidate JSONL, a quality summary, deterministic PNG
overlays, and an overlay index. Red points are MediaPipe observations, green points are OpenCV
reprojections, and yellow segments are residuals. `data build-stage-a-manifest` accepts only an
approved review whose materialization and aggregate overlay digests match, then freezes the
commercial v3 shard. Rejected geometry and pose are always `null` with zero weight and a stable
failure code; residuals never become confidence.

## A/B/C/D commercial route

- **A-final:** train real geometry, pose, landmarks, and visibility from first-party captures and
  the admitted MediaPipe/OpenCV teacher flow.
- **B-final:** train NTP parameter/confidence heads from project-owned synthetic or directly
  controlled first-party labels. The first 20% freezes encoder/pose/geometry; the remainder
  unfreezes only the final encoder stage at 0.1 times the head learning rate.
- **C-final:** alternate first-party real batches and project-owned synthetic batches. Real batches
  compute geometry/pose/visibility/multiview losses; synthetic batches compute complete NTP losses.
- **D-final:** evaluate on locked first-party identities, devices, and continuous clips. ARKit may
  appear only in a separately reported comparison column.

No current real-data checkpoint exists. The checked-in recipe is a `commercial-candidate`, starts
without a parent checkpoint, and cannot become `commercial-reproduced` until the consented
first-party development set reproduces the expected behavior. Only then may a locked first-party
test set be opened for `release-eligible` review.

## Required gates

- Admit the first-party capture program and both teachers for the exact stage before labeling.
- Verify the MediaPipe bundle SHA-256 and the ordered 16-point mapping.
- Pin calibration, teacher/model, mapping, data, config, lock, Git, NTP, and Signal Registry
  revisions in every run.
- Compare the staged model with identical-schedule scratch and synthetic-only FaceBasic baselines.
- Use identity-level paired bootstrap with 10,000 samples and report 95% confidence intervals.
- Require improvement over both internal baselines in geometry, pose, and continuous behavior,
  without stable rig, confidence, latency, or recovery regression.
- Keep raw recordings, consent records, teacher caches, checkpoints, and full reports outside Git.

The machine policy is `configs/data/license-registry.json`; the pinned MediaPipe teacher contract
is `configs/data/mediapipe-face-landmarker-v1.json`; and the candidate schedule is
`configs/training/nana-training-recipe-1.0.0.json`. Missing or pending records fail closed.
