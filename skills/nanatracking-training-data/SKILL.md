---
name: nanatracking-training-data
description: Enforce NanaTracking first-party-only commercial data admission, MediaPipe/OpenCV teacher provenance, ARKit evaluation isolation, identity-safe splitting, labeling, checkpoints, and release evidence. Use for adding or evaluating data; changing capture schemas, teachers, pseudo-labels, mappings, loaders, splits, training, reports, or derived artifacts.
---

# NanaTracking training data

Apply fail-closed admission before collection, labeling, training, or evaluation. Read
[`references/contracts.md`](references/contracts.md) and
[`references/checklists.md`](references/checklists.md) for the matching workflow.

## Fixed policy

- Use only explicitly consented, project-owned first-party recordings for real model development.
  Do not use existing face datasets, their labels, caches, weights, statistics, or thresholds.
- MediaPipe Face Landmarker is a pinned pseudo-label teacher for the reviewed 16 semantic 2D
  anchors only. Its blendshapes, pose, metric depth, identity, and confidence are not NTP truth.
- OpenCV is a deterministic geometry teacher only when synchronized views and reviewed camera
  calibration support triangulation, pose, and reprojection checks.
- ARKit/TrueDepth outputs are an isolated evaluation reference. They never enter training,
  pseudo-labeling, calibration, threshold/recipe selection, checkpoint initialization, or release
  lineage.
- Repository synthetic fixtures are smoke-only and never prove tracking quality.
- Do not commit raw recordings, biometric metadata, consent records, teacher caches, checkpoints,
  or model packages.

## Admit captures and teachers

1. Record authoritative code/model licenses, exact versions, model digests, attribution, allowed
   outputs, and prohibited roles.
2. Obtain per-participant consent for commercial training, derived labels, weight distribution,
   review, retention, and withdrawal.
3. Add or update the machine registry before collection or labeling. Pending means denied.
4. Run `data validate-licenses` for the exact stage and commercial tier.
5. Verify the MediaPipe teacher descriptor and bundle digest before inference.

## Build Stage A labels

1. Split identities before sampling. Keep every session for an identity in one split and reserve
   test devices/cameras.
2. Preserve synchronized front/left/right RGB, calibration, timestamps, and continuous 15-30 FPS
   validation/test clips; only training may sample to approximately 5 FPS.
3. Store MediaPipe 16-point outputs as `pseudo_label` with source/model/mapping version and
   confidence.
4. Use OpenCV to undistort, triangulate, solve pose, and calculate reprojection residuals. Reject
   missing/unsynchronized/invalid groups rather than filling values with zero.
5. Human-review sampled overlays and freeze the calibration, mapping, and quality threshold before
   training.
6. Exclude rig/confidence heads from the Stage A optimizer and prove they remain bitwise unchanged.

## Train, evaluate, and release

- Keep PyTorch authoritative and NTP free of framework tensors or teacher-native types.
- Pin manifest, teacher, mapping, calibration, recipe, seed, Git, lock, NTP, Signal Registry,
  metrics, and checkpoint metadata.
- Compare identical-schedule scratch, synthetic-only, and A-to-B-to-C FaceBasic runs.
- Report identity-level bootstrap confidence intervals and continuous jitter, delay, peak retention,
  and recovery.
- Report ARKit comparison separately and ensure its outputs cannot affect model selection.
- Release only from a `commercial-reproduced` recipe that passes the locked first-party test set
  and all license/consent gates.

Keep this skill synchronized with Issues
[#7](https://github.com/sena-nana/NanaTracking/issues/7) and
[#12](https://github.com/sena-nana/NanaTracking/issues/12), the registry, schemas, tests, and report
templates whenever this boundary changes.
