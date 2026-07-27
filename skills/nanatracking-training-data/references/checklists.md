# Execution checklists

## Capture admission

- [ ] First-party source only; no existing dataset media, labels, weights, caches, or statistics
- [ ] Commercial training, derived-label, model-distribution, review, retention, and withdrawal
      consent complete
- [ ] Identity/session/device IDs are de-identified and identity-safe splits are frozen
- [ ] Raw recordings and consent records remain access-controlled outside Git

## Teacher labeling

- [ ] MediaPipe 0.10.35 bundle, model cards, license, SHA-256, and 16-point mapping are pinned
- [ ] Only semantic 2D landmarks are marked `pseudo_label`
- [ ] OpenCV version, camera calibration, synchronization, triangulation/PnP method, and
      reprojection thresholds are pinned
- [ ] Missing views, failed detections, invalid depth, and excessive residuals fail closed
- [ ] Overlay samples and calibration are human-reviewed before aggregate approval
- [ ] Every admitted record has independent human evidence; corrected records are re-materialized
- [ ] ARKit labels are absent from every training manifest and checkpoint lineage

## Stage A training

- [ ] At least 8 identities, default 5/1/2 identity split
- [ ] Training may use approximately 5 FPS; validation/test retain continuous 15-30 FPS
- [ ] MediaPipe/OpenCV/first-party records pass commercial admission
- [ ] Rig/confidence heads are excluded from the optimizer and bitwise unchanged
- [ ] Single-view and multiview runs use identical seed, schedule, and eligible data
- [ ] Checkpoint metadata pins all source, teacher, mapping, calibration, recipe, and code digests

## Evaluation and release

- [ ] Scratch, synthetic-only, and A-to-B-to-C internal baselines are compared
- [ ] Identity-paired bootstrap uses 10,000 samples and reports 95% confidence intervals
- [ ] Geometry, pose, jitter, delay, peak retention, recovery, confidence, and runtime are reported
- [ ] ARKit is a separate comparison only and cannot select parameters, thresholds, or recipes
- [ ] Locked first-party test data is opened only after the commercial recipe is frozen
- [ ] Synthetic smoke and pending/rejected sources cannot support release claims
