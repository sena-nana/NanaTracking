# Commercial dataset and two-stage expression strategy

## Hard boundary

Model F maps monocular RGB or a short causal RGB history to NTP BasicSet signals, head pose,
visibility, auxiliary geometry, and per-signal confidence. Only reviewed ICT-derived synthetic
renders and explicitly consented first-party captures may supervise F. CREMA-D emotion labels may
never be interpreted as NTP numeric truth, geometry truth, or confidence truth.

Model G consumes an offline cache produced by a frozen, digest-pinned F. Its inputs are BasicSet
history, velocity, acceleration, confidence, visibility, duration, frame quality, and optionally
head pose. It outputs an emotion probability distribution, intensity, neutral probability, and
confidence. The initial G workflow has no handle to F parameters or optimizer state, so an emotion
loss cannot change NTP semantics. Any future joint optimization must retain F's physical losses and
pass an independent F regression suite before it can be considered.

CREMA-D results only measure whether the parameter representation retains downstream expression
information across held-out actors. They do not validate the numerical accuracy of any BasicSet
signal. The RGB expression model is an upper-bound reference, not a teacher for F.

## Source roles

| Source | Permitted role after license approval | Never permitted |
| --- | --- | --- |
| ICT Face Model Light derived renders | F parameter/geometry truth, controlled coverage | Sole evidence for real-camera quality |
| Consented RGB plus TrueDepth/ARKit | Real-domain F adaptation, temporal/device/identity evaluation | Unreviewed SDK output as absolute physical truth |
| CREMA-D | Frozen-F cache to G, actor-held-out expression evaluation | Direct F supervision or BasicSet accuracy claims |

The machine registry is `configs/data/license-registry.json`. Missing, pending, rejected,
stage-incompatible, smoke-only-for-production, or commercially incomplete records fail closed.
The current ICT, CREMA-D, ARKit/TrueDepth, and first-party collection entries remain pending; no
production training is authorized by this repository snapshot.

## Reproducible pipeline

1. Validate the global license registry for the exact stage and source record IDs.
2. Validate the dataset manifest, digests, revisions, teacher permissions, synchronization, and
   identity/session split. A production capture additionally requires environment, action-script,
   consent-record, and approved human-review fields.
3. Build F splits with `data split-captures`, assigning every identity and all its sessions to one
   split and reserving one or more devices exclusively for test.
4. Train/evaluate F without CREMA-D. Pin the manifest digest in the checkpoint and report.
5. Run the released F offline over CREMA-D only after its license record is approved. Emit the
   `nana-expression-cache/1.1.0` format; the contract requires a frozen F digest, BasicSet IDs 1..36
   in order, and an exact source dataset name/revision bound to its admitted dataset-license record.
6. Build actor-level G splits with `data split-actors`; clips from one actor cannot cross splits.
7. Train G and the required ablations. Store F and G reports separately.

The checked-in G report is synthetic smoke-only. It verifies the frozen boundary, actor split,
feature paths, validation/test metrics, confidence-head error, and complete ablation execution. The
report pins its resolved configuration, generated data digest, Git state, and dependency lock, but
does not claim CREMA-D performance.

## Required evaluation

F reports parameter MAE/RMSE/CCC, landmark NME, head-pose error, asymmetric-action preservation,
event F1, neutral jitter, dynamic delay/peak retention, cross-identity/session/device behavior, and
confidence calibration on ICT-derived and first-party held-out truth only.

G reports actor-held-out macro-F1, balanced accuracy, intensity error or rank correlation,
calibration, and the RGB upper-bound gap. The mandatory suite is all parameters, single frame,
parameters plus velocity, parameters plus velocity/acceleration, mouth/jaw only, mouth/viseme
removed, head pose only, head pose removed, shuffled time, and RGB upper-bound reference.

## Release stop rules

- No source or render asset enters a run without an approved machine record and pinned license text.
- No identity or session crosses F splits; held-out test devices cannot occur in development.
- No actor crosses G splits.
- A G cache whose F is not marked frozen, whose digest/revisions drift, or whose BasicSet is
  incomplete is rejected.
- F and G reports stay separate. Emotion accuracy cannot waive F regression failures.
- Raw recordings, CREMA-D media, caches, model weights, consent documents, or biometric identifiers
  are never committed.
