---
name: nanatracking-training-data
description: Enforce NanaTracking commercial data admission, provenance, identity-safe splitting, synthetic generation, ICT-to-NTP mapping, frozen-F expression caches, F/G training boundaries, and separate evaluation/release evidence. Use for adding, downloading, purchasing, or evaluating datasets or render assets; changing capture schemas, action scripts, loaders, labels, teachers, pseudo-labels, mappings, caches, splits, training, model export metadata, or F/G reports; and any FaceBasic data/loss/evaluation change.
---

# NanaTracking training data

Apply fail-closed commercial admission before downloading, rendering, labeling, or training. Read
[`references/contracts.md`](references/contracts.md) for repository schemas, commands, and source
roles. Read [`references/checklists.md`](references/checklists.md) for the workflow being executed.

## Stop rules

- Reject a source until its dataset license, content rights, biometric/likeness consent, and any
  SDK/teacher-output terms explicitly allow the requested stage. Downloadability, open-source code,
  or not redistributing raw data are insufficient.
- Add every dataset, capture program, render asset, and teacher SDK to the machine registry before a
  manifest or training configuration references it. Treat missing or pending permission as denial.
- Never call teacher predictions ground truth. Record source/model/version, synchronization,
  confidence, NTP mapping revision, and allowed supervision roles.
- Split F by identity, keep all sessions with that identity, and reserve explicit devices for test.
  Split CREMA-D/G by actor. Never split video by frame. Preserve split IDs through augmentation and
  caches.
- Mark every synthetic artifact and report smoke-only. Never use it as production quality evidence.
- Do not commit raw recordings, third-party media, biometric metadata, caches, checkpoints, or model
  packages.

## Preserve source roles

- Use licensed ICT Face Model Light-derived synthetic data only for controlled parameter, geometry,
  pose, occlusion, and visibility truth for model F. Do not treat it as the real camera domain.
- Use explicitly consented and licensed RGB plus TrueDepth/ARKit captures for real-domain,
  cross-identity/device/session, temporal, and final F calibration evidence. Review teacher outputs;
  do not assume they are absolute physical truth.
- Use CREMA-D only after approval to evaluate expression information and train model G from a frozen
  F prediction cache. Never convert emotion classes into NTP/Blendshape truth or cite CREMA-D
  classification as BasicSet numeric accuracy.

## Preserve the two-stage boundary

Treat PyTorch model F as `RGB/causal RGB -> BasicSet + pose/geometry/state/confidence`. Train F only
from admitted synthetic/first-party parameter or geometry supervision.

Treat PyTorch model G as `frozen F parameter history -> expression distribution/intensity/neutral/
confidence`. Pin F digest and revisions, generate the cache offline, and expose no F parameters or
optimizer to initial G training. A later joint optimization requires explicit approval, continued F
physical supervision, and an independent F regression gate; emotion gains never waive parameter
error, jitter, latency, peak, or semantic regressions.

## Execute the matching workflow

### Admit data, assets, or teachers

1. Locate authoritative license and content terms plus participant/biometric consent basis.
2. Record commercial training, weight distribution, raw redistribution, attribution/share-alike,
   distillation, pseudo-labeling, derivative-label, allowed-stage, and prohibited-use decisions.
3. Pin the license text digest and add the record to `configs/data/license-registry.json`.
4. Run stage-specific `data validate-licenses`. Stop if it rejects the source.
5. Only then download or reference the source from a versioned manifest.

### Change schema, labels, mapping, or synthetic generation

1. Version the schema/mapping; never mutate published semantics in place.
2. Describe migration and cache/checkpoint invalidation.
3. Preserve old fixed vectors and add functional validation for the new behavior.
4. For capture mappings, pin the mapping file and teacher license in the frozen dataset, regenerate
   derived records without mutating raw chunks, and build the training manifest only from the
   verified frozen revision.
5. For synthetic sampling, cover isolated actions, reviewed combinations, asymmetry,
   neutral-onset-peak-recovery, viseme coexistence, camera/light/occlusion/imaging variation, and
   bounded edge cases. Never sample all parameters independently and uniformly.
6. Re-run license, coverage, provenance, and leakage gates.

### Train or evaluate F

1. Complete the F checklist and pin manifest/frozen-capture digests, seed, revisions, Git/lock
   state, and config.
2. Verify complete label provenance/confidence and licensed renderer/asset/teacher records.
3. Train without CREMA-D.
4. Report direct metrics only on parameter/geometry-labeled ICT-derived or first-party holdouts.

### Build a G cache, train G, or evaluate expression

1. Complete the G checklist and verify CREMA-D admission before accessing it.
2. Pin the released F digest and NTP/Signal/feature revisions; require `frozen=true`.
3. Cache BasicSet 1..36, confidence, visibility, head pose, frame quality, timestamps, label
   distribution/source, intensity, shard digests, and actor split.
4. Run every required temporal/parameter/head-pose/mouth/RGB ablation.
5. Interpret results only as actor-held-out expression separability and temporal information.

### Release a model or report

1. Complete the release checklist and re-run license admission for `model-release`.
2. Attach the source/license manifest and separate F direct and G downstream reports.
3. Pin NTP schema, Signal Registry, mapping, feature, data/cache, model, and config revisions/digests.
4. Block release on any non-commercial, unknown, withdrawn, expired, unapproved-teacher, split-leak,
   or semantic-regression input.

## Keep governance synchronized

Keep this skill consistent with GitHub issues
[#2](https://github.com/sena-nana/NanaTracking/issues/2),
[#6](https://github.com/sena-nana/NanaTracking/issues/6),
[#7](https://github.com/sena-nana/NanaTracking/issues/7), and
[#12](https://github.com/sena-nana/NanaTracking/issues/12), plus the repository contracts listed in
the references. When any protocol or data boundary changes, update this skill, machine
registry/schema, tests, templates, and migration notes in the same change set.
