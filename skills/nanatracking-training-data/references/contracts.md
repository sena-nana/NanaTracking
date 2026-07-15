# Repository contracts and source roles

## Authoritative paths

- License registry: `configs/data/license-registry.json`
- Dataset/capture contracts: `src/nana_tracking/data/manifest.py`, `schema.py`, and `capture.py`
- Local-first recorder/Studio contract: `src/nana_tracking/data/studio.py` and
  `docs/data/capture-archive-v1.md`
- Provenance/materialization: `src/nana_tracking/data/labeling.py`
- Frozen-F cache and split contracts: `src/nana_tracking/data/strategy.py`
- NTP label catalog: `configs/data/ntp-v1-label-catalog.json`
- ICT mapping gate: `configs/data/ict-facekit-light-to-ntp-v1.json`
- ARKit mapping gate: versioned files such as `configs/data/arkit-to-ntp-v1-smoke.json`; every
  mapping identifies its teacher license record and is digest-pinned by the frozen capture dataset
- G ablations: `configs/expression/ablation-v1.json`
- Strategy/specification: `docs/training/dataset-strategy.md` and
  `docs/training/synthetic-sequence-spec.md`
- Capture script/consent: `docs/training/collection-action-script.md` and
  `docs/training/capture-consent-template.md`
- F/G report templates: `examples/evaluation/f-direct-report-template-v1.json` and
  `expression-downstream-report-template-v1.json`

## Machine registry fields

Each record identifies kind, name/version/source, license and pinned license-text digest, review
state, commercial training/model distribution/raw redistribution/distillation/pseudo-label/
derivative-label permissions, attribution/share-alike duties, consent basis, allowed stages,
prohibited uses, evidence, and smoke-only status. Approved local license text must exist and match
its digest. A production run rejects a smoke-only record.

Stages are `base-model-training`, `expression-model-training`, `teacher-labeling`,
`synthetic-rendering`, `evaluation`, and `model-release`. Approval for one does not imply another.

## Commands

```bash
uv run --extra cpu nana-tracking data validate-licenses \
  configs/data/license-registry.json --stage <stage> --records <comma-separated-ids>
uv run --extra cpu nana-tracking data validate <manifest>
uv run --extra cpu nana-tracking data split-captures <records.jsonl> \
  --output <splits.json> --held-out-test-devices <ids>
uv run --extra cpu nana-tracking data split-actors <clip-index.json> \
  --output <splits.json> --validation-actors <n> --test-actors <n>
uv run --extra cpu nana-tracking data capture-freeze \
  --session-manifests <session.json,...> --capture-records <records.jsonl> \
  --arkit-mappings <mapping.json,...> --license-registry <registry.json> \
  --license-records <ids> --held-out-test-devices <ids> --data-revision <revision> \
  --output <frozen.json>
uv run --extra cpu nana-tracking data capture-build-training-manifest <frozen.json> \
  --label-catalog configs/data/ntp-v1-label-catalog.json --output <manifest.json>
uv run --extra cpu nana-tracking benchmark-expression-ablation \
  --config configs/expression/ablation-v1.json --output <report.json>
```

Prefix repository commands with `rtk` when available. Use CPython 3.14 and uv. A synthetic command
proves only schema/control flow.

Capture-based F configurations set `data.dataset=frozen_capture`, `data.manifest`, and
`data.frozen_capture`. Training re-verifies record, split, license, mapping, data revision, and NTP
revision equality before creating a loader. Other reviewed manifest sources keep their independent
immutable-manifest path and are not forced through the ARKit capture archive.

## Required source interpretation

ICT-derived truth can validate F parameters only after commercial source and render-asset approval.
First-party truth requires per-participant commercial training/model-distribution consent, retention
and withdrawal mapping, reviewed SDK rights, and synchronized provenance. CREMA-D can validate G
expression behavior only; its ODbL/DbCL and person/content obligations require an approved registry
decision before use.

## Report constraints

F reports MAE/RMSE/CCC, geometry/head pose, asymmetry, event F1, neutral jitter, dynamic delay/peak,
occlusion recovery, identity/session/device generalization, and confidence calibration. F reports
must state `crema_d_used=false`.

G reports actor-held-out macro-F1, balanced accuracy, intensity, probability calibration, temporal
and parameter-group ablations, and the RGB upper-bound gap. It must state the frozen F digest and
that results do not measure BasicSet numeric accuracy.

Never combine F and G into one “face tracking accuracy” score.
