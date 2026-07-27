# Repository contracts

## Authoritative paths

- License registry: `configs/data/license-registry.json`
- MediaPipe teacher descriptor: `configs/data/mediapipe-face-landmarker-v1.json`
- Legacy pilot recipe: `configs/training/nana-training-recipe-1.0.0.json`
- Capture and manifest contracts: `src/nana_tracking/data/capture.py`, `manifest.py`, `schema.py`
- Teacher contract: `src/nana_tracking/data/teachers.py`
- Label materialization: `src/nana_tracking/data/labeling.py`
- Strategy: `docs/training/dataset-strategy.md`
- Capture action/consent: `docs/training/collection-action-script.md`,
  `docs/training/capture-consent-template.md`

## Source roles

- First-party consented RGB is the only real training/evaluation source.
- `mediapipe-face-landmarker-v1` may produce `semantic-landmarks-2d` pseudo-labels.
- `opencv-calibrated-geometry-v1` may derive calibrated multiview geometry and pose.
- `apple-arkit-truedepth-teacher` may be admitted only at `evaluation`.
- Existing dataset records remain rejected as machine-readable negative policy evidence.

## Commands

```bash
uv run --extra cpu nana-tracking data validate-licenses \
  configs/data/license-registry.json --stage teacher-labeling \
  --records mediapipe-face-landmarker-v1,opencv-calibrated-geometry-v1 \
  --usage-tier commercial
uv run --extra cpu nana-tracking data validate-teacher-model \
  configs/data/mediapipe-face-landmarker-v1.json \
  --model-asset <face_landmarker.task>
uv run --extra cpu nana-tracking data split-captures <records.jsonl> \
  --output <splits.json> --held-out-test-devices <ids>
uv run --extra cpu nana-tracking data validate <manifest.json>
```

Use CPython 3.14 and uv. Synthetic commands prove control flow only.

## Artifact rule

Every pseudo-label stores source ID, teacher/model version, mapping revision, timestamp,
confidence, and evidence=`mediapipe_pseudo_label`. A genuinely reviewed record may use
`human_corrected_pseudo_label`; aggregate approval alone cannot change this value. Every
OpenCV-derived label uses evidence=`deterministic_geometry` and stores calibration and derivation
revisions plus residual-based quality evidence. ARKit comparison data is stored in an
evaluation-only manifest and cannot be loaded by a training configuration.
