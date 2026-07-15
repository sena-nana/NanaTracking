# NTP training dataset contract v1

## Revisions and authoritative files

- Dataset schema: `ntp-dataset/2.0.0`
- Capture schema: `ntp-capture/1.0.0`
- Label catalog: `ntp-label-catalog/1.0.0`
- NTP schema: `ntp/1.0`
- Signal Registry: `ntp-signals/1.0.0`
- Normalization: `ntp-normalization/1.0.0`
- Calibration: `ntp-calibration/1.0.0`
- Feature revision: `ntp-features/1.0.0`

The Pydantic models in `nana_tracking.data.manifest`, `nana_tracking.data.schema`, and
`nana_tracking.data.labeling` are the machine-readable schemas. The checked-in label catalog binds
all 76 Core signals and 12 optional signals to a reviewed source strategy. Any semantic change
requires a new schema or catalog revision; an existing revision is immutable.

The manifest pins SHA-256 digests for the label catalog and each JSONL record file. Its dataset
digest is the SHA-256 of the canonical, sorted complete manifest excluding only the digest field.
Validation fails if a file, revision, license decision, split, synchronization policy, count, or
digest drifts.

## Capture record

Every record contains the authoritative RGB capture timestamp and sequence, RGB URI, width,
height, exposure duration, ISO, frame duration, camera intrinsics and distortion, and
identity/session/device groups. It also records lighting, occlusion conditions, timestamped teacher
frames, per-label confidence and state, and optional timestamped depth observations.

Raw RGB, depth, private metadata, and user recordings remain outside Git. A URI may resolve only in
the access-controlled data environment. Reviewed manifests, schemas, digests, and tiny explicitly
synthetic fixtures are the only dataset artifacts admitted to this repository.

## Deterministic label materialization

For every stable Signal ID, the materializer does the following in stable ID order:

1. Select candidates whose source type, evidence type, method revision, and state match the pinned
   catalog strategy.
2. Reject candidates outside the RGB/teacher synchronization window, outside the Signal Registry
   range, or carrying a non-finite value.
3. Reject predicted candidates as training truth. Strategies for tongue and auricle additionally
   accept only directly observed candidates.
4. If approved teachers differ beyond the scalar-type threshold, emit `teacher_disagreement` and
   make the label unavailable instead of selecting a convenient teacher.
5. Otherwise combine candidates in source-ID order using confidence weights and reduce confidence
   according to the observed spread.
6. If no candidate survives, emit a label with no value, confidence zero, and a deterministic
   unavailable reason. Missing truth is never replaced with a neutral zero.

Metric depth is available only from a synchronized, directly observed TrueDepth or multiview
source. Fused, predicted, stale, monocular-relative, and missing depth remain explicitly
unavailable. Relative monocular Z may still be used by a separately versioned normalized geometry
method, but it must never be described as metric depth.

## Core source coverage

| Stable IDs | Source contract |
| --- | --- |
| 1-36 face | reviewed TrueDepth/RGB observations through `ntp-face-orthogonal/1.0.0` |
| 37-40 gaze | Head-local eye-ray geometry through `head-local-eye-ray/1.0.0` |
| 41 tongue extension | directly visible reviewed tongue label |
| 42-47 torso | multiview camera-to-torso geometry |
| 48-53 head relative pose | synchronized torso/head geometry |
| 54-56 tongue detail | directly visible reviewed tongue geometry |
| 57-62 auricle | directly observed reviewed auricle geometry |
| 63-66 shoulder girdle | torso/shoulder geometry |
| 67-76 arms | shoulder/elbow/wrist geometry |

The catalog also covers optional IDs 77-88. A new source method must publish its coordinate,
normalization, calibration, observability, and license decision before it can replace or extend a
strategy.

The manifest-backed model loaders apply profile-specific admission gates. `FaceBasicDataset`
loads Basic or Spatial truth and masked face geometry. `FullSetDataset` loads the Full-only 42-76
block plus synchronized torso-local shoulder/elbow/wrist geometry, limb directions and twists, and
normalized bone lengths. `require_complete_full` rejects incomplete Full scalar truth; auxiliary
geometry remains confidence-masked rather than filled with training targets.

## TrueDepth and RGB synchronization

- The RGB capture clock is authoritative. Teacher timestamps use the same monotonic clock domain
  or carry a reviewed clock conversion before admission.
- The default teacher/RGB skew limit is 5 ms. Metric depth uses a 2 ms limit. Production manifests
  may tighten these limits but cannot silently relax the catalog revision's published limit.
- Sequence must strictly increase inside a session. Capture timestamps must strictly increase;
  duplicate or backward frames fail validation.
- Nearest-neighbour relabeling across the skew boundary and timestamp rewriting are forbidden.
  Interpolation is a separately identified derived source, never an observed label.
- Frame duration, exposure interval, dropped-frame gaps, and source clock drift are retained so
  temporal evaluation can use capture time rather than arrival order.

## Quality gates and identity-safe splitting

Structural failures stop validation: unknown sources, duplicate record IDs, undeclared groups,
identity overlap, invalid license permission, count/digest drift, non-monotonic sequence/time, and
invalid values. Recoverable observations are retained as unavailable labels and quality issues:
teacher skew, teacher disagreement, occlusion, out of frame, prediction-only evidence, and
unreliable depth.

Identity is assigned to exactly one of train, validation, or test before sessions and devices are
stratified. All sessions and devices for that identity remain in the same split. Failure-sample
replay retains the original identity split; moving a hard sample must move the whole identity and
publish a new data revision and digest.

Run:

```bash
uv run --extra cpu nana-tracking data validate examples/manifests/synthetic-v1.json
uv run --extra cpu nana-tracking data materialize-labels \
  examples/manifests/synthetic-v1.json --output artifacts/data/synthetic-labels.jsonl
```

The example is synthetic smoke-only evidence. It proves schema and control-flow behavior, not
FaceBasic quality or production data readiness.
