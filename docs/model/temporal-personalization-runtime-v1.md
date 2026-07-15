# Causal temporal, confidence, personalization, and runtime v1

Status: executable local/synthetic baseline for Issue #10. It proves capture-time causal behavior,
bounded prediction, monotonic confidence calibration, resettable Levels A/B/C, latest-frame
scheduling, stage telemetry, package compatibility, and ONNX adapter parity. It does not prove
real-sequence tracking-quality gains or untested deployment backends.

## Causal temporal contract

`CausalTemporalRefiner` accepts plain scalar values, confidence, NTP state, source capture time,
session/generation, camera identity, and calibration revision. It keeps a fixed 4-8-frame
compressed history and uses no future frame. Any session, generation, camera, or calibration change
clears history. Backward or duplicate capture timestamps fail rather than silently changing `dt`.

Each signal family has a separate capture-time time constant: fast eyelid/tongue, expression,
pose, body, and auricle. Refinement is a bounded residual on the current absolute value. Large
blink/mouth/tongue/elbow changes bypass smoothing so peaks are not weakened. Occluded or
out-of-frame values may use bounded constant-velocity prediction for at most 120 ms; prediction
horizon is explicit, confidence decays with horizon, and expiry becomes `TrackingLost`. No
unbounded integrator or arrival-time filter exists. Velocity is estimated only from the latest two
real `Observed`/`Fused` samples: predicted frames never become synthetic observations. Repeated
low-cadence `Fused` frames with the same source timestamp are deduplicated before velocity
estimation. Repeated occluded frames therefore advance from one stable capture-time origin, and the
first recovered observation bypasses prediction smoothing instead of inheriting recovery lag.

`FaceBasicProducer` optionally applies this refiner after Level A normalization and before NTP
serialization. Predicted values carry the original sample timestamp and nonzero prediction
horizon; conformance is tested on an observed-to-occluded sequence.

## Confidence calibration

Held-out identity-safe evidence supplies per-signal raw confidence and binary correctness.
`fit_confidence_calibration` bins each signal deterministically and applies pool-adjacent-violators
isotonic fitting. The versioned artifact binds model family/version and Signal Registry revision.
Runtime lookup is monotonic; a signal without a curve retains its original confidence. Training
evidence and artifact compatibility are validated, not inferred from synthetic model accuracy.

## Personalization levels

- Level A is numerical only. A profile may cover any ordered stable Signal IDs and includes robust
  neutral, asymmetric observed range, deadzone, optional shoulder width, and torso neutral. It is
  model/revision/user bound and resettable.
- Level B trains only a bounded per-signal affine residual on offline user captures. The base
  encoder is never passed to the optimizer. The ONNX adapter emits residuals, carries a source-data
  digest and base-model digest, and refuses a different user slot, model version, digest, or feature
  revision. Fixed vectors verify PyTorch/ONNX Runtime parity.
- Level C updates only after explicit stable neutral or range evidence with high confidence. Every
  step and total drift are bounded; snapshots support rollback and reset. Merely holding an
  expression never supplies the required explicit neutral evidence.

## Runtime and packages

`LatestFrameRuntime` retains one pending frame, sleeps on a condition variable, and replaces stale
pending work. It exposes Performance/Quality mode plus bounded-window P50/P95/P99/mean telemetry
for mailbox wait, preprocessing, inference, small-output readback/mapping, total producer work, and
result age. Closing joins the worker; no busy polling or unbounded telemetry queue exists.

Model metadata now declares guaranteed profile, signals, structures, features, temporal
compatibility, allowed backend family, runtime mode scheduling, precision, and adapter schema.
Current portable artifacts are verified for ONNX Runtime FP32. TensorRT FP16, DirectML, Core ML,
Metal, and INT8 remain target-specific acceptance work; absent execution evidence is not advertised
as validated support.

## Local smoke evidence

The checked-in irregular-`dt` synthetic benchmark used 5,000 frames on Apple M4 macOS. Static
mouth-signal standard deviation fell 33.28%, the injected fast eyelid peak retained 100%, and
refiner overhead was 0.00313 ms p50 / 0.00354 ms p95 / 0.00442 ms p99. Two consecutive occluded
frames advanced from 10 ms to 20 ms prediction horizon, confidence decreased from 0.7053 to 0.6218,
and the first recovered observation had zero synthetic recovery error. The report now records the
resolved policy, fixed seed, generated-input digest, NTP revisions, Git state, lock digest, and
runtime versions. This benchmark proves algorithm behavior and overhead on fixed numerical inputs
only; it is not real-camera quality or cross-platform acceptance.

- `artifacts/benchmarks/issue10-temporal-macos-m4-smoke.json`
