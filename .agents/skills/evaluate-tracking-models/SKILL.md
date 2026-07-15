---
name: evaluate-tracking-models
description: Evaluate NanaTracking models with per-head accuracy, temporal response, confidence calibration, identity leakage, occlusion recovery, latency, resource, and failure-sample evidence. Use for metrics, baselines, benchmark reports, acceptance checks, or quality regressions.
---

# Evaluate tracking models

1. Read `references/evaluation-contract.md` before adding or interpreting a metric.
2. Pin the checkpoint, data revision, configuration, hardware, and backend in the report.
3. Report each output family separately; do not hide failures behind one aggregate loss.
4. Compare latency as capture-to-result and result age, not inference time alone.
5. Preserve failure samples and distinguish observed, predicted, occluded, out-of-frame, and lost.
6. Compare executor or backend performance only on representative target hardware.
7. Keep synthetic smoke reports out of production acceptance claims.
