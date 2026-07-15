# NTP evaluation standard v1

`configs/evaluation/ntp-v1-standard.json` is the executable standard. It binds the NTP and Signal
Registry revisions, pins fixed-sequence and report-template digests, and requires exactly one Basic,
Spatial, and Full suite. Profile signal coverage is nested, but all suites use the same metric set.

The shared metrics cover per-signal error and correlation, neutral jitter, dynamic response delay,
peak attenuation, anatomical left/right preservation, identity neutral bias, geometry consistency,
state classification, occlusion recovery, confidence calibration, capture-to-result latency,
result age at consume, CPU/hottest-thread/GPU/VRAM/copies/queue-depth/long-run resources, and blind
character-drive A/B video review.

Every report pins checkpoint, data revision/digest, resolved configuration digest, NTP, Signal
Registry, normalization, calibration, and feature revisions, Git commit, hardware, backend, and
whether evidence is smoke-only. Results stay separate by output family; a single aggregate loss
cannot hide a failed family. Failure sample IDs and A/B video URIs remain attached to the report.

Latency comes from authoritative capture, produced, and consume timestamps. Inference-only timing
cannot substitute for capture-to-result or result age. State evaluation distinguishes observed,
predicted, occluded, out-of-frame, and tracking-lost frames. Runtime comparisons are valid only on
representative target hardware.

Run:

```bash
uv run --extra cpu nana-tracking evaluation validate-standard \
  configs/evaluation/ntp-v1-standard.json
```

Checked-in fixed sequences and templates are contracts, not benchmark results. Any report using the
synthetic dataset or smoke model must set `smoke_only: true` and cannot support production quality,
latency, or FaceBasic acceptance claims.
