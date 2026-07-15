---
name: manage-training-data
description: Manage NanaTracking training data schemas, manifests, collection plans, provenance, identity-safe splits, privacy, licensing, and quality gates. Use for dataset ingestion, teacher labels, augmentation inputs, failure-sample feedback, or any change under the data pipeline.
---

# Manage training data

1. Read `references/data-contract.md` before changing a manifest, split, or collection workflow.
2. Identify the schema, data, NTP, and Signal Registry revisions affected by the change.
3. Preserve identity/session/device grouping and reject identity overlap across splits.
4. Keep raw recordings and private metadata outside Git; commit only reviewed manifests, digests,
   schemas, and tiny fixtures.
5. Record teacher provenance and verify that licenses permit the intended training or distillation.
6. Run `uv run --extra cpu nana-tracking data validate <manifest>` and the data tests.
7. Report unavailable or unreliable labels explicitly; never fabricate a certain target.
