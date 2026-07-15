---
name: align-ntp-contracts
description: Align NanaTracking training labels, model heads, normalization, metadata, and artifacts with versioned NTP contracts while preventing ML framework types from leaking into protocol or consumer interfaces. Use for signal mappings, output schemas, temporal state, confidence, geometry, or revision compatibility.
---

# Align NTP contracts

1. Read `references/ntp-boundary.md` before changing model inputs, outputs, or metadata.
2. Name the NTP schema, Signal Registry, normalization, calibration, and feature revisions involved.
3. Represent outputs as framework-neutral named values at artifact boundaries.
4. Preserve value, confidence, and tracking state semantics independently.
5. Reject silent semantic changes; require a revision and compatibility decision.
6. Verify that protocol and consumer code need no Python, PyTorch, ONNX Runtime, or backend type.
7. Add contract and fixed-vector tests for every mapping change.
