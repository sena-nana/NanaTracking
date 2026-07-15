---
name: release-onnx-models
description: Export, parity-check, verify, and package NanaTracking PyTorch checkpoints as portable ONNX model artifacts with metadata, digests, schemas, and fixed test vectors. Use for ONNX export, opset, runtime compatibility, release artifacts, or backend parity.
---

# Release ONNX models

1. Read `references/model-package.md` before changing export or packaging.
2. Export only from a recorded checkpoint and resolved configuration.
3. Compare every named PyTorch output with ONNX Runtime CPU before packaging.
4. Fail on missing metadata, undeclared custom operators, digest mismatch, or tolerance violation.
5. Keep PyTorch checkpoints out of deployable model packages.
6. Keep test vectors in interoperable JSON/NPZ formats, not Python-only cache formats.
7. Run `nana-tracking verify-export` and integration tests before declaring an artifact ready.
