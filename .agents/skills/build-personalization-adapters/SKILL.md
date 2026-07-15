---
name: build-personalization-adapters
description: Build safe NanaTracking personalization through Level A numerical calibration, Level B offline PyTorch-to-ONNX residual adapters, or optional Level C bounded online adapters. Use for calibration, user profiles, adapters, compatibility, rollback, reset, or drift protection.
---

# Build personalization adapters

1. Read `references/personalization-contract.md` before changing profiles or adapters.
2. Prefer Level A numerical calibration unless evidence shows a learned adapter is necessary.
3. Freeze the base model and make learned outputs residual corrections.
4. Bind profiles to model family/version, feature revision, Signal Registry revision, and user slot.
5. Reject incompatible profiles safely; preserve reset, cancellation, and rollback.
6. Never share a learned adapter across users automatically or train the complete encoder on device.
7. Compare improvement against Level A and keep the adapter disabled when benefit is insufficient.
