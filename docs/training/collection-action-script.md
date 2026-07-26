# First-party capture action script v1

Each session records the script revision, de-identified identity/session/environment/device IDs,
camera calibration, consent-record ID, reviewer state, and authoritative monotonic timestamps.

The operator captures neutral holds, isolated brows/eyes/cheeks/nose/jaw/mouth actions, symmetric
and asymmetric combinations, slow and fast onset/peak/recovery, blink and jaw events, visemes both
alone and combined with expression, head rotations/translations, glasses/hair/hand occlusion, dim
and backlit conditions, and deliberate re-entry after out-of-frame loss. Participants may stop or
skip any action.

Training sessions synchronize front and approximately left/right 30-degree RGB, camera intrinsics,
extrinsics, and timestamps without rewriting them. The pinned MediaPipe bundle proposes only the
reviewed 16 semantic 2D anchors; OpenCV derives triangulated geometry, pose, reprojection residuals,
and optical-flow consistency. A human reviewer records approved/rejected/pending and per-label
confidence. Reviewers never fill missing truth with zero.

ARKit/TrueDepth may be recorded only in separately declared validation/test comparison sessions.
Those outputs are excluded from training manifests, pseudo-label caches, calibration, threshold
selection, recipe selection, and checkpoint ancestry. Rejected, withdrawn, expired,
unsynchronized, uncalibrated, or license-ineligible sessions fail closed.

