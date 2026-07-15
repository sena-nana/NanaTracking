# First-party capture action script v1

Each session records the script revision, de-identified identity/session/environment/device IDs,
camera calibration, consent-record ID, reviewer state, and authoritative monotonic timestamps.

The operator captures neutral holds, isolated brows/eyes/cheeks/nose/jaw/mouth actions, symmetric
and asymmetric combinations, slow and fast onset/peak/recovery, blink and jaw events, visemes both
alone and combined with expression, head rotations/translations, glasses/hair/hand occlusion, dim
and backlit conditions, and deliberate re-entry after out-of-frame loss. Participants may stop or
skip any action.

RGB, camera metadata, ARKit/TrueDepth parameters, teacher mesh/head pose, depth and confidence are
synchronized without rewriting timestamps. A human reviewer records approved/rejected/pending and
per-label confidence. Reviewers never fill missing truth with zero. Rejected, withdrawn, expired,
unsynchronized, or SDK-license-ineligible sessions cannot enter a production manifest.

