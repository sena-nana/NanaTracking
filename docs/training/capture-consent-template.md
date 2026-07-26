# Capture consent template v1

This template requires legal review and a signed, access-controlled record; it is not itself a
completed consent.

- Participant and guardian status where applicable; de-identified identity ID.
- Modalities: synchronized multi-camera RGB, camera calibration, MediaPipe semantic landmark
  pseudo-labels, OpenCV-derived geometry/pose/reprojection evidence, optional evaluation-only
  ARKit/TrueDepth comparison outputs, device metadata, scripted actions, and quality annotations.
- Purposes: commercial NanaTracking model training, validation, internal human review, creation of
  derived labels and synthetic identity parameters, and distribution of trained model weights and
  inference products.
- Explicit exclusions: identity recognition, unrelated biometric identification, raw-data sale,
  and any unlisted use.
- Retention deadline, storage region, encryption/access policy, authorized reviewer roles, audit
  policy, and whether de-identified failure samples may be retained.
- Withdrawal/deletion route and identity-to-artifact deletion mapping, including future training
  revision invalidation.
- Whether raw or derived data may be redistributed; default is no.
- A separate opt-in states whether ARKit/TrueDepth comparison is allowed. Declining it does not
  prevent participation in the RGB training capture.
- Signature, date, consent text digest, reviewer, and approval/expiry state.

