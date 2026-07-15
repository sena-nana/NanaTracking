# NTP guided collection protocol v1

## Participant experience

The participant sees only the current motion, a simple demonstration, progress, and a clear retry
or stop choice. User-facing states are `Get ready`, `Start`, `Hold`, `Rest`, `Done`, and `Try again`.
Clock synchronization, label names, teacher disagreement, model details, and quality thresholds are
recorded internally and are not shown as technical UI text.

Before capture, the flow explains what is recorded, the purpose, retention, optional actions, and
how to withdraw. Capture never starts before explicit consent. Tongue, auricle, extreme range, and
accessory removal are optional and may be skipped without presenting an error.

Each action has a demonstration, a comfortable-range reminder, three reviewed repetitions, and a
neutral/rest interval. The operator can mark discomfort, instruction failure, or teacher failure;
those takes are retained only as unavailable/failure metadata unless the participant requests
deletion.

## Basic plan

- Several natural neutral faces and held neutral segments.
- Left/right independent close, widen, and squint; then bilateral motion.
- Left/right inner, outer, and medial brow changes.
- Puff and suck each cheek, cheek raise, and left/right nose motion.
- Jaw open, lateral shift, retraction, and protraction.
- Mouth-corner vertical/horizontal motion; upper/lower lip motion; seal, protrusion, roundness,
  upper/lower roll, left/right press, and dimples.
- Natural speech followed by reviewed rapid phoneme transitions.

## Spatial plan

- Continuous gaze through eight directions and reviewed asymmetric gaze targets.
- Large head angles while gaze remains fixed, then gaze motion while the head remains fixed.
- Optional tongue extension only when the reviewed teacher sees it directly.
- Repeated gaze/head actions in dim light, glasses reflection, and partial occlusion.

## Full plan

- Torso translation, rotation, side bend, forward lean, and backward lean.
- Head motion independent of a held torso, then compensating head/torso motion.
- Left/right shoulder elevation, depression, retraction, and protraction.
- Left/right forward raise, side raise, overhead raise, elbow bend, arm crossing, and hand near face.
- Each wrist deliberately exits and re-enters the frame.
- Optional voluntary auricle elevation, retraction/protraction, and flattening for users whose motion
  is directly visible; this is a personalization subset, not a population-wide required label.
- Repetitions with hair, headphones, loose clothing, and self-occlusion.

## Capture controls

RGB and available TrueDepth frames are acquired from the same capture session and retain original
timestamps and sequence numbers. Exposure, ISO, frame duration, resolution, intrinsics, and
distortion are recorded automatically. The operator records device, session, and de-identified
identity groups; the UI never asks the participant to type private identity into a prompt.

An action completes only when required duration, repetitions, visibility, and timestamp continuity
are satisfied. A retry creates a new take and never overwrites the failed one. The failed take is
tagged for the failure flow, while label materialization decides each signal independently.
