# Spatial same-capture fusion contract

Status: framework-neutral NTP 1.0 fusion contract.

## Boundary

Sensor SDK values are normalized before this boundary. The fusion API accepts only a versioned
`NanaTrackingDescriptor` and `NanaTrackingResult`; it does not accept ARKit names, raw 52-entry
dictionaries, model tensors, ONNX Runtime values, or transport objects. The result exposes only NTP
value, confidence, state, geometry, skeleton, and quality.

The two inputs must have identical `session_id`, `generation`, `sequence`, and
`capture_timestamp_ns`. Completion time and arrival order are never synchronization keys. A product
adapter should create both inputs from the same captured-frame identity and perform extension
inference off the capture callback through a latest-frame-only worker.

For an iOS capture adapter, one capture record owns the RGB image, normalized face state, head and
eye transforms, look-at, normalized/depth geometry, camera intrinsics, tracking validity, and
capture timestamp from one camera frame. Raw teacher fields may be retained only in the governed
training capture format; they do not enter NTP streaming frames.

## Decision rules

- The reference input owns head pose, eye geometry, look-at, face geometry, and continuous gaze
  conflicts while it is available.
- The extension input fills unavailable signals and structures. For non-gaze scalar conflicts it
  replaces the reference only after a configured confidence margin.
- Agreeing values corroborate one selected value and become `Fused`; values are never averaged.
- A visible RGB tongue value replaces an occluded tongue observation. An invisible tongue remains
  `Occluded` with no numeric zero.
- Capability is the union of both descriptors. The certified profile remains the highest complete
  standard set; additional Full or optional signals are preserved and never clipped by Spatial.
- Structured fallback keeps the reference geometry when both are available and uses the extension
  only when the reference has no value.

The default tolerance and confidence margin are bounded policy, not protocol constants. Changing
them does not change Signal semantics but must be recorded in producer configuration and evaluation
evidence.

## Reconfiguration and threading

A session or layout reconfiguration increments `generation` and resets pending frame identity. A
result from an old generation is rejected. Capture callbacks do not wait for RGB extension
inference; a single replaceable slot prevents result-age growth. Transport, encryption, pairing,
and reconnect behavior remain outside this crate.

## Validation

The crate validates both descriptors and frames before fusion, validates the fused result against
the union descriptor, and runs the output through `ntp-conformance` in its acceptance test. Tests
cover exact-capture rejection, reference gaze/geometry priority, RGB tongue completion, Spatial
certification, and preservation of additional Full signals.
