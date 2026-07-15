# Nana Capture iOS core

This Swift package is the local-first capture core for NanaTracking Issue #16. It is not a mock UI
and does not display unsupported controls.

- `LocalChunkRecorder` writes bytes and synchronizes them before appending the durable chunk
  journal. A network sender may acknowledge a chunk only after the receiver performs the same
  descriptor, length, and SHA-256 verification.
- A restarted recorder reconstructs pending synchronization work from the local journals; preview
  frames are not part of this package's durable chunk contract.
- On iOS with ARKit, `ARKitCapturePipeline` extracts RGB, raw blendshapes, face/eye transforms,
  geometry, camera intrinsics, tracking state, and optional caller-supplied depth from the same
  `ARFrame`. Exposure/ISO metadata is required from the app camera integration rather than filled
  with invented defaults.
- Raw ARKit JSON remains separate from NTP labels. The Python capture tooling regenerates labels
  under an explicit mapping revision.
- `NTPContract` and `NTPCanonicalCodec` provide framework-neutral Swift value types and the same
  canonical `NTP1` binary format as Rust. `NTPSpatialProducer` accepts only a complete, normalized
  same-capture Spatial payload; it owns session generation and sequence state and cannot emit raw
  ARKit names or Swift struct memory as a network ABI.
- `NTPLatestFrameWorker` gives RGB inference one replaceable pending slot. A capture callback only
  swaps the pending value; it never waits for inference or builds an unbounded frame queue.
- `CaptureStudioClient` polls authenticated control commands, posts applied-command ACKs and quality
  samples, publishes latest-only previews, and uploads durable chunk files without base64 expansion.
  `LocalChunkRecorder.synchronizePending` records the Studio ACK only after the returned ID and digest
  match the local descriptor.

Run the portable local persistence self-test on macOS. The command-line Swift bundle on the
repository host does not ship XCTest or Swift Testing, so this executable performs the same
cross-language Rust-vector round-trip, Spatial generation lifecycle, latest-only scheduling,
restart, pending-retry, corrupt-payload, and acknowledgement assertions without a test framework:

```bash
swift run --package-path apps/nana-capture-ios NanaCaptureSelfTest
```

The repository host currently has Swift command-line tools but not full Xcode. Building the iOS
ARKit branch, signing an app, TrueDepth device capture, lifecycle/background behavior, and App UI
remain device/Xcode acceptance work. The cross-platform Studio backend/UI is implemented separately;
its Windows and RTX workflow remains Windows-side acceptance work.
