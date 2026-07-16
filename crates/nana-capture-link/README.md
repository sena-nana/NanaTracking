# Nana capture Link adapter

`nana-capture-link` is the runtime-neutral owner adapter between NanaTracking capture and
MutsukiLink. It does not own capture schemas or durable storage.

The adapter consumes a one-use authorization derived from an active MutsukiLink trust record and
an authenticated Link session. Its three explicit permissions are
`nana.capture.control.v1`, `nana.capture.preview.v1`, and `nana.capture.sync.v1`.

- Control uses MutsukiLink's independent reliable control stream.
- Preview uses the existing prioritized latest-only Datagram path when the connection supports it.
  The trust record must also grant MutsukiLink's explicit `Datagram` permission. Otherwise one
  replaceable reliable preview slot is used; it can never grow into a frame queue.
- Hello exchanges the device/studio role and preview mode. Datagram is selected only when both
  authenticated peers advertise it, preventing asymmetric trust permissions from creating a
  one-way preview path.
- Dataset segments, verified acknowledgements, and missing ranges always use bounded reliable
  delivery. Each segment has an offset, total length, and SHA-256 checked before application code
  receives it.
- Every envelope is bound to the authenticated Link session. Reconnect requires a different newly
  authenticated session and resets preview replay state and queues.

The Rust boundary deliberately contains no PyTorch, ONNX Runtime, Swift, Python, NTP label, or
dataset types. `CaptureChunk`, local journals, reconciliation, label regeneration, and freeze gates
remain authoritative in their existing NanaTracking layers.

Functional tests use MutsukiLink's bounded in-memory transport to exercise permission denial,
control/data isolation under backpressure, latest-only fallback, Datagram delivery, reliable
segment/ACK/missing-range exchange, digest rejection, session mismatch, and reconnect reset.

This crate is ready for a Swift/Python binding or a native host service, but no such binding is
claimed here. The existing HTTP client/server remains the executable cross-language transport until
that deployment integration is selected and validated. iOS device and Windows evidence remain
external acceptance.
