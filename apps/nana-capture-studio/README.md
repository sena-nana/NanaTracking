# Nana Capture Studio core

This directory documents the implemented Capture Studio surface for Issue #16. The operator UI and
authenticated API are served by `nana-tracking studio serve`; the durable domain model lives in
`nana_tracking.data.studio`, and the dataset commands live in `nana_tracking.data.capture`. The same
CPython 3.14 code is cross-platform; this change validates it on macOS and does not claim Windows
acceptance.

Implemented behavior:

- durable descriptor, length, and SHA-256 verification before ACK;
- receiver inventory and exact missing/mismatched range reconciliation;
- restart-safe sender pending state;
- immutable raw ARKit to versioned NTP teacher-label regeneration;
- license, consent/session consistency, identity split, held-out device, and frozen-digest gates;
- latest-only preview handoff in the backend without coupling preview loss to durable recording.
- a functional browser operator surface for session creation, Take controls, preview, quality state,
  command acknowledgement, and received-chunk progress;
- localhost-by-default binding and mandatory bearer authentication plus TLS for a non-loopback bind.

`crates/nana-capture-link` supplies the authenticated MutsukiLink control/preview/reliable-sync
owner boundary, including bounded backpressure and Link-session replay protection. The CPython
Studio does not yet bind that Rust crate; its implemented HTTP API remains the executable transport.
This distinction prevents the operator surface from implying an unconnected native transport.

See `docs/data/capture-archive-v1.md` for commands and recovery rules. Manual label annotation, an
installer, Windows integration tests, and RTX performance remain external acceptance.
The UI never treats preview bytes as training input and never shows a control that lacks a backend
command.
