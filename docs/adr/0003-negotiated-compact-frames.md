# ADR 0003: negotiate immutable compact frame layouts

Status: accepted for NTP v1

## Decision

NTP keeps two deliberately separate binary surfaces:

- the canonical `NTP1` descriptor/result codec is the owned, extensible contract used for local
  interchange, fixed vectors, recording expansion, FFI conversion, and diagnostics;
- the negotiated `NTC1` compact codec is the live scalar data plane. It is legal only after both
  peers confirm one immutable `ActiveLayout` ID and SHA-256 hash.

`BasicV1`, `SpatialV1`, and `FullV1` contain stable Signal IDs `1..=36`, `1..=41`, and `1..=76` in
that order. A producer may append any other stable ID that its descriptor advertises, including a
Basic producer appending Spatial, Full, or optional signals. The descriptor guarantee is a floor,
not a filter. Experimental, unknown, duplicate, unsupported, and base-duplicating IDs fail closed.

The compact scalar plane does not replace the canonical structured geometry and skeleton
contracts. Producers must validate `NanaTrackingResult` structure state through the descriptor and
the existing `Validate` implementation before selecting scalar samples. A later compact layout
version may assign fixed structured blocks; it cannot silently add them to CompactV1.

## Canonical layout hash

The SHA-256 input is a byte-exact little-endian sequence:

1. domain `NTP-COMPACT-LAYOUT\0`;
2. protocol version and schema revision;
3. Signal Registry, normalization, calibration, and feature revisions;
4. profile, base-layout version, value encoding, quality encoding, and requested FPS;
5. `u16` parameter count followed by every ordered `u16` Signal ID.

The connection-local `layout_id` is not part of the semantic hash. `LayoutAccept` carries the ID,
hash, parameter count, and derived exact frame length; `LayoutConfirm` must echo the ID and hash.
The machine-readable vector at
`crates/ntp-conformance/tests/vectors/compact-basic-extras-v1.json` pins this algorithm for other
languages.

## CompactV1 frame

All integers are little-endian. The header is exactly 56 bytes:

| Offset | Width | Meaning |
| ---: | ---: | --- |
| 0 | 4 | ASCII `NTC1` |
| 4 | 1 | compact wire version `1` |
| 5 | 1 | confirmed quality encoding |
| 6 | 2 | reserved, must be zero |
| 8 | 16 | session ID |
| 24 | 4 | generation |
| 28 | 4 | layout ID |
| 32 | 8 | sequence |
| 40 | 8 | capture timestamp ns |
| 48 | 8 | produced timestamp ns |

The header is followed by a contiguous `i16[parameter_count]` value plane. With
`StateAndConfidenceU8`, a contiguous two-byte `(SignalState, confidence)` quality plane follows in
the same order. No frame contains a count, Signal ID, offset, optional object, map, backend name, or
dynamic type. The receiver accepts only the exact length derived from its local confirmed layout.

`I16Normalized` maps each registry range linearly onto `-32767..=32767` using deterministic
nearest rounding. `-32768` is reserved for missing values and is legal only with a non-value state.
The exclusive upper endpoint for `Angle` therefore rejects wire value `32767`. Producers reject
non-finite and out-of-range source values rather than silently clamping them. `Unsupported` is not
legal inside an active layout; a valid neutral zero remains distinct from missing.

## State, resource, and transport boundaries

`LayoutNegotiator` limits pending layouts to at most two, applies a monotonic renegotiation rate
budget, and charges invalid proposals to that budget. `CompactStreamGuard` requires confirmation,
checks session/generation/layout, rejects replay and excessive gaps, and applies age, future-skew,
capture-regression, and capture-jump policies without advancing state on failure. Since producer and
receiver monotonic clocks have unrelated epochs, the transport adapter must pass a bounded
`ProducerClockEstimate` mapped into the producer clock domain. The guard includes its uncertainty in
age/skew limits and rejects estimates above the configured uncertainty ceiling. Layout switches must
advance generation and clear sequence/time history. `LatestFrame` is a single-slot handoff and counts
overwritten unread frames.

The codec encodes into caller-owned exact-size storage and decodes as a borrowed view. It does not
allocate per frame, recurse, build object graphs, or depend on a network implementation.
`CompactSessionTransport` is the only adapter-shaped boundary; discovery, pairing, authentication,
encryption, reliable control delivery, datagram policy, reconnect proof, and disconnect decisions
belong to MutsukiLink or another transport.

Recorders must write `LayoutRecord` before frames that reference it and write a new record for every
generation/layout switch. Debug JSON may expand ordered samples, but JSON is never the real-time
wire format.

## Evidence

`ntp-conformance` covers the golden hash, all 88 scalar ranges, malformed/truncated/trailing frames,
layout validation, handshake bounds, state/value consistency, replay, time policy, and layout
switches. Its fuzz target feeds arbitrary bytes to both canonical codecs and a fixed confirmed
compact layout. The release microbenchmark is synthetic smoke-only and writes the reviewed artifact
under `artifacts/benchmarks/issue14-compact-frame-macos-arm64-smoke.json`. Version 2 separately
measures full stream-guard acceptance, including the producer-clock uncertainty check. It does not
establish tracking quality or production readiness.
