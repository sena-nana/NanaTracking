# NTP v1 canonical codec and Rust/C contract

- NTP schema: `ntp/1.0`, numeric schema revision `1`
- Signal Registry: `ntp-signals/1.0.0`
- Normalization: `ntp-normalization/1.0.0`
- Calibration: `ntp-calibration/1.0.0`
- Feature registry: `ntp-features/1.0.0`
- Reference crate: `crates/nana-tracking-protocol`

The Signal Registry remains the normative semantic source. This document fixes the reference
implementation and encoding boundary for Issue #3; it does not add or reinterpret a Signal ID.

## Contract layout

`NanaRigResult` always has 88 stable slots. Slot `n` is Signal ID `n + 1`; a backend cannot rename,
remove, or reorder a slot. Every slot carries an optional value, confidence, state, sample capture
time, and prediction horizon. `Unsupported` is the default slot state and never masquerades as a
numeric zero.

`NanaGeometryResult` fixes head pose, left/right eye origin and direction, look-at, normalized face
geometry state, and stable semantic landmarks. `NanaSkeletonResult` fixes torso, shoulder, elbow,
wrist, upper-arm/forearm direction, and twist fields for both anatomical sides. Quality keeps the
overall confidence and face, eyes, torso, left/right arm, and left/right auricle states independent
from every signal state. The stabilization revision says which semantic stabilization contract was
applied without naming an algorithm, SDK, or backend.

`NanaTrackingDescriptor::from_capabilities` derives the highest completely satisfied profile. Extra
signals remain advertised independently. Dense-face-mesh and wrist-pose feature bits are rejected
unless their required structure capability exists. Metric-coordinate capability requires at least
one structure. Experimental IDs `0x8000..0xffff` are rejected for strict v1 producers.

## Canonical binary framing

The wire header is 12 bytes: ASCII `NTP1`, message kind, wire major, wire minor, one reserved byte,
and a little-endian `u32` payload length. The payload is an ordered sequence of `u16 tag + u32
length + bytes` fields. Descriptor tags encode revisions, derived profile, a sorted sparse Signal ID
list, structure bits, and feature bits. Result tags encode the envelope, rig, geometry, skeleton,
and quality blocks.

The in-memory 88-slot rig is encoded sparsely as sorted `Signal ID + entry length + sample` records.
Unsupported slots need no per-frame bytes. An old reader skips unknown additive Signal IDs, unknown
top-level structure blocks, and compatible trailing block fields. Required known fields cannot be
duplicated or omitted. Existing-field meaning changes require a protocol major revision.

All integers and IEEE-754 `f32` values are little-endian. Non-finite and out-of-contract values are
rejected before encoding and after decoding. Negative zero is encoded as positive zero. Quaternion
sign is canonicalized according to the Signal Registry, so equivalent `q` and `-q` values have one
byte representation. The fixed vector under `tests/vectors` pins descriptor bytes across languages.

Rust layout and C ABI padding are never serialized. `ffi::NtpDescriptorC` and `ffi::NtpCoreResultC`
are separately versioned `repr(C)` conversion views. The core C view contains all fixed v1 fields;
variable future landmark/mesh extension arrays stay in the canonical codec until assigned their own
C ABI revision. C tracking-state constants intentionally map `Unsupported` to zero, so zero-initialized
sample and structure views are safely unavailable rather than falsely observed.

Diagnostic JSON is behind the `diagnostic-json` feature and is explicitly non-canonical. Consumers
must not use JSON as the network, storage, equality, or signature format.

## Negotiated live compact frames

The canonical codec is not the high-frequency network representation. Live scalar streaming uses
the separately versioned `NTC1` codec after an immutable `ActiveLayout` is validated and confirmed.
Frames have a fixed 56-byte header, a dense ordered `i16` value plane, and the confirmed fixed-width
quality plane; per-frame Signal IDs, counts, offsets, maps, and nullable objects are forbidden.
Layout hashing, quantization, exact-size parsing, bounded negotiation, replay/time guards, recording
requirements, and transport ownership are frozen in [ADR 0003](../adr/0003-negotiated-compact-frames.md).
The canonical `NTP1` result remains the owned/debug/structured contract and is never confused with
the compact data plane.

## Session safety

`ResultStreamGuard` accepts only the negotiated `session_id` and `generation`, and requires strictly
increasing sequence numbers. Gaps are reported but accepted. An algorithm/calibration/basis switch
must explicitly advance generation before its first frame; late old-generation frames remain
rejected. A replacement session must have a different ID, preventing sequence reset under the same
clock/calibration identity.

## `no_std` / `alloc` assessment

The protocol core, validation, codec, stream guard, and C conversion layer compile with `no_std` and
`alloc` using:

```bash
cargo check --workspace --no-default-features
```

The default `std` feature implements standard error traits. `diagnostic-json` requires `std` and
`serde_json`; disabling it does not remove the canonical binary codec. The only heap-backed core
data are the session-level full-domain capability bitmap, fixed rig slots, and explicitly variable
future landmark collections. There is no framework, runtime, transport, camera, or UI dependency.
