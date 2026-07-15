# NTP v1 freeze review checklist

Scope revisions: `ntp/1.0`, `ntp-signals/1.0.0`, `ntp-normalization/1.0.0`,
`ntp-calibration/1.0.0`, `ntp-features/1.0.0`.

This checklist gates implementation work that depends on Issue #2. A later compatible addition
must repeat the applicable checks under a new registry or feature revision; a semantic change to a
published item requires an NTP major revision.

| Review gate | Frozen v1 decision |
| --- | --- |
| Stable scalar inventory | IDs `1..76` are Core; `77..88` are optional fine features; IDs are unique and never reused. |
| Orthogonality | Signed axes replace positive/negative aliases; semantics, velocity, phase, and region relationships are derived. |
| Set nesting | Basic has 36, Spatial adds 5 for 41, Full adds 35 for 76; both inclusions are proper. |
| Per-ID metadata | Every row resolves range, unit, neutral, soft limit, polarity, symmetry, and lowest guaranteed set. |
| Profile semantics | Profile is the highest fully guaranteed set, while extra capabilities remain independently advertised. |
| Structured state | Head pose/timing/quality, Spatial eye/face geometry, and Full torso/skeleton semantics are defined. |
| Duplicate pose views | Torso/head/arm scalar and structured representations are constrained views of one state, not independent freedoms. |
| Coordinates | Camera, torso-local, and head-local spaces, handedness, axes, anatomical sides, and display mirroring are fixed. |
| Rotations | Unit canonical Hamilton quaternion is authoritative; Euler ordering is limited to Rig views. |
| Scale and depth | Metric and relative bases are explicit; monocular relative Z cannot be labelled metres. |
| Time | Session/generation/sequence, capture/produced timestamps, sample age, and prediction horizon are distinct. |
| Quality | Value, confidence, and state are independent; invalid/missing state cannot be represented as a fabricated zero. |
| Framework boundary | No protocol or consumer contract requires Python, PyTorch, ONNX Runtime, or another backend type. |
| Vendor boundary | Third-party parameter names are excluded from the registry and handled only by adapters/binding templates per ADR 0001. |
| Extensibility | Unknown additive IDs are ignorable; stable IDs are not reused; every new signal must prove non-derivability. |

Implementation reviewers must compare generated protocol constants, training label maps, model
heads, artifact metadata, and binding/conformance vectors against these exact revisions. Any
silent mismatch blocks release.
