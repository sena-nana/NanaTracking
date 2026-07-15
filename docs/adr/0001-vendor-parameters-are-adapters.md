# ADR 0001: Vendor parameters are adapters, not NTP

- Status: Accepted
- Date: 2026-07-15
- NTP schema: `ntp/1.0`
- Signal Registry: `ntp-signals/1.0.0`

## Context

Tracking SDKs and model ecosystems expose overlapping parameter dictionaries. Copying one of
those dictionaries into NTP would duplicate signed freedoms, couple the stable ABI to a vendor's
release cycle, and force other producers and consumers to branch on backend identity.

NTP instead needs one final orthogonal state that can be produced by monocular RGB, depth fusion,
or third-party inputs and consumed by native Nana rigs or compatibility-oriented model rigs.

## Decision

ARKit and every other third-party naming scheme are outside NTP. They may appear only in:

- producer-side input adapters that normalize observations into NTP;
- consumer-side declarative binding profiles that derive model control values from NTP;
- test fixtures for those adapters and bindings.

They must not appear as stable Signal IDs, capability names, result fields, required structures, or
backend selectors in protocol logic. NTP descriptors declare profile, signals, structures, and
features—not a producer brand.

A signed NTP axis deterministically supplies both compatibility half-axes. For example, a closed
and widened eyelid binding consumes the negative and positive halves of `eye.*.aperture`; left and
right jaw bindings consume the halves of `jaw.lateral`; smile and frown bindings consume the
halves of `mouth.corner.*.vertical`. These are model-side views and are never transmitted as extra
NTP parameters.

For example, a model-side ARKit-style binding can consume one NTP frame without adding any fields
to that frame:

| NTP value | Model binding input | Deterministic value |
| --- | --- | ---: |
| `eye.left.aperture = -0.8` | `eyeBlinkLeft` | `max(-aperture, 0) = 0.8` |
| `eye.left.aperture = -0.8` | `eyeWideLeft` | `max(aperture, 0) = 0` |
| `jaw.lateral = 0.6` | `jawRight` | `max(lateral, 0) = 0.6` |
| `jaw.lateral = 0.6` | `jawLeft` | `max(-lateral, 0) = 0` |
| `mouth.corner.left.vertical = 0.5` | `mouthSmileLeft` | `max(vertical, 0) = 0.5` |
| `mouth.corner.left.vertical = 0.5` | `mouthFrownLeft` | `max(-vertical, 0) = 0` |

The complete executable binding templates belong to the semantics/binding layer, not this protocol
ADR. This example establishes the direction of dependency and proves that compatibility half-axes
do not require duplicate wire parameters.

## Consequences

- Different backends that resolve to the same normalized NTP state are indistinguishable to a
  consumer and produce the same derived model controls.
- A model authored with ARKit-style names can be supported by a binding profile without adding
  ARKit fields to NTP.
- Input adapters own vendor-version changes; binding templates own model naming changes. Neither
  silently changes a stable NTP meaning.
- A genuinely new observable freedom follows the admission rule in the Signal Registry. A vendor
  adding a new parameter is not, by itself, justification for a new NTP signal.

## Rejected alternatives

1. **Adopt a vendor dictionary as the canonical schema.** Rejected because it imports redundant
   half-axes and implementation history into a stable protocol.
2. **Carry both Nana and vendor parameters.** Rejected because independent values can disagree and
   create duplicate control paths.
3. **Select mappings by backend name in consumers.** Rejected because capabilities and semantics,
   not product identity, determine compatibility.
