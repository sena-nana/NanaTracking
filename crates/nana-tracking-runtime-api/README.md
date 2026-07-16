# `NanaTracking` runtime API

This crate defines the stable, framework-neutral consumer boundary for model packages and model
sessions. It intentionally depends on no ML runtime and exposes only bytes, numeric slices,
timestamps, revisions, provider identity, and structured errors. Backend crates own their tensor,
device, stream, cache, and engine types.

Inputs carry capture time and backend processing-start time in one monotonic clock domain. This
lets a backend include mailbox or scheduler delay in `produced_timestamp_ns` instead of pretending
inference began at capture, while stage telemetry remains split by backend work.

Tracked scalars and structures keep value presence, confidence, state, capture timestamp, and
prediction horizon separate. `Occluded`, `OutOfFrame`, and `TrackingLost` therefore remain
distinguishable from `Unsupported` without inserting a numeric zero.

`ActiveProvider` reports the backend that actually executed the graph. The ORT Core ML variant also
records whether profiled graph nodes fell back to ORT CPU; merely requesting an execution provider
is not sufficient to select that variant.
