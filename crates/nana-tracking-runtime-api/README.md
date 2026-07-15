# `NanaTracking` runtime API

This crate defines the stable, framework-neutral consumer boundary for model packages and model
sessions. It intentionally depends on no ML runtime and exposes only bytes, numeric slices,
timestamps, revisions, provider identity, and structured errors. Backend crates own their tensor,
device, stream, cache, and engine types.

Tracked scalars and structures keep value presence, confidence, state, capture timestamp, and
prediction horizon separate. `Occluded`, `OutOfFrame`, and `TrackingLost` therefore remain
distinguishable from `Unsupported` without inserting a numeric zero.
