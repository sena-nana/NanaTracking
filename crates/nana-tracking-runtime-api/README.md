# `NanaTracking` runtime API

This crate defines the stable, framework-neutral consumer boundary for model packages and model
sessions. It intentionally depends on no ML runtime and exposes only bytes, numeric slices,
timestamps, revisions, provider identity, and structured errors. Backend crates own their tensor,
device, stream, cache, and engine types.
