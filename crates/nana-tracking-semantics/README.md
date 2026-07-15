# `NanaTracking` semantics

This crate is the executable reference for deterministic semantic actions and model-side rig
bindings over NTP 1.0. It consumes the orthogonal, framework-neutral
`nana-tracking-protocol` result and never adds compatibility parameters to the network schema.

The reference formulas bind to NTP schema revision 1 and Signal Registry, normalization,
calibration, feature, and semantic revisions 1.0.0. `SemanticDeriver` owns the minimal capture-time
history required for occlusion continuity and auricle motion. A session or generation change resets
that history. `BindingEvaluator` validates target ownership before evaluating a profile.
