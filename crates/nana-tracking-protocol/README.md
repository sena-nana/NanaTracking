# nana-tracking-protocol

Framework-neutral NTP v1 contracts, fixed stable-signal slots, capability/profile validation,
canonical binary encoding, and a C ABI conversion layer. The crate contains no tracking SDK,
model runtime, UI, camera, or transport dependency.

The Rust structures are semantic contracts. `CanonicalCodec` is the network/storage encoding;
`ffi` contains separate `repr(C)` views. Neither Rust nor C layout is used as the wire format.

