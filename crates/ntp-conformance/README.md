# NTP conformance

`ntp-conformance` certifies a producer from declared capabilities and observed result frames. It
checks the NTP v1 schema and revisions, derives the highest complete profile, enforces capability
versus per-frame state, validates stream clocks and generation ordering, checks geometry and
semantic relationships, and emits stable JSON failure codes.

Canonical binary streams contain one descriptor frame followed by result frames:

```bash
cargo run -p ntp-conformance -- stream.ntp --output json
```

Diagnostic JSON Lines use tagged events and are intentionally separate from the wire format:

```json
{"kind":"descriptor","value":{...}}
{"kind":"result","value":{...}}
```

```bash
cargo run -p ntp-conformance -- --input-format jsonl stream.jsonl
```

Exit status is `0` for a passing certification, `1` for a completed failing report, and `2` for an
unreadable or malformed input stream.

Protocol legality is the default. Applications may set finite capture-to-result, sample-age, and
prediction-horizon policy limits through `ConformanceOptions`; the crate does not invent a latency
target for every producer.

The conventional `cargo-fuzz` target is under `fuzz/fuzz_targets/decode_stream.rs`. The always-run
property suite also feeds deterministic arbitrary byte buffers to both canonical decoders.
