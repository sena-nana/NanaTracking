# Local-first capture archive v1

Status: executable local and synthetic-smoke implementation for Issue #16. The archive contract,
macOS-hosted studio CLI, label regeneration, split gates, and frozen-dataset verification are
implemented. The cross-platform operator UI and authenticated Studio API are implemented and tested
on macOS. Real TrueDepth capture still requires a signed iOS application and device; Windows runtime,
installer, RTX throughput, and Windows recovery acceptance are not implied by the local evidence.

## Boundaries

The iOS recorder owns the authoritative bytes. It records locally before any network transfer and
never treats preview delivery as persistence. A studio is a verified replica and may acknowledge a
chunk only after its bytes have been atomically stored and its length and SHA-256 match the sender's
descriptor.

The durable model has four layers:

1. `CaptureChunk` describes one bounded sequence range and one RGB, depth, ARKit, geometry, or
   camera payload. Paths are normalized and relative; IDs and paths cannot be reused.
2. `CaptureSessionManifest` freezes subject, session, device, consent, license, mapping revision,
   timing, and every chunk digest under its own canonical digest.
3. `RawArkitFrame` preserves the same-frame RGB reference, camera intrinsics, raw blendshapes,
   head and eye transforms, geometry/depth references, tracking state, and capture conditions.
   Revision `nana-raw-arkit-frame/1.1.0` additionally records the independently sampled TrueDepth
   timestamp, dimensions, float32-metre encoding, accuracy/quality/filter state, and whether a
   numeric confidence source actually exists. Readers remain compatible with recorded v1.0 frames.
4. `FrozenCaptureDataset` binds verified sessions, regenerated `ntp-capture/1.0.0` records, the
   license registry, admitted license records, identity/device-isolated splits, and a data revision.

Raw ARKit values are not NTP signals. `ArkitMapping` is a versioned, framework-neutral derivation
contract. Re-running a new mapping creates new teacher labels without mutating raw capture bytes.
Out-of-range mapped values fail validation instead of being silently clamped. The checked-in
`configs/data/arkit-to-ntp-v1-smoke.json` covers only a deliberately small synthetic subset and is
not a reviewed production mapping.

## Recording and recovery

`apps/nana-capture-ios` contains a Swift 6 local-first core. On an iOS ARKit build,
`ARKitCaptureSessionController` owns a real face-tracking `ARSession`, camera permission, Studio
control transitions, and a single-task/latest-frame worker. `ARKitCapturePipeline` takes all face
fields from one `ARFrame`, reads per-frame EXIF exposure where available, records front-camera
`capturedDepthData` under its separate timestamp, and writes payloads through
`LocalChunkRecorder`. RGB/geometry/depth URIs use the recorder's single canonical path function,
so raw records cannot point at a filename different from the durable chunk descriptor. The
portable self-test proves bounded asynchronous scheduling, durable restart, pending retry,
corrupt-payload rejection, and acknowledgement handling:

```bash
swift run --package-path apps/nana-capture-ios NanaCaptureSelfTest
swift run --package-path apps/nana-capture-ios -c release NanaCaptureSchedulingBenchmark
```

Chunk bytes are synchronized before their descriptor is appended to `chunks.jsonl`; acknowledgements
are synchronized to a separate append-only journal. On restart, a complete final JSON record that
lost only its newline is retained and repaired; an incomplete crash-torn tail is durably truncated.
Invalid newline-terminated records and duplicates still fail closed. The journals are then indexed
once, and new chunk/acknowledgement checks use in-memory ID and path indexes rather than rescanning
the growing journal. Callers should use bounded multi-frame chunk ranges instead of an unbounded
recording or one growing file.

Recovery is deterministic:

1. Reopen the sender store and enumerate `capture-pending`.
2. Ask the receiver for `capture-receiver-index`.
3. Reconcile the receiver index against the finalized session.
4. Retransmit missing or digest-mismatched ranges from the local source.
5. Record an acknowledgement on the sender only after the receiver has emitted its verified ACK.
6. Run `capture-verify` before deleting any device-local copy.

The receiver commands are transport-neutral building blocks. `studio serve` provides the matching
HTTP transport: raw file upload, bearer authentication, TLS for non-loopback binding, bounded request
sizes, and exact receiver ACKs. The iOS client performs retry from the local pending journal. A
deployment still owns certificate/token provisioning and network retry policy.

`crates/nana-capture-link` implements the preferred MutsukiLink owner adapter without changing the
archive contract. It derives one-use capture authorization from an active trust record and an
authenticated Link session; binds all envelopes to that session; keeps control on its independent
reliable stream; sends preview through the existing prioritized latest-only Datagram path or one
replaceable reliable slot (Datagram also requires its explicit MutsukiLink trust permission); and
sends dataset segments, durable ACKs, and missing ranges only through bounded reliable delivery.
Segment SHA-256 is checked before an event reaches the owner. Reconnect
requires a different authenticated Link session so replay state cannot be reset in place.
Hello exchanges the device/studio role and preview mode; Datagram is used only when both peers
advertise it, so asymmetric trust permissions fail closed to reliable latest-only delivery.

The adapter is intentionally runtime-neutral Rust. It is not yet a Swift or CPython binding, and the
functional HTTP path above remains the current cross-language deployment. Local memory-transport
tests prove the owner scheduling, security, and recovery envelopes; they do not prove discovery,
pairing UI, iOS lifecycle, Windows installation, or device networking.

The checked-in release smoke forces reliable transport backpressure for 200,000 preview submissions
and then performs 64 MiB of reliable segment encode/receive/decode/SHA-256 work. Five Apple M4 runs
measured a median 18.872 ns per owned single-slot preview submission and 192.187 MiB/s verified
segment throughput. See
`artifacts/benchmarks/issue16-capture-link-macos-m4-smoke.json`. These are synthetic in-memory
numbers only, not Wi-Fi, USB, iPhone, Windows, thermal, or production evidence.

## Studio CLI

Create and run a Studio session:

```bash
uv run --extra cpu nana-tracking studio create studio/session-1 \
  --session-id session-1 --subject-id subject-1 --device-id iphone-1 \
  --device-model iPhone17,1 --os-version "iOS 20" \
  --ntp-mapping-revision arkit-to-ntp/1.0.0-smoke \
  --consent-record-id consent-1 --license-records nana-synthetic-smoke

# Loopback needs no token. LAN binding additionally requires --token-file, --tls-cert, and --tls-key.
uv run --extra cpu nana-tracking studio serve studio/session-1 \
  --host 127.0.0.1 --port 8765
```

The UI at `http://127.0.0.1:8765` creates sessions, issues validated start/pause/stop/retake/end
commands, shows the latest preview and quality state, and reports command ACK and chunk progress.
Equivalent automation is available through `studio state`, `studio control`, and `studio finalize`.

The lower-level receiver primitives remain useful for offline import and recovery:

```bash
# Receiver: store and verify one chunk, then emit its ACK JSON.
uv run --extra cpu nana-tracking data capture-receive \
  studio/session-1 chunk.json chunk.bin

# Receiver: export all durable ACK descriptors for reconciliation.
uv run --extra cpu nana-tracking data capture-receiver-index \
  studio/session-1 > receiver-index.json

# Sender: list locally durable chunks that have no recorded ACK.
uv run --extra cpu nana-tracking data capture-pending ios/session-1

# Compare a finalized session to the receiver; non-complete results exit nonzero.
uv run --extra cpu nana-tracking data capture-reconcile \
  ios/session-1/session.json receiver-index.json

# Verify all session paths, lengths, digests, and the manifest digest.
uv run --extra cpu nana-tracking data capture-verify ios/session-1/session.json
```

Live preview is intentionally latest-only and non-durable. `LatestPreview` has one pending slot;
publishing a newer frame drops the stale preview rather than queuing work behind current capture.
Preview drops never remove or acknowledge training chunks.

The checked-in macOS scheduling smoke holds the first asynchronous item busy while submitting
200,000 replacements. Exactly the first and latest values are processed, 199,999 stale pending
values are counted as dropped, and five release runs measured 45.91 ns median per submission on an
Apple M4. See `artifacts/benchmarks/issue16-ios-capture-scheduling-macos-m4-smoke.json`. This proves
the bounded handoff implementation only; it is not iPhone, ARKit, storage, thermal, or battery
evidence.

Apple documents in [`ARFrame.capturedDepthData`](https://developer.apple.com/documentation/arkit/arframe/captureddepthdata)
that front-camera depth can run at a different cadence from the color feed, so a frame may have no
depth and a present depth sample may carry a different capture timestamp. The v1.1 raw record never
substitutes the RGB timestamp for that sample. `AVDepthData` exposes map accuracy/quality but no
per-frame scalar confidence; the controller therefore records confidence source `unavailable` and
value `0`, rather than inventing a teacher confidence.

## Derivation and freeze gate

Keep raw recordings outside Git. Once reviewed sessions have fully synchronized, derive records and
freeze a dataset revision:

```bash
uv run --extra cpu nana-tracking data capture-convert-arkit raw-arkit.jsonl \
  --mapping configs/data/arkit-to-ntp-v1-smoke.json \
  --output derived/capture-records.jsonl

uv run --extra cpu nana-tracking data capture-freeze \
  --session-manifests captures/a/session.json,captures/b/session.json,captures/c/session.json \
  --capture-records derived/capture-records.jsonl \
  --arkit-mappings configs/data/arkit-to-ntp-v1-smoke.json \
  --license-registry configs/data/license-registry.json \
  --license-records nana-synthetic-smoke \
  --held-out-test-devices heldout-device \
  --validation-identities 1 \
  --data-revision reviewed-capture-v1 \
  --smoke-only \
  --output frozen/frozen-capture.json

uv run --extra cpu nana-tracking data capture-verify-frozen \
  frozen/frozen-capture.json

uv run --extra cpu nana-tracking data capture-build-training-manifest \
  frozen/frozen-capture.json \
  --label-catalog configs/data/ntp-v1-label-catalog.json \
  --output frozen/training-manifest.json
```

Freeze fails if session files are absent or changed, derived records disagree with session
identity/device/consent, license text is missing or not admitted for the requested stage, the same
identity crosses splits, the held-out device rule is violated, or any frozen reference changes.
Synthetic licenses cannot pass a production freeze. The freeze copies derived records into the
frozen revision with RGB/depth URIs rewritten to their verified session chunks, pins every mapping
file and license registry digest, and checks both teacher-labeling and base-training permission.

For capture training, set `data.dataset: frozen_capture`, `data.manifest` to the generated training
manifest, and `data.frozen_capture` to the frozen dataset. The training engine re-verifies both and
requires exact record digest/count, split, license, mapping, smoke status, data revision, and NTP
revision equality before constructing a loader. Non-smoke training configurations fail validation
without `data.frozen_capture`.

The deterministic end-to-end local smoke is:

```bash
uv run --extra cpu nana-tracking data capture-smoke \
  --work-dir runs/capture-smoke
```

It creates three synthetic identities with valid tiny PNG frames, persists and verifies twelve
chunks, reconciles a studio replica, regenerates three records, performs identity/device isolation,
builds the training manifest, re-verifies the frozen dataset, and is exercised by a one-step
FaceBasic training integration test. This is contract and recovery smoke evidence only. It does not
prove TrueDepth fidelity, Windows behavior, tracking quality, privacy approval, or production
throughput.

The checked-in macOS ARM64 filesystem smoke used 256 fsynced 64 KiB chunks (16 MiB total). Local
recording measured 0.270 ms p50 / 0.347 ms p95 and 221.6 MiB/s; verified streaming receive measured
0.219 ms p50 / 0.282 ms p95 and 228.7 MiB/s. Restart indexing, including durable recovery of a
synthetic 29-byte torn journal tail, took 1.75 ms; the pending scan took 0.012 ms. See
`artifacts/benchmarks/issue16-capture-store-macos-arm64-smoke.json`. These figures show the indexed
append-only implementation on this host only; they are not iPhone flash, Windows disk, LAN, or
production throughput acceptance.

## Privacy and admission

Before collection, follow `collection-protocol-v1.md`, `collection-action-script.md`, and
`governance-v1.md`. Record an explicit consent ID and approved license records for every session.
Raw face video, depth, geometry, identifiers, private metadata, frozen datasets, and derived labels
stay in access-controlled storage and are never committed. A production mapping and dataset require
human review, licensed collection, retention/deletion enforcement, identity-safe splits, and real
device quality evaluation.
