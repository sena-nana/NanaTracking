# NanaTracking

NanaTracking is the reproducible PyTorch training and ONNX export workspace for the
NanaTracking Protocol ecosystem. The checked-in model and dataset are synthetic smoke fixtures;
they do not implement or validate a face-tracking algorithm.

## Setup

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required.

```bash
# CPU on Linux/Windows and CPU/MPS development on macOS
uv sync --locked --extra cpu --all-groups

# CUDA 13.0 on a compatible Windows/Linux host
uv sync --locked --extra cu130 --all-groups
```

The `cpu` and `cu130` extras are intentionally mutually exclusive.

## Workflows

```bash
uv run --extra cpu nana-tracking doctor
uv run --extra cpu nana-tracking data validate examples/manifests/synthetic-v1.json
uv run --extra cpu nana-tracking data materialize-labels \
  examples/manifests/synthetic-v1.json --output artifacts/data/synthetic-labels.jsonl
uv run --extra cpu nana-tracking evaluation validate-standard \
  configs/evaluation/ntp-v1-standard.json
uv run --extra cpu nana-tracking train --config configs/smoke.yaml
uv run --extra cpu nana-tracking smoke --work-dir runs/smoke
uv run --extra cpu nana-tracking train --config configs/face-basic-smoke.yaml
uv run --extra cpu nana-tracking train --config configs/face-spatial-smoke.yaml
uv run --extra cpu nana-tracking benchmark-python
cargo run -p ntp-conformance -- stream.ntp --output json
```

The interpreter benchmark runs `InterpreterPoolExecutor` inside an isolated broker process.
PyTorch 2.11 autograd never shares a process that has created subinterpreters. Benchmark reports
are written under `artifacts/benchmarks/`; adopt an executor only when target-workload evidence
shows a throughput benefit without unacceptable startup or memory cost.

The [FaceBasic v1 baseline](docs/model/face-basic-v1.md) documents the shared-encoder multi-task
model, manifest loader, Level A calibration, latest-frame-only NTP producer, ONNX package, target
hardware benchmark, and failure-sample workflow. Its checked-in configuration is smoke-only and
cannot serve as real tracking-quality or RTX 4060 acceptance evidence.

Run the quality gates before handing off a change:

```bash
uv lock --check
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
uv run --extra cpu pyright
uv run --extra cpu pytest --cov=nana_tracking --cov-report=term-missing
uv build
```

## Artifact boundary

Real datasets, checkpoints, caches, run directories, and exported models are ignored by Git.
Commit only reviewed schemas, manifests, revisions, digests, and deliberately small fixtures.
The ONNX model package remains separate from PyTorch checkpoints and from user personalization
profiles.

## Protocol specifications

- [NTP v1 Signal Registry](docs/protocol/ntp-v1-signal-registry.md)
- [NTP v1 freeze review checklist](docs/protocol/ntp-v1-freeze-checklist.md)
- [NTP v1 canonical codec and Rust/C contract](docs/protocol/ntp-v1-codec.md)
- [NTP v1 semantic derivation and rig binding reference](docs/semantics/ntp-v1-reference.md)
- [NTP v1 conformance and compatibility matrix](docs/conformance/ntp-v1-compatibility-matrix.md)
- [FaceSpatial v1 model and producer](docs/model/face-spatial-v1.md)
- [Spatial same-capture fusion contract](docs/contracts/spatial-fusion-v1.md)
- [ADR 0001: Vendor parameters are adapters, not NTP](docs/adr/0001-vendor-parameters-are-adapters.md)
- [ADR 0002: PyTorch authority and portable runtime boundary](docs/adr/0002-pytorch-onnx-runtime-boundary.md)

## Training data and evaluation

- [Dataset schema, synchronization, labeling, and quality gates](docs/data/dataset-contract-v1.md)
- [Guided Basic, Spatial, and Full collection protocol](docs/data/collection-protocol-v1.md)
- [Shared evaluation standard and report contract](docs/data/evaluation-standard-v1.md)
- [License, privacy, and failure-sample feedback](docs/data/governance-v1.md)

The framework-neutral Rust implementations live in `crates/nana-tracking-protocol` and
`crates/nana-tracking-semantics`. Protocol codec/capability code and deterministic semantic/model
binding code remain independent from the Python/PyTorch training workspace.

`crates/ntp-conformance` derives certification from complete capability sets and validates binary
or diagnostic JSONL descriptor/result streams. Its checked-in vectors are synthetic contract-only
smoke evidence; they do not claim tracking-model quality or production latency.
