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
uv run --extra cpu nana-tracking train --config configs/smoke.yaml
uv run --extra cpu nana-tracking smoke --work-dir runs/smoke
uv run --extra cpu nana-tracking benchmark-python
```

The interpreter benchmark runs `InterpreterPoolExecutor` inside an isolated broker process.
PyTorch 2.11 autograd never shares a process that has created subinterpreters. Benchmark reports
are written under `artifacts/benchmarks/`; adopt an executor only when target-workload evidence
shows a throughput benefit without unacceptable startup or memory cost.

Run the quality gates before opening a pull request:

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
- [ADR 0001: Vendor parameters are adapters, not NTP](docs/adr/0001-vendor-parameters-are-adapters.md)
