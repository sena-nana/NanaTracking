# NanaTracking agent instructions

## Architecture boundaries

- Treat PyTorch as the authoritative implementation for model definitions, losses, training,
  checkpoints, and evaluation. Treat ONNX as a deployment artifact.
- Do not expose `torch.Tensor`, ONNX Runtime values, TensorRT bindings, Core ML types, Burn
  tensors, or other backend types through NTP, protocol, FFI, or consumer-facing contracts.
- Keep the base model, Level A calibration, learned ONNX adapters, and optional online adapters
  independently versioned and resettable.
- Do not add Candle to the production path or make Burn-to-ONNX a build requirement.
- Mark every synthetic model, dataset, benchmark, and report as smoke-only. Never claim that a
  synthetic test proves FaceBasic quality, target latency, or production readiness.

## Environment and commands

- Use CPython 3.14 only and manage environments and dependencies with uv.
- Prefix repository shell commands with `rtk` when it is available in the active environment.
- Install exactly one accelerator extra:
  `uv sync --locked --extra cpu --all-groups` or
  `uv sync --locked --extra cu130 --all-groups`.
- Before handing off code, run `uv lock --check`, Ruff check and format check, Pyright, pytest with
  coverage, `uv build`, skill validation, and the synthetic smoke pipeline.
- CUDA/RTX acceptance must run on a real compatible GPU. Do not infer CUDA performance from CPU,
  MPS, or hosted CI.

## Reproducibility and data

- Record the resolved configuration, fixed seed, data revision and digest, NTP and Signal Registry
  revisions, Git state, lockfile digest, metrics, and checkpoint metadata for every run.
- Split data by identity; never allow the same identity in train, validation, and test splits.
- Confirm licenses permit collection, distillation, pseudo-labeling, and commercial training before
  admitting third-party teacher data.
- Never commit raw user recordings, secrets, private metadata, large datasets, checkpoints, caches,
  or exported model packages. Commit reviewed schemas, manifests, digests, and tiny fixtures only.

## Python 3.14 concurrency

- Use `InterpreterPoolExecutor` only through the isolated broker process and only for pickle-safe
  pure-Python or byte-oriented work. PyTorch 2.11 autograd must never share a process that has
  created subinterpreters. Do not pass mutable objects, NumPy arrays, or PyTorch tensors between
  interpreters.
- Bound submitted work with `buffersize`; no unbounded prefetch queues.
- Keep free-threaded builds and the experimental JIT opt-in until target-workload benchmarks prove
  a benefit and every C extension is compatible.
- Preserve benchmark JSON alongside implementation evidence when changing the default executor.

## Project skills

- Use `$manage-training-data` for manifests, collection, split, provenance, privacy, and quality.
- Use `$align-ntp-contracts` for labels, outputs, revisions, normalization, and framework boundaries.
- Use `$train-pytorch-models` for models, losses, AMP, determinism, checkpoints, and resume.
- Use `$evaluate-tracking-models` for quality, confidence, temporal, failure, and performance reports.
- Use `$release-onnx-models` for export, parity, test vectors, metadata, digest, and package work.
- Use `$build-personalization-adapters` for calibration, residual adapters, compatibility, reset,
  rollback, and user isolation.
