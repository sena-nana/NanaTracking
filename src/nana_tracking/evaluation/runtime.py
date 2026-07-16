"""Target-hardware ONNX runtime benchmark with machine-readable provenance."""

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import deque
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.export import verify_model_package
from nana_tracking.reproducibility import git_state, sha256_file
from nana_tracking.runtime import (
    FaceBasicProducer,
    FaceBox,
    FaceSpatialProducer,
    OrtFaceBasicBackend,
    OrtFaceSpatialBackend,
    OrtFullSetBackend,
    RgbRoiWorkspace,
)
from nana_tracking.runtime.face_basic import prepare_rgb_roi


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def _peak_rss_native_units() -> int | None:
    if sys.platform == "win32":
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _nvidia_telemetry() -> dict[str, object] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired:
        return None
    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None
    first = lines[0].split(", ")
    if len(first) != 5:
        return None
    return {
        "name": first[0],
        "driver_version": first[1],
        "utilization_percent": float(first[2]),
        "vram_used_mib": float(first[3]),
        "vram_total_mib": float(first[4]),
    }


def _current_process_resources(elapsed_seconds: float) -> dict[str, object]:
    rss_bytes: int | None = None
    thread_count: int | None = None
    if sys.platform == "linux":
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
            rss_bytes = resident_pages * page_size
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("Threads:"):
                    thread_count = int(line.split()[1])
                    break
        except OSError, ValueError, IndexError:
            pass
    elif sys.platform == "darwin":
        try:
            rss = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            threads = subprocess.run(
                ["ps", "-M", "-p", str(os.getpid())],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            rss_bytes = int(rss.stdout.strip()) * 1024
            thread_count = max(0, len(threads.stdout.splitlines()) - 1)
        except OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError:
            pass
    return {
        "elapsed_seconds": elapsed_seconds,
        "rss_bytes": rss_bytes,
        "thread_count": thread_count,
    }


class _BoundedLatencySamples:
    def __init__(self, *, capacity: int, edge_capacity: int, seed: int) -> None:
        if capacity < 1 or edge_capacity < 1:
            raise ValueError("latency sample capacities must be positive")
        self._capacity = capacity
        self._edge_capacity = edge_capacity
        self._random = random.Random(seed)
        self._reservoir: list[tuple[float, float]] = []
        self._first: list[tuple[float, float]] = []
        self._last: deque[tuple[float, float]] = deque(maxlen=edge_capacity)
        self._seen = 0

    def add(self, capture_to_result_ms: float, result_age_ms: float) -> None:
        sample = (capture_to_result_ms, result_age_ms)
        self._seen += 1
        if len(self._first) < self._edge_capacity:
            self._first.append(sample)
        self._last.append(sample)
        if len(self._reservoir) < self._capacity:
            self._reservoir.append(sample)
            return
        replacement = self._random.randrange(self._seen)
        if replacement < self._capacity:
            self._reservoir[replacement] = sample

    @property
    def seen(self) -> int:
        return self._seen

    @property
    def retained(self) -> int:
        return len(self._reservoir) + len(self._first) + len(self._last)

    @staticmethod
    def _summary(samples: list[tuple[float, float]], index: int) -> dict[str, float]:
        values = [sample[index] for sample in samples]
        return {
            "p50": statistics.median(values),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "mean": statistics.fmean(values),
        }

    def summary(self, index: int) -> dict[str, object]:
        if not self._reservoir:
            raise ValueError("stability benchmark produced no latency samples")
        return {
            "all_reservoir": self._summary(self._reservoir, index),
            "first_window": self._summary(self._first, index),
            "last_window": self._summary(list(self._last), index),
        }


def benchmark_rgb_roi_preprocessor(
    output: Path,
    *,
    input_width: int = 1280,
    input_height: int = 720,
    roi_side: int = 640,
    output_sizes: tuple[int, ...] = (64, 96, 128),
    roi_positions: int = 32,
    frames_per_roi: int = 5,
    warmup: int = 100,
    iterations: int = 2_000,
) -> dict[str, object]:
    """Benchmark moving-ROI preprocessing with persistent source/output/workspace buffers."""

    if input_width < 1 or input_height < 1:
        raise ValueError("benchmark input dimensions must be positive")
    if roi_side < 1 or roi_side > min(input_width, input_height):
        raise ValueError("benchmark ROI must fit inside the input frame")
    if not output_sizes or any(size < 1 for size in output_sizes):
        raise ValueError("benchmark output sizes must be positive")
    if len(set(output_sizes)) != len(output_sizes):
        raise ValueError("benchmark output sizes must be unique")
    if roi_positions < 1 or frames_per_roi < 1 or warmup < 1 or iterations < 1:
        raise ValueError("benchmark positions, cadence, warmup, and iterations must be positive")

    frame = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    horizontal_range = input_width - roi_side
    vertical_range = input_height - roi_side
    divisor = max(roi_positions - 1, 1)
    boxes = tuple(
        FaceBox(
            horizontal_range * index // divisor,
            vertical_range * index // divisor,
            horizontal_range * index // divisor + roi_side,
            vertical_range * index // divisor + roi_side,
        )
        for index in range(roi_positions)
    )
    results: dict[str, object] = {}
    for size in output_sizes:
        workspace = RgbRoiWorkspace(size, size)
        model_input = np.empty((1, 3, size, size), dtype=np.float32)
        for index in range(warmup):
            prepare_rgb_roi(
                frame,
                boxes[(index // frames_per_roi) % len(boxes)],
                model_input,
                workspace=workspace,
            )
        latencies_ns: list[float] = []
        for index in range(iterations):
            started = time.perf_counter_ns()
            prepare_rgb_roi(
                frame,
                boxes[(index // frames_per_roi) % len(boxes)],
                model_input,
                workspace=workspace,
            )
            latencies_ns.append(float(time.perf_counter_ns() - started))

        traced_iterations = min(iterations, 100)
        tracemalloc.start()
        tracemalloc.reset_peak()
        for index in range(traced_iterations):
            prepare_rgb_roi(
                frame,
                boxes[(index // frames_per_roi) % len(boxes)],
                model_input,
                workspace=workspace,
            )
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mean_ns = statistics.fmean(latencies_ns)
        results[str(size)] = {
            "latency_ns": {
                "p50": statistics.median(latencies_ns),
                "p95": _percentile(latencies_ns, 0.95),
                "p99": _percentile(latencies_ns, 0.99),
                "mean": mean_ns,
            },
            "frames_per_second_at_mean": 1_000_000_000.0 / mean_ns,
            "persistent_workspace_bytes": workspace.workspace_bytes,
            "steady_tracemalloc_peak_bytes": traced_peak,
            "traced_iterations": traced_iterations,
        }

    commit, dirty = git_state()
    lock_path = Path("uv.lock")
    report: dict[str, object] = {
        "schema_version": "nana-rgb-roi-preprocess-benchmark/1.0.0",
        "smoke_only": True,
        "implementation": "numpy-preallocated-row-workspace-nearest-v1",
        "input": {
            "shape_hwc": [input_height, input_width, 3],
            "dtype": "uint8",
            "source_buffer_reused": True,
        },
        "roi_strategy": {
            "shape": "square",
            "side_pixels": roi_side,
            "moving_positions": roi_positions,
            "frames_per_roi": frames_per_roi,
            "indices_recomputed_only_when_roi_changes": True,
        },
        "output_sizes": list(output_sizes),
        "warmup": warmup,
        "iterations": iterations,
        "results": results,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "provenance": {
            "git_commit": commit,
            "working_tree_dirty_during_measurement": dirty,
            "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        },
        "note": (
            "Synthetic preprocessing smoke only. Persistent workspace is bounded by model input "
            "size; traced peak includes Python and NumPy call overhead and is not GPU evidence."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_face_benchmark_context(
    package_dir: Path,
    *,
    providers: list[str],
    tensorrt_fp16: bool,
) -> tuple[
    ModelPackageMetadata,
    OrtFaceBasicBackend | OrtFaceSpatialBackend,
    FaceBasicProducer | FaceSpatialProducer,
    str,
    np.ndarray,
]:
    verify_model_package(package_dir)
    metadata = ModelPackageMetadata.model_validate_json(
        (package_dir / "runtime-metadata.json").read_text(encoding="utf-8")
    )
    available = cast(list[str], ort.get_available_providers())
    unavailable = set(providers).difference(available)
    if unavailable:
        raise RuntimeError(f"requested ONNX Runtime providers are unavailable: {unavailable}")
    if metadata.supported_signals == list(range(1, 37)):
        backend = OrtFaceBasicBackend(
            package_dir,
            providers=providers,
            tensorrt_fp16=tensorrt_fp16,
        )
        producer = FaceBasicProducer(backend)
        profile_name = "face-basic"
    elif metadata.supported_signals == list(range(1, 42)):
        backend = OrtFaceSpatialBackend(
            package_dir,
            providers=providers,
            tensorrt_fp16=tensorrt_fp16,
        )
        producer = FaceSpatialProducer(backend)
        profile_name = "face-spatial"
    else:
        raise ValueError("benchmark requires a complete Basic or Spatial face package")
    with np.load(package_dir / "test-vectors" / "input.npz") as vectors:
        image = vectors["image"]
    frame = np.rint(np.transpose(image[0], (1, 2, 0)) * 255.0).astype(np.uint8)
    return metadata, backend, producer, profile_name, frame


def benchmark_face_package(
    package_dir: Path,
    output: Path,
    *,
    providers: list[str],
    warmup: int = 20,
    iterations: int = 200,
    tensorrt_fp16: bool = False,
) -> dict[str, object]:
    """Benchmark the packaged fixed ROI on the active provider and hardware."""

    metadata, backend, producer, profile_name, frame = _load_face_benchmark_context(
        package_dir,
        providers=providers,
        tensorrt_fp16=tensorrt_fp16,
    )
    sequence = 0
    for _ in range(warmup):
        sequence += 1
        producer.produce(
            frame,
            roi=None,
            capture_timestamp_ns=time.monotonic_ns(),
            sequence=sequence,
        )

    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    capture_to_result: list[float] = []
    result_age: list[float] = []
    for _ in range(iterations):
        sequence += 1
        captured = time.monotonic_ns()
        event = producer.produce(
            frame,
            roi=None,
            capture_timestamp_ns=captured,
            sequence=sequence,
        )
        consumed = time.monotonic_ns()
        value = cast(dict[str, object], event["value"])
        produced = cast(int, value["produced_timestamp_ns"])
        capture_to_result.append((produced - captured) / 1_000_000.0)
        result_age.append((consumed - captured) / 1_000_000.0)
    wall_seconds = (time.perf_counter_ns() - wall_start) / 1_000_000_000.0
    cpu_seconds = (time.process_time_ns() - cpu_start) / 1_000_000_000.0
    gpu = _nvidia_telemetry()
    report: dict[str, object] = {
        "schema_version": f"{profile_name}-runtime-benchmark/1.0.0",
        "smoke_only": metadata.smoke_only,
        "model_digest": metadata.model_digest,
        "source_checkpoint_digest": metadata.source_checkpoint_digest,
        "ntp_schema_revision": metadata.ntp_schema_revision,
        "signal_registry_revision": metadata.signal_registry_revision,
        "normalization_revision": metadata.normalization_revision,
        "calibration_revision": metadata.calibration_revision,
        "feature_revision": metadata.feature_revision,
        "geometry_topology_revision": metadata.geometry_topology_revision,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "requested_providers": providers,
            "active_providers": backend.active_providers,
            "provider_evidence": (
                "ORT session provider registration; optimized provider node assignment requires "
                "its separate profile and fixed-vector parity gate"
            ),
            "precision_support": metadata.precision_support,
            "tensorrt_fp16": tensorrt_fp16,
            "input_shape": metadata.input_shape,
            "iterations": iterations,
            "warmup": warmup,
        },
        "capture_to_result_ms": {
            "p50": statistics.median(capture_to_result),
            "p95": _percentile(capture_to_result, 0.95),
            "p99": _percentile(capture_to_result, 0.99),
            "mean": statistics.fmean(capture_to_result),
        },
        "result_age_at_consume_ms": {
            "p50": statistics.median(result_age),
            "p95": _percentile(result_age, 0.95),
            "p99": _percentile(result_age, 0.99),
            "mean": statistics.fmean(result_age),
        },
        "resources": {
            "cpu_core_equivalents": cpu_seconds / max(wall_seconds, 1e-9),
            "process_peak_rss_native_units": _peak_rss_native_units(),
            "nvidia_smi_snapshot": gpu,
            "note": (
                "NVIDIA values are an end-of-run device snapshot, not inferred from CPU results."
                if gpu is not None
                else "NVIDIA telemetry unavailable; GPU and VRAM are not inferred."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def benchmark_face_package_stability(
    package_dir: Path,
    output: Path,
    *,
    providers: list[str],
    duration_seconds: float = 1_800.0,
    target_fps: float = 60.0,
    resource_sample_interval_seconds: float = 60.0,
    warmup: int = 100,
    reservoir_capacity: int = 65_536,
    edge_window_capacity: int = 4_096,
    seed: int = 47,
    maximum_result_age_p95_drift_ms: float = 2.0,
    maximum_rss_growth_bytes: int = 32 * 1024 * 1024,
    maximum_thread_growth: int = 2,
    tensorrt_fp16: bool = False,
) -> dict[str, object]:
    """Run a paced, bounded-memory stability benchmark on an actual face package backend."""

    if not 0.0 < duration_seconds <= 7_200.0:
        raise ValueError("stability duration must be in (0, 7200] seconds")
    if not 0.0 < target_fps <= 240.0:
        raise ValueError("target FPS must be in (0, 240]")
    if resource_sample_interval_seconds <= 0.0 or warmup < 1:
        raise ValueError("resource interval and warmup must be positive")
    if maximum_result_age_p95_drift_ms < 0.0:
        raise ValueError("result-age drift threshold must be non-negative")
    if maximum_rss_growth_bytes < 0 or maximum_thread_growth < 0:
        raise ValueError("resource growth thresholds must be non-negative")
    metadata, backend, producer, profile_name, frame = _load_face_benchmark_context(
        package_dir,
        providers=providers,
        tensorrt_fp16=tensorrt_fp16,
    )
    sequence = 0
    for _ in range(warmup):
        sequence += 1
        producer.produce(
            frame,
            roi=None,
            capture_timestamp_ns=time.monotonic_ns(),
            sequence=sequence,
        )

    period_ns = max(1, round(1_000_000_000 / target_fps))
    resource_interval_ns = max(1, round(resource_sample_interval_seconds * 1_000_000_000))
    samples = _BoundedLatencySamples(
        capacity=reservoir_capacity,
        edge_capacity=edge_window_capacity,
        seed=seed,
    )
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    deadline = wall_started
    end = wall_started + round(duration_seconds * 1_000_000_000)
    next_resource_sample = wall_started + resource_interval_ns
    resource_samples = [_current_process_resources(0.0)]
    skipped_capture_periods = 0
    while True:
        now = time.perf_counter_ns()
        if now < deadline:
            time.sleep((deadline - now) / 1_000_000_000.0)
            now = time.perf_counter_ns()
        if now >= end and samples.seen:
            break
        if now - deadline >= period_ns:
            skipped = (now - deadline) // period_ns
            skipped_capture_periods += skipped
            deadline += skipped * period_ns
        sequence += 1
        captured = time.monotonic_ns()
        event = producer.produce(
            frame,
            roi=None,
            capture_timestamp_ns=captured,
            sequence=sequence,
        )
        consumed = time.monotonic_ns()
        value = cast(dict[str, object], event["value"])
        produced = cast(int, value["produced_timestamp_ns"])
        samples.add(
            (produced - captured) / 1_000_000.0,
            (consumed - captured) / 1_000_000.0,
        )
        deadline += period_ns
        if now >= next_resource_sample:
            elapsed = (now - wall_started) / 1_000_000_000.0
            resource_samples.append(_current_process_resources(elapsed))
            next_resource_sample += resource_interval_ns

    wall_seconds = (time.perf_counter_ns() - wall_started) / 1_000_000_000.0
    cpu_seconds = (time.process_time_ns() - cpu_started) / 1_000_000_000.0
    resource_samples.append(_current_process_resources(wall_seconds))
    capture_to_result = samples.summary(0)
    result_age = samples.summary(1)
    first_age = cast(dict[str, float], result_age["first_window"])
    last_age = cast(dict[str, float], result_age["last_window"])
    result_age_p95_drift_ms = last_age["p95"] - first_age["p95"]
    rss_values = [
        cast(int, sample["rss_bytes"])
        for sample in resource_samples
        if sample["rss_bytes"] is not None
    ]
    thread_values = [
        cast(int, sample["thread_count"])
        for sample in resource_samples
        if sample["thread_count"] is not None
    ]
    rss_growth_bytes = rss_values[-1] - rss_values[0] if len(rss_values) >= 2 else None
    thread_growth = thread_values[-1] - thread_values[0] if len(thread_values) >= 2 else None
    delivered_fps = samples.seen / max(wall_seconds, 1e-9)
    gates = {
        "duration_reached": wall_seconds >= duration_seconds * 0.99,
        "target_cadence_reached": delivered_fps >= target_fps * 0.95,
        "result_age_p95_drift_within_limit": (
            result_age_p95_drift_ms <= maximum_result_age_p95_drift_ms
        ),
        "rss_growth_within_limit": (
            rss_growth_bytes is not None and rss_growth_bytes <= maximum_rss_growth_bytes
        ),
        "thread_growth_within_limit": (
            thread_growth is not None and thread_growth <= maximum_thread_growth
        ),
    }
    git_commit, git_dirty = git_state()
    report: dict[str, object] = {
        "schema_version": f"{profile_name}-runtime-stability/1.0.0",
        "smoke_only": metadata.smoke_only,
        "model_digest": metadata.model_digest,
        "source_checkpoint_digest": metadata.source_checkpoint_digest,
        "ntp_schema_revision": metadata.ntp_schema_revision,
        "signal_registry_revision": metadata.signal_registry_revision,
        "normalization_revision": metadata.normalization_revision,
        "calibration_revision": metadata.calibration_revision,
        "feature_revision": metadata.feature_revision,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "runtime": {
            "python": platform.python_version(),
            "onnxruntime_version": ort.__version__,
            "requested_providers": providers,
            "active_providers": backend.active_providers,
            "provider_evidence": (
                "ORT session provider registration; optimized provider node assignment requires "
                "its separate profile and fixed-vector parity gate"
            ),
            "precision_support": metadata.precision_support,
            "tensorrt_fp16": tensorrt_fp16,
            "input_shape": metadata.input_shape,
            "target_fps": target_fps,
            "delivered_fps": delivered_fps,
            "completed_frames": samples.seen,
            "skipped_capture_periods": skipped_capture_periods,
            "warmup": warmup,
            "duration_seconds_requested": duration_seconds,
            "duration_seconds_measured": wall_seconds,
            "scheduling": "paced latest-capture; overdue capture periods are skipped, never queued",
        },
        "bounded_sampling": {
            "algorithm": "deterministic Algorithm R reservoir plus fixed first/last windows",
            "seed": seed,
            "observed_samples": samples.seen,
            "retained_samples_including_windows": samples.retained,
            "reservoir_capacity": reservoir_capacity,
            "edge_window_capacity": edge_window_capacity,
        },
        "capture_to_result_ms": capture_to_result,
        "result_age_at_consume_ms": result_age,
        "resources": {
            "cpu_core_equivalents": cpu_seconds / max(wall_seconds, 1e-9),
            "samples": resource_samples,
            "rss_growth_bytes": rss_growth_bytes,
            "peak_sampled_rss_bytes": max(rss_values) if rss_values else None,
            "thread_growth": thread_growth,
            "nvidia_smi_snapshot": _nvidia_telemetry(),
        },
        "stability": {
            "passed": all(gates.values()),
            "gates": gates,
            "result_age_p95_drift_ms": result_age_p95_drift_ms,
            "maximum_result_age_p95_drift_ms": maximum_result_age_p95_drift_ms,
            "maximum_rss_growth_bytes": maximum_rss_growth_bytes,
            "maximum_thread_growth": maximum_thread_growth,
        },
        "provenance": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "uv_lock_sha256": sha256_file(Path("uv.lock")),
        },
        "limitations": (
            "Fixed package test-vector RGB smoke only. This proves runtime scheduling and resource "
            "stability on the CPU-only backend/hardware used by this run. Provider registration "
            "alone does not prove optimized node assignment. This does not prove camera I/O, "
            "tracking quality, or production readiness."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def benchmark_face_basic_package(
    package_dir: Path,
    output: Path,
    *,
    providers: list[str],
    warmup: int = 20,
    iterations: int = 200,
    tensorrt_fp16: bool = False,
) -> dict[str, object]:
    report = benchmark_face_package(
        package_dir,
        output,
        providers=providers,
        warmup=warmup,
        iterations=iterations,
        tensorrt_fp16=tensorrt_fp16,
    )
    if report["schema_version"] != "face-basic-runtime-benchmark/1.0.0":
        raise ValueError("package is not FaceBasic")
    return report


def benchmark_face_spatial_package(
    package_dir: Path,
    output: Path,
    *,
    providers: list[str],
    warmup: int = 20,
    iterations: int = 200,
    tensorrt_fp16: bool = False,
) -> dict[str, object]:
    report = benchmark_face_package(
        package_dir,
        output,
        providers=providers,
        warmup=warmup,
        iterations=iterations,
        tensorrt_fp16=tensorrt_fp16,
    )
    if report["schema_version"] != "face-spatial-runtime-benchmark/1.0.0":
        raise ValueError("package is not FaceSpatial")
    return report


def benchmark_full_set_package(
    package_dir: Path,
    output: Path,
    *,
    providers: list[str],
    warmup: int = 20,
    iterations: int = 200,
    tensorrt_fp16: bool = False,
) -> dict[str, object]:
    """Benchmark the low-cadence upper-body ONNX package on explicit hardware."""

    verify_model_package(package_dir)
    metadata = ModelPackageMetadata.model_validate_json(
        (package_dir / "runtime-metadata.json").read_text(encoding="utf-8")
    )
    backend = OrtFullSetBackend(package_dir, providers=providers, tensorrt_fp16=tensorrt_fp16)
    with np.load(package_dir / "test-vectors" / "input.npz") as vectors:
        image = vectors["image"]
    for _ in range(warmup):
        backend.infer(image)
    latencies: list[float] = []
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        backend.infer(image)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    wall_seconds = (time.perf_counter_ns() - wall_start) / 1_000_000_000.0
    cpu_seconds = (time.process_time_ns() - cpu_start) / 1_000_000_000.0
    gpu = _nvidia_telemetry()
    report: dict[str, object] = {
        "schema_version": "full-set-upper-body-runtime-benchmark/1.0.0",
        "smoke_only": metadata.smoke_only,
        "model_digest": metadata.model_digest,
        "source_checkpoint_digest": metadata.source_checkpoint_digest,
        "ntp_schema_revision": metadata.ntp_schema_revision,
        "signal_registry_revision": metadata.signal_registry_revision,
        "normalization_revision": metadata.normalization_revision,
        "calibration_revision": metadata.calibration_revision,
        "feature_revision": metadata.feature_revision,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "requested_providers": providers,
            "active_providers": backend.active_providers,
            "precision_support": metadata.precision_support,
            "tensorrt_fp16": tensorrt_fp16,
            "input_shape": metadata.input_shape,
            "iterations": iterations,
            "warmup": warmup,
            "scheduling": "latest-frame-only; intended lower cadence than face inference",
        },
        "upper_body_inference_ms": {
            "p50": statistics.median(latencies),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "mean": statistics.fmean(latencies),
        },
        "resources": {
            "cpu_core_equivalents": cpu_seconds / max(wall_seconds, 1e-9),
            "process_peak_rss_native_units": _peak_rss_native_units(),
            "nvidia_smi_snapshot": gpu,
            "note": (
                "NVIDIA values are an end-of-run snapshot, not inferred from CPU results."
                if gpu is not None
                else "NVIDIA telemetry unavailable; GPU and VRAM are not inferred."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
