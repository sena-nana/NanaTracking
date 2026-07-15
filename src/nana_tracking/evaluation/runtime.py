"""Target-hardware ONNX runtime benchmark with machine-readable provenance."""

import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
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
