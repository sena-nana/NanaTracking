"""Target-hardware ONNX runtime benchmark with machine-readable provenance."""

import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.export import verify_model_package
from nana_tracking.runtime import (
    FaceBasicProducer,
    FaceSpatialProducer,
    OrtFaceBasicBackend,
    OrtFaceSpatialBackend,
    OrtFullSetBackend,
)


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
