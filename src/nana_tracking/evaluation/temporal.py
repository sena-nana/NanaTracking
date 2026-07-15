"""Deterministic smoke benchmark for causal temporal behavior and overhead."""

import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from nana_tracking.reproducibility import git_state, sha256_file
from nana_tracking.runtime.temporal import (
    CausalTemporalRefiner,
    TemporalConfig,
    TemporalSample,
    TemporalState,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * quantile), len(ordered) - 1)]


def benchmark_temporal_refiner(
    output: Path,
    *,
    frames: int = 2_000,
    seed: int = 31,
) -> dict[str, object]:
    """Measure jitter, peak retention, bounded prediction, and per-frame CPU overhead."""

    if frames < 100:
        raise ValueError("temporal benchmark requires at least 100 frames")
    rng = np.random.default_rng(seed)
    dt_ns = rng.integers(12_000_000, 22_000_001, size=frames)
    timestamps = np.cumsum(dt_ns)
    raw_static = rng.normal(0.0, 0.025, size=frames)
    blink = raw_static.copy()
    blink[frames // 2] = -1.0
    mouth = rng.normal(0.1, 0.025, size=frames)
    config = TemporalConfig()
    refiner = CausalTemporalRefiner([7, 20], config=config)
    refined_blink: list[float] = []
    refined_mouth: list[float] = []
    latencies_ms: list[float] = []
    for index, timestamp in enumerate(timestamps):
        started = time.perf_counter_ns()
        samples = refiner.process(
            [
                TemporalSample(float(blink[index]), 0.9, TemporalState.OBSERVED, int(timestamp)),
                TemporalSample(float(mouth[index]), 0.9, TemporalState.OBSERVED, int(timestamp)),
            ],
            capture_timestamp_ns=int(timestamp),
            session_id=b"temporal-smoke",
            generation=0,
            camera_id="synthetic-camera",
            calibration_revision="synthetic-calibration",
        )
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        assert samples[0].value is not None and samples[1].value is not None
        refined_blink.append(samples[0].value)
        refined_mouth.append(samples[1].value)
    peak_index = frames // 2
    static_mask = np.ones(frames, dtype=np.bool_)
    static_mask[max(0, peak_index - 2) : peak_index + 3] = False
    raw_jitter = float(np.std(mouth[static_mask]))
    refined_jitter = float(np.std(np.asarray(refined_mouth)[static_mask]))
    peak_retention = abs(refined_blink[peak_index]) / abs(float(blink[peak_index]))
    prediction_refiner = CausalTemporalRefiner([20], config=config)
    prediction_samples = [
        (0.0, TemporalState.OBSERVED, 10_000_000),
        (0.2, TemporalState.OBSERVED, 20_000_000),
        (None, TemporalState.OCCLUDED, 30_000_000),
        (None, TemporalState.OCCLUDED, 40_000_000),
        (0.1, TemporalState.OBSERVED, 50_000_000),
    ]
    prediction_outputs = [
        prediction_refiner.process(
            [TemporalSample(value, 0.8, state, timestamp)],
            capture_timestamp_ns=timestamp,
            session_id=b"temporal-prediction-smoke",
            generation=0,
            camera_id="synthetic-camera",
            calibration_revision="synthetic-calibration",
        )[0]
        for value, state, timestamp in prediction_samples
    ]
    first_prediction, second_prediction = prediction_outputs[2:4]
    recovery = prediction_outputs[4]
    assert first_prediction.value is not None and second_prediction.value is not None
    assert recovery.value is not None
    if second_prediction.value <= first_prediction.value:
        raise AssertionError("repeated occlusion prediction stopped advancing")
    data_digest = hashlib.sha256()
    for values in (dt_ns, raw_static, blink, mouth):
        data_digest.update(values.tobytes())
    commit, dirty = git_state()
    lock_path = Path.cwd() / "uv.lock"
    report: dict[str, object] = {
        "schema_version": "causal-temporal-smoke-benchmark/1.1.0",
        "smoke_only": True,
        "seed": seed,
        "frames": frames,
        "resolved_config": asdict(config),
        "data_revision": "synthetic-temporal-numerical-v1",
        "data_digest_sha256": data_digest.hexdigest(),
        "checkpoint": None,
        "revisions": {
            "ntp": "ntp/1.0",
            "signal_registry": "ntp-signals/1.0.0",
            "normalization": "ntp-normalization/1.0.0",
            "calibration": "ntp-calibration/1.0.0",
            "features": "ntp-features/1.0.0",
        },
        "git": {
            "commit": commit,
            "dirty": dirty,
            "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        },
        "capture_dt_ms": {
            "minimum": float(dt_ns.min() / 1_000_000.0),
            "maximum": float(dt_ns.max() / 1_000_000.0),
            "mean": float(dt_ns.mean() / 1_000_000.0),
        },
        "static_jitter": {
            "raw_std": raw_jitter,
            "refined_std": refined_jitter,
            "reduction_percent": (1.0 - refined_jitter / raw_jitter) * 100.0,
        },
        "fast_peak_retention": peak_retention,
        "occlusion_prediction": {
            "prediction_horizon_ms": [
                first_prediction.prediction_horizon_ns / 1_000_000.0,
                second_prediction.prediction_horizon_ns / 1_000_000.0,
            ],
            "predicted_values": [first_prediction.value, second_prediction.value],
            "confidence": [first_prediction.confidence, second_prediction.confidence],
            "recovery_absolute_error": abs(recovery.value - 0.1),
        },
        "processing_ms": {
            "p50": statistics.median(latencies_ms),
            "p95": _percentile(latencies_ms, 0.95),
            "p99": _percentile(latencies_ms, 0.99),
            "mean": statistics.fmean(latencies_ms),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "byte_order": sys.byteorder,
        },
        "note": "Synthetic irregular-dt smoke only; not real tracking quality acceptance.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
