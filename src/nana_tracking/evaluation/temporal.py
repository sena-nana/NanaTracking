"""Deterministic smoke benchmark for causal temporal behavior and overhead."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from nana_tracking.runtime.temporal import (
    CausalTemporalRefiner,
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
    refiner = CausalTemporalRefiner([7, 20])
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
    report: dict[str, object] = {
        "schema_version": "causal-temporal-smoke-benchmark/1.0.0",
        "smoke_only": True,
        "seed": seed,
        "frames": frames,
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
        },
        "note": "Synthetic irregular-dt smoke only; not real tracking quality acceptance.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
