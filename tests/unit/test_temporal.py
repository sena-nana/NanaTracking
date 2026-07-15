from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from nana_tracking.evaluation import ConfidenceCalibration, fit_confidence_calibration
from nana_tracking.evaluation.temporal import benchmark_temporal_refiner
from nana_tracking.runtime import (
    CausalTemporalRefiner,
    TemporalConfig,
    TemporalSample,
    TemporalState,
)


def _sample(
    value: float | None,
    timestamp_ns: int,
    state: TemporalState = TemporalState.OBSERVED,
    confidence: float = 0.9,
) -> TemporalSample:
    return TemporalSample(value, confidence, state, timestamp_ns)


def _process(
    refiner: CausalTemporalRefiner,
    sample: TemporalSample,
    timestamp_ns: int,
    *,
    generation: int = 0,
    camera_id: str = "camera-a",
) -> TemporalSample:
    return refiner.process(
        [sample],
        capture_timestamp_ns=timestamp_ns,
        session_id=b"session-a",
        generation=generation,
        camera_id=camera_id,
        calibration_revision="calibration-a",
    )[0]


def test_causal_refiner_reduces_static_jitter_without_erasing_blink_peak() -> None:
    refiner = CausalTemporalRefiner([7])
    raw = [0.02, -0.02, 0.03, -0.01, 0.01]
    refined = [
        _process(refiner, _sample(value, index * 16_000_000), index * 16_000_000).value
        for index, value in enumerate(raw, start=1)
    ]
    assert all(value is not None for value in refined)
    refined_values = [float(value) for value in refined if value is not None]
    assert np.std(refined_values) < np.std(raw)
    peak = _process(refiner, _sample(-0.9, 96_000_000), 96_000_000)
    assert peak.value == pytest.approx(-0.9)


def test_prediction_uses_capture_dt_decays_and_stops_at_bound() -> None:
    refiner = CausalTemporalRefiner([20], config=TemporalConfig(maximum_prediction_ns=60_000_000))
    _process(refiner, _sample(0.0, 10_000_000), 10_000_000)
    last_observed = _process(refiner, _sample(0.2, 20_000_000), 20_000_000)
    predicted = _process(
        refiner,
        _sample(None, 50_000_000, TemporalState.OCCLUDED, 0.8),
        50_000_000,
    )
    assert predicted.state == TemporalState.PREDICTED
    assert predicted.prediction_horizon_ns == 30_000_000
    assert predicted.sample_capture_timestamp_ns == 20_000_000
    assert predicted.value is not None and last_observed.value is not None
    assert last_observed.value < predicted.value <= last_observed.value + 0.35
    assert predicted.confidence < 0.8
    lost = _process(
        refiner,
        _sample(None, 100_000_000, TemporalState.OUT_OF_FRAME, 0.7),
        100_000_000,
    )
    assert lost.state == TemporalState.TRACKING_LOST
    assert lost.value is None


def test_repeated_prediction_uses_observations_and_recovery_does_not_lag() -> None:
    refiner = CausalTemporalRefiner([20])
    _process(refiner, _sample(0.0, 10_000_000), 10_000_000)
    _process(refiner, _sample(0.2, 20_000_000), 20_000_000)
    first = _process(
        refiner,
        _sample(None, 30_000_000, TemporalState.OCCLUDED, 0.8),
        30_000_000,
    )
    second = _process(
        refiner,
        _sample(None, 40_000_000, TemporalState.OCCLUDED, 0.8),
        40_000_000,
    )
    assert first.value is not None and second.value is not None
    assert second.value > first.value
    assert second.prediction_horizon_ns == 20_000_000
    assert second.confidence < first.confidence

    recovered = _process(refiner, _sample(0.1, 50_000_000), 50_000_000)
    assert recovered.state == TemporalState.OBSERVED
    assert recovered.value == pytest.approx(0.1)


def test_prediction_ignores_reused_fused_samples_with_the_same_source_time() -> None:
    refiner = CausalTemporalRefiner([63])
    _process(refiner, _sample(0.0, 10_000_000), 10_000_000)
    _process(refiner, _sample(0.2, 20_000_000), 20_000_000)
    _process(
        refiner,
        _sample(0.2, 20_000_000, TemporalState.FUSED),
        30_000_000,
    )
    _process(
        refiner,
        _sample(0.2, 20_000_000, TemporalState.FUSED),
        40_000_000,
    )
    predicted = _process(
        refiner,
        _sample(None, 50_000_000, TemporalState.OCCLUDED, 0.8),
        50_000_000,
    )
    assert predicted.value is not None and predicted.value > 0.2
    assert predicted.prediction_horizon_ns == 30_000_000


def test_predicted_input_preserves_exact_horizon_and_rejects_invalid_confidence() -> None:
    refiner = CausalTemporalRefiner([20])
    predicted = _process(
        refiner,
        TemporalSample(0.2, 0.7, TemporalState.PREDICTED, 10_000_000, 5_000_000),
        15_000_000,
    )
    assert predicted.value == 0.2
    assert predicted.prediction_horizon_ns == 5_000_000
    with pytest.raises(ValueError, match="exact horizon"):
        _process(
            refiner,
            TemporalSample(0.2, 0.7, TemporalState.PREDICTED, 20_000_000, 1),
            25_000_000,
        )

    class InvalidCalibration:
        @staticmethod
        def apply(signal_id: int, confidence: float) -> float:
            return 1.1

    invalid = CausalTemporalRefiner([20], confidence_calibration=InvalidCalibration())
    with pytest.raises(ValueError, match="calibrated temporal confidence"):
        _process(invalid, _sample(0.1, 10_000_000), 10_000_000)


def test_camera_or_generation_change_resets_history() -> None:
    refiner = CausalTemporalRefiner([20])
    _process(refiner, _sample(0.8, 10_000_000), 10_000_000)
    after_camera_switch = _process(
        refiner, _sample(0.0, 20_000_000), 20_000_000, camera_id="camera-b"
    )
    assert after_camera_switch.value == 0.0
    after_generation = _process(refiner, _sample(0.6, 10_000_000), 10_000_000, generation=1)
    assert after_generation.value == 0.6


def test_tracking_lost_is_a_prediction_history_barrier() -> None:
    refiner = CausalTemporalRefiner([20])
    _process(refiner, _sample(0.7, 10_000_000), 10_000_000)
    _process(
        refiner,
        _sample(None, 20_000_000, TemporalState.TRACKING_LOST, 0.0),
        20_000_000,
    )
    unavailable = _process(
        refiner,
        _sample(None, 30_000_000, TemporalState.OCCLUDED, 0.5),
        30_000_000,
    )
    assert unavailable.state == TemporalState.OCCLUDED
    assert unavailable.value is None


def test_confidence_calibration_is_monotonic_versioned_and_compatible(tmp_path: Path) -> None:
    predicted = np.tile(np.linspace(0.05, 0.95, 96)[:, None], (1, 2))
    correct = np.column_stack(
        (
            (predicted[:, 0] > 0.55).astype(np.int64),
            (predicted[:, 1] > 0.35).astype(np.int64),
        )
    )
    calibration = fit_confidence_calibration(
        predicted,
        correct,
        signal_ids=[1, 2],
        model_family="nana-face-basic",
        model_version="1.0.0",
        signal_registry_revision="ntp-signals/1.0.0",
        bins=8,
    )
    values = [calibration.apply(1, value) for value in np.linspace(0.0, 1.0, 50)]
    assert all(left <= right for left, right in pairwise(values))
    path = tmp_path / "confidence.json"
    calibration.save(path)
    loaded = ConfidenceCalibration.load_compatible(
        path,
        model_family="nana-face-basic",
        model_version="1.0.0",
        signal_registry_revision="ntp-signals/1.0.0",
    )
    assert loaded == calibration
    with pytest.raises(ValueError, match="incompatible"):
        ConfidenceCalibration.load_compatible(
            path,
            model_family="nana-face-basic",
            model_version="2.0.0",
            signal_registry_revision="ntp-signals/1.0.0",
        )


def test_temporal_benchmark_records_jitter_peak_and_overhead(tmp_path: Path) -> None:
    report = benchmark_temporal_refiner(tmp_path / "temporal.json", frames=100)
    jitter = report["static_jitter"]
    processing = report["processing_ms"]
    assert isinstance(jitter, dict) and jitter["reduction_percent"] > 0.0
    assert report["fast_peak_retention"] == 1.0
    assert isinstance(processing, dict) and processing["p99"] >= processing["p50"]
