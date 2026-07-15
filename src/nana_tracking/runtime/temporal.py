"""Capture-time causal residual refinement for plain NTP scalar state."""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ConfidenceMapper(Protocol):
    def apply(self, signal_id: int, confidence: float) -> float: ...


class TemporalState(StrEnum):
    OBSERVED = "Observed"
    FUSED = "Fused"
    PREDICTED = "Predicted"
    OCCLUDED = "Occluded"
    OUT_OF_FRAME = "OutOfFrame"
    TRACKING_LOST = "TrackingLost"


@dataclass(frozen=True, slots=True)
class TemporalSample:
    value: float | None
    confidence: float
    state: TemporalState
    sample_capture_timestamp_ns: int
    prediction_horizon_ns: int = 0


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    history_frames: int = 8
    fast_time_constant_ns: int = 8_000_000
    expression_time_constant_ns: int = 18_000_000
    pose_time_constant_ns: int = 35_000_000
    body_time_constant_ns: int = 50_000_000
    auricle_time_constant_ns: int = 80_000_000
    peak_threshold: float = 0.18
    maximum_prediction_ns: int = 120_000_000
    prediction_confidence_half_life_ns: int = 55_000_000
    maximum_prediction_delta: float = 0.35

    def __post_init__(self) -> None:
        if not 4 <= self.history_frames <= 8:
            raise ValueError("causal history must retain 4 to 8 compressed frames")
        durations = (
            self.fast_time_constant_ns,
            self.expression_time_constant_ns,
            self.pose_time_constant_ns,
            self.body_time_constant_ns,
            self.auricle_time_constant_ns,
            self.maximum_prediction_ns,
            self.prediction_confidence_half_life_ns,
        )
        if any(value <= 0 for value in durations):
            raise ValueError("temporal durations must be positive")
        if self.peak_threshold <= 0.0 or self.maximum_prediction_delta <= 0.0:
            raise ValueError("temporal residual bounds must be positive")


@dataclass(frozen=True, slots=True)
class _Frame:
    capture_timestamp_ns: int
    samples: tuple[TemporalSample, ...]


class CausalTemporalRefiner:
    """Bounded residual refiner using only current and previous capture-time state."""

    def __init__(
        self,
        signal_ids: Sequence[int],
        *,
        config: TemporalConfig | None = None,
        confidence_calibration: ConfidenceMapper | None = None,
    ) -> None:
        ids = tuple(signal_ids)
        if not ids or ids != tuple(sorted(set(ids))) or any(not 1 <= value <= 88 for value in ids):
            raise ValueError("signal_ids must be unique increasing stable IDs")
        self.signal_ids = ids
        self.config = config or TemporalConfig()
        self.confidence_calibration = confidence_calibration
        self._history: deque[_Frame] = deque(maxlen=self.config.history_frames)
        self._scope: tuple[bytes, int, str, str] | None = None

    def reset(self) -> None:
        self._history.clear()
        self._scope = None

    def process(
        self,
        samples: Sequence[TemporalSample],
        *,
        capture_timestamp_ns: int,
        session_id: bytes,
        generation: int,
        camera_id: str,
        calibration_revision: str,
    ) -> tuple[TemporalSample, ...]:
        if len(samples) != len(self.signal_ids):
            raise ValueError("sample count does not match configured Signal IDs")
        scope = (session_id, generation, camera_id, calibration_revision)
        if self._scope != scope:
            self._history.clear()
            self._scope = scope
        if self._history and capture_timestamp_ns <= self._history[-1].capture_timestamp_ns:
            raise ValueError("capture timestamps must strictly increase within a temporal scope")
        previous = self._history[-1] if self._history else None
        refined = tuple(
            self._refine_one(signal_id, sample, previous, capture_timestamp_ns, index)
            for index, (signal_id, sample) in enumerate(zip(self.signal_ids, samples, strict=True))
        )
        self._history.append(_Frame(capture_timestamp_ns, refined))
        return refined

    def _refine_one(
        self,
        signal_id: int,
        sample: TemporalSample,
        previous: _Frame | None,
        now_ns: int,
        index: int,
    ) -> TemporalSample:
        if not 0.0 <= sample.confidence <= 1.0:
            raise ValueError("temporal input confidence must be in [0, 1]")
        if sample.sample_capture_timestamp_ns < 0 or sample.sample_capture_timestamp_ns > now_ns:
            raise ValueError("temporal sample timestamp must not be in the future")
        confidence = (
            self.confidence_calibration.apply(signal_id, sample.confidence)
            if self.confidence_calibration is not None
            else sample.confidence
        )
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("calibrated temporal confidence must be in [0, 1]")
        if sample.state in {TemporalState.OBSERVED, TemporalState.FUSED}:
            if sample.prediction_horizon_ns != 0:
                raise ValueError("observed temporal samples cannot carry a prediction horizon")
            if sample.value is None or not math.isfinite(sample.value):
                raise ValueError("observed temporal samples require a finite value")
            value = sample.value
            if previous is not None:
                prior = previous.samples[index]
                if prior.value is not None and prior.state in {
                    TemporalState.OBSERVED,
                    TemporalState.FUSED,
                }:
                    dt = now_ns - previous.capture_timestamp_ns
                    retention = math.exp(-dt / self._time_constant(signal_id))
                    if self._preserve_peak(signal_id, value - prior.value):
                        retention = 0.0
                    value += retention * (prior.value - value)
            return TemporalSample(
                value=value,
                confidence=confidence,
                state=sample.state,
                sample_capture_timestamp_ns=sample.sample_capture_timestamp_ns,
            )
        if sample.state == TemporalState.PREDICTED:
            expected_horizon = now_ns - sample.sample_capture_timestamp_ns
            if (
                sample.value is None
                or not math.isfinite(sample.value)
                or sample.prediction_horizon_ns <= 0
                or sample.prediction_horizon_ns != expected_horizon
            ):
                raise ValueError("predicted temporal samples require a value and exact horizon")
            return TemporalSample(
                value=sample.value,
                confidence=confidence,
                state=sample.state,
                sample_capture_timestamp_ns=sample.sample_capture_timestamp_ns,
                prediction_horizon_ns=sample.prediction_horizon_ns,
            )
        if sample.prediction_horizon_ns != 0:
            raise ValueError("unavailable temporal samples cannot carry a prediction horizon")
        if sample.value is not None:
            raise ValueError("unavailable temporal samples cannot carry a value")
        if sample.state == TemporalState.TRACKING_LOST:
            return TemporalSample(None, 0.0, TemporalState.TRACKING_LOST, now_ns)
        if sample.state not in {TemporalState.OCCLUDED, TemporalState.OUT_OF_FRAME}:
            return TemporalSample(
                None, confidence, sample.state, sample.sample_capture_timestamp_ns
            )
        last, older = self._last_two_observations(index)
        if last is None:
            return TemporalSample(
                None, confidence, sample.state, sample.sample_capture_timestamp_ns
            )
        assert last.value is not None
        horizon = now_ns - last.sample_capture_timestamp_ns
        if horizon <= 0 or horizon > self.config.maximum_prediction_ns:
            return TemporalSample(
                None, 0.0, TemporalState.TRACKING_LOST, last.sample_capture_timestamp_ns
            )
        velocity = 0.0
        if older is not None:
            dt = last.sample_capture_timestamp_ns - older.sample_capture_timestamp_ns
            if dt > 0 and older.value is not None:
                velocity = (last.value - older.value) / dt
        delta = max(
            -self.config.maximum_prediction_delta,
            min(self.config.maximum_prediction_delta, velocity * horizon),
        )
        predicted = max(-1.0, min(1.0, last.value + delta))
        decay = 0.5 ** (horizon / self.config.prediction_confidence_half_life_ns)
        return TemporalSample(
            value=predicted,
            confidence=min(confidence, last.confidence) * decay,
            state=TemporalState.PREDICTED,
            sample_capture_timestamp_ns=last.sample_capture_timestamp_ns,
            prediction_horizon_ns=horizon,
        )

    def _last_two_observations(
        self, index: int
    ) -> tuple[TemporalSample | None, TemporalSample | None]:
        values: list[TemporalSample] = []
        latest_source_timestamp: int | None = None
        for frame in reversed(self._history):
            sample = frame.samples[index]
            if sample.state == TemporalState.TRACKING_LOST:
                break
            if (
                sample.value is not None
                and sample.state
                in {
                    TemporalState.OBSERVED,
                    TemporalState.FUSED,
                }
                and sample.sample_capture_timestamp_ns != latest_source_timestamp
            ):
                values.append(sample)
                latest_source_timestamp = sample.sample_capture_timestamp_ns
        return (
            values[0] if values else None,
            values[1] if len(values) > 1 else None,
        )

    def _time_constant(self, signal_id: int) -> int:
        if signal_id in {7, 8, 17, 28, 41, 54, 55, 56}:
            return self.config.fast_time_constant_ns
        if 1 <= signal_id <= 41:
            return self.config.expression_time_constant_ns
        if 42 <= signal_id <= 53:
            return self.config.pose_time_constant_ns
        if 57 <= signal_id <= 62:
            return self.config.auricle_time_constant_ns
        return self.config.body_time_constant_ns

    def _preserve_peak(self, signal_id: int, delta: float) -> bool:
        return (
            signal_id in {7, 8, 17, 28, 41, 54, 55, 56, 70, 75}
            and abs(delta) >= self.config.peak_threshold
        )
