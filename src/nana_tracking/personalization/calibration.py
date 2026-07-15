"""Framework-neutral Level A neutral and user-range calibration."""

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

_UNSIGNED_IDS = {9, 10, 13, 14, 15, 16, 17, 28, 30, 33, 34, 35, 36, 41, 70, 75, 79, 80}


class CalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignalCalibration(CalibrationModel):
    signal_id: int = Field(ge=1, le=88)
    neutral: float
    negative_span: float = Field(gt=0)
    positive_span: float = Field(gt=0)
    deadzone: float = Field(default=0.0, ge=0.0, lt=1.0)


class LevelACalibration(CalibrationModel):
    schema_version: str = "ntp-level-a-calibration/1.0.0"
    user_slot: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    calibration_revision: str = Field(min_length=1)
    signals: list[SignalCalibration]
    shoulder_width: float | None = Field(default=None, gt=0.0)
    torso_neutral_xyz: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def validate_signal_order(self) -> Self:
        ids = [signal.signal_id for signal in self.signals]
        if not ids or ids != sorted(set(ids)):
            raise ValueError("Level A Signal IDs must be unique and increasing")
        return self

    def apply(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.signals):
            raise ValueError("Level A values must match the calibrated Signal ID count")
        calibrated = np.empty_like(values, dtype=np.float32)
        for slot, signal in enumerate(self.signals):
            centered = values[..., slot] - signal.neutral
            scaled = np.where(
                centered < 0.0,
                centered / signal.negative_span,
                centered / signal.positive_span,
            )
            scaled = np.where(
                np.abs(scaled) <= signal.deadzone,
                0.0,
                np.sign(scaled) * (np.abs(scaled) - signal.deadzone) / (1.0 - signal.deadzone),
            )
            low = 0.0 if signal.signal_id in _UNSIGNED_IDS else -1.0
            calibrated[..., slot] = np.clip(scaled, low, 1.0)
        return calibrated

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_compatible(
        cls,
        path: Path,
        *,
        model_family: str,
        model_version: str,
        feature_revision: str,
        signal_registry_revision: str,
    ) -> Self:
        profile = cls.model_validate_json(path.read_text(encoding="utf-8"))
        expected = (
            model_family,
            model_version,
            feature_revision,
            signal_registry_revision,
        )
        actual = (
            profile.model_family,
            profile.model_version,
            profile.feature_revision,
            profile.signal_registry_revision,
        )
        if actual != expected:
            raise ValueError(
                "calibration is incompatible with the active model contract: "
                f"expected={expected!r}, actual={actual!r}"
            )
        return profile


def fit_level_a_calibration(
    neutral_samples: np.ndarray,
    range_samples: np.ndarray,
    confidence: np.ndarray,
    *,
    user_slot: str,
    model_family: str,
    model_version: str,
    feature_revision: str,
    signal_registry_revision: str,
    normalization_revision: str,
    calibration_revision: str,
    minimum_samples: int = 20,
    confidence_threshold: float = 0.8,
    signal_ids: Sequence[int] = tuple(range(1, 37)),
    deadzone: float = 0.0,
    shoulder_width: float | None = None,
    torso_neutral_xyz: tuple[float, float, float] | None = None,
) -> LevelACalibration:
    """Fit robust offsets and asymmetric ranges from explicit calibration captures."""

    ids = tuple(signal_ids)
    if not ids or ids != tuple(sorted(set(ids))) or any(not 1 <= value <= 88 for value in ids):
        raise ValueError("signal_ids must be unique increasing stable IDs")
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    for name, values in {
        "neutral_samples": neutral_samples,
        "range_samples": range_samples,
        "confidence": confidence,
    }.items():
        if values.ndim != 2 or values.shape[1] != len(ids):
            raise ValueError(f"{name} must have shape [samples, {len(ids)}]")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    if range_samples.shape != confidence.shape:
        raise ValueError("range_samples and confidence must have identical shapes")
    if len(neutral_samples) < minimum_samples or len(range_samples) < minimum_samples:
        raise ValueError("calibration capture does not contain enough samples")

    signals: list[SignalCalibration] = []
    neutral = np.median(neutral_samples, axis=0)
    for slot, signal_id in enumerate(ids):
        usable = range_samples[confidence[:, slot] >= confidence_threshold, slot]
        if len(usable) < minimum_samples:
            raise ValueError(f"signal {slot + 1} has insufficient high-confidence range samples")
        centered = usable - neutral[slot]
        negative = max(abs(float(np.quantile(centered, 0.05))), 1e-3)
        positive = max(float(np.quantile(centered, 0.95)), 1e-3)
        signals.append(
            SignalCalibration(
                signal_id=signal_id,
                neutral=float(neutral[slot]),
                negative_span=negative,
                positive_span=positive,
                deadzone=deadzone,
            )
        )
    return LevelACalibration(
        user_slot=user_slot,
        model_family=model_family,
        model_version=model_version,
        feature_revision=feature_revision,
        signal_registry_revision=signal_registry_revision,
        normalization_revision=normalization_revision,
        calibration_revision=calibration_revision,
        signals=signals,
        shoulder_width=shoulder_width,
        torso_neutral_xyz=torso_neutral_xyz,
    )


@dataclass(frozen=True, slots=True)
class OnlineCalibrationSnapshot:
    capture_timestamp_ns: int
    signals: tuple[SignalCalibration, ...]


class BoundedOnlineCalibration:
    """Level C guardrail for explicit, slow, resettable neutral/range adaptation."""

    def __init__(
        self,
        baseline: LevelACalibration,
        *,
        minimum_confidence: float = 0.9,
        minimum_stable_ns: int = 1_000_000_000,
        maximum_neutral_drift: float = 0.05,
        maximum_span_drift_fraction: float = 0.1,
        maximum_step: float = 0.001,
        rollback_depth: int = 8,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum confidence must be in [0, 1]")
        if minimum_stable_ns <= 0 or maximum_step <= 0.0 or rollback_depth < 1:
            raise ValueError("online calibration bounds must be positive")
        self.baseline = baseline
        self.current = baseline.model_copy(deep=True)
        self.minimum_confidence = minimum_confidence
        self.minimum_stable_ns = minimum_stable_ns
        self.maximum_neutral_drift = maximum_neutral_drift
        self.maximum_span_drift_fraction = maximum_span_drift_fraction
        self.maximum_step = maximum_step
        self._history: deque[OnlineCalibrationSnapshot] = deque(maxlen=rollback_depth)

    def update(
        self,
        values: np.ndarray,
        confidence: np.ndarray,
        *,
        capture_timestamp_ns: int,
        stable_duration_ns: int,
        evidence: Literal["explicit_neutral", "explicit_range"],
    ) -> bool:
        signal_count = len(self.current.signals)
        if values.shape != (signal_count,) or confidence.shape != (signal_count,):
            raise ValueError("online calibration values must match the Level A signal count")
        if not np.isfinite(values).all() or not np.isfinite(confidence).all():
            raise ValueError("online calibration evidence must be finite")
        if stable_duration_ns < self.minimum_stable_ns:
            return False
        eligible = confidence >= self.minimum_confidence
        if not bool(eligible.any()):
            return False
        self._history.append(
            OnlineCalibrationSnapshot(capture_timestamp_ns, tuple(self.current.signals))
        )
        updated: list[SignalCalibration] = []
        for slot, (base, active) in enumerate(
            zip(self.baseline.signals, self.current.signals, strict=True)
        ):
            if not eligible[slot]:
                updated.append(active)
                continue
            if evidence == "explicit_neutral":
                delta = float(
                    np.clip(values[slot] - active.neutral, -self.maximum_step, self.maximum_step)
                )
                neutral = float(
                    np.clip(
                        active.neutral + delta,
                        base.neutral - self.maximum_neutral_drift,
                        base.neutral + self.maximum_neutral_drift,
                    )
                )
                updated.append(active.model_copy(update={"neutral": neutral}))
            else:
                centered = float(values[slot] - active.neutral)
                field = "negative_span" if centered < 0.0 else "positive_span"
                current_span = getattr(active, field)
                target = abs(centered)
                step = float(np.clip(target - current_span, -self.maximum_step, self.maximum_step))
                limit = getattr(base, field) * self.maximum_span_drift_fraction
                span = float(
                    np.clip(
                        current_span + step,
                        getattr(base, field) - limit,
                        getattr(base, field) + limit,
                    )
                )
                updated.append(active.model_copy(update={field: max(span, 1e-3)}))
        self.current = self.current.model_copy(update={"signals": updated})
        return True

    def rollback(self) -> bool:
        if not self._history:
            return False
        snapshot = self._history.pop()
        self.current = self.current.model_copy(update={"signals": list(snapshot.signals)})
        return True

    def reset(self) -> None:
        self.current = self.baseline.model_copy(deep=True)
        self._history.clear()
