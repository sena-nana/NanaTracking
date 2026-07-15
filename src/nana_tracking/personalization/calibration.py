"""Framework-neutral Level A neutral and user-range calibration."""

import json
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

_UNSIGNED_IDS = {9, 10, 13, 14, 15, 16, 17, 28, 30, 33, 34, 35, 36}


class CalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignalCalibration(CalibrationModel):
    signal_id: int = Field(ge=1, le=36)
    neutral: float
    negative_span: float = Field(gt=0)
    positive_span: float = Field(gt=0)


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

    @model_validator(mode="after")
    def validate_complete_basic(self) -> Self:
        ids = [signal.signal_id for signal in self.signals]
        if ids != list(range(1, 37)):
            raise ValueError("Level A profile must contain Basic signal IDs 1..36 in order")
        return self

    def apply(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != 36:
            raise ValueError("Level A calibration requires 36 Basic values")
        calibrated = np.empty_like(values, dtype=np.float32)
        for slot, signal in enumerate(self.signals):
            centered = values[..., slot] - signal.neutral
            scaled = np.where(
                centered < 0.0,
                centered / signal.negative_span,
                centered / signal.positive_span,
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
) -> LevelACalibration:
    """Fit robust offsets and asymmetric ranges from explicit calibration captures."""

    for name, values in {
        "neutral_samples": neutral_samples,
        "range_samples": range_samples,
        "confidence": confidence,
    }.items():
        if values.ndim != 2 or values.shape[1] != 36:
            raise ValueError(f"{name} must have shape [samples, 36]")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    if range_samples.shape != confidence.shape:
        raise ValueError("range_samples and confidence must have identical shapes")
    if len(neutral_samples) < minimum_samples or len(range_samples) < minimum_samples:
        raise ValueError("calibration capture does not contain enough samples")

    signals: list[SignalCalibration] = []
    neutral = np.median(neutral_samples, axis=0)
    for slot in range(36):
        usable = range_samples[confidence[:, slot] >= confidence_threshold, slot]
        if len(usable) < minimum_samples:
            raise ValueError(f"signal {slot + 1} has insufficient high-confidence range samples")
        centered = usable - neutral[slot]
        negative = max(abs(float(np.quantile(centered, 0.05))), 1e-3)
        positive = max(float(np.quantile(centered, 0.95)), 1e-3)
        signals.append(
            SignalCalibration(
                signal_id=slot + 1,
                neutral=float(neutral[slot]),
                negative_span=negative,
                positive_span=positive,
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
    )
