"""Versioned per-signal monotonic confidence calibration."""

import json
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConfidenceCurve(_Contract):
    signal_id: int = Field(ge=1, le=88)
    thresholds: list[float]
    calibrated: list[float]

    @model_validator(mode="after")
    def validate_curve(self) -> Self:
        if len(self.thresholds) != len(self.calibrated) or not self.thresholds:
            raise ValueError("confidence curve arrays must be non-empty and aligned")
        if any(not 0.0 <= value <= 1.0 for value in (*self.thresholds, *self.calibrated)):
            raise ValueError("confidence curve values must be in [0, 1]")
        if any(
            left >= right for left, right in zip(self.thresholds, self.thresholds[1:], strict=False)
        ):
            raise ValueError("confidence thresholds must strictly increase")
        if any(
            left > right for left, right in zip(self.calibrated, self.calibrated[1:], strict=False)
        ):
            raise ValueError("calibrated confidence must be monotonic")
        return self


class ConfidenceCalibration(_Contract):
    schema_version: str = "ntp-confidence-calibration/1.0.0"
    model_family: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    curves: list[ConfidenceCurve]

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        ids = [curve.signal_id for curve in self.curves]
        if ids != sorted(set(ids)):
            raise ValueError("confidence curves must use unique increasing Signal IDs")
        return self

    def apply(self, signal_id: int, confidence: float) -> float:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("raw confidence must be in [0, 1]")
        curve = next((item for item in self.curves if item.signal_id == signal_id), None)
        if curve is None:
            return confidence
        return float(
            np.interp(
                confidence,
                curve.thresholds,
                curve.calibrated,
                left=curve.calibrated[0],
                right=curve.calibrated[-1],
            )
        )

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
        signal_registry_revision: str,
    ) -> Self:
        calibration = cls.model_validate_json(path.read_text(encoding="utf-8"))
        expected = (model_family, model_version, signal_registry_revision)
        actual = (
            calibration.model_family,
            calibration.model_version,
            calibration.signal_registry_revision,
        )
        if actual != expected:
            raise ValueError(
                "confidence calibration is incompatible with the active model contract: "
                f"expected={expected!r}, actual={actual!r}"
            )
        return calibration


def _pav(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    levels = [float(value) for value in values]
    masses = [float(value) for value in weights]
    starts = list(range(len(levels)))
    ends = list(range(len(levels)))
    index = 0
    while index < len(levels) - 1:
        if levels[index] <= levels[index + 1]:
            index += 1
            continue
        mass = masses[index] + masses[index + 1]
        level = (levels[index] * masses[index] + levels[index + 1] * masses[index + 1]) / mass
        levels[index : index + 2] = [level]
        masses[index : index + 2] = [mass]
        ends[index : index + 2] = [ends[index + 1]]
        starts[index : index + 2] = [starts[index]]
        index = max(index - 1, 0)
    fitted = np.empty(len(values), dtype=np.float64)
    for start, end, level in zip(starts, ends, levels, strict=True):
        fitted[start : end + 1] = level
    return fitted


def fit_confidence_calibration(
    predicted: np.ndarray,
    correct: np.ndarray,
    *,
    signal_ids: list[int],
    model_family: str,
    model_version: str,
    signal_registry_revision: str,
    bins: int = 12,
    minimum_bin_samples: int = 8,
) -> ConfidenceCalibration:
    """Fit per-signal isotonic reliability curves from held-out identity-safe evidence."""

    if predicted.shape != correct.shape or predicted.ndim != 2:
        raise ValueError("predicted and correct must be aligned [samples, signals] arrays")
    if predicted.shape[1] != len(signal_ids) or signal_ids != sorted(set(signal_ids)):
        raise ValueError("signal_ids must uniquely match the confidence columns")
    if not np.isfinite(predicted).all() or not np.isfinite(correct).all():
        raise ValueError("confidence evidence must be finite")
    if not ((predicted >= 0.0).all() and (predicted <= 1.0).all()):
        raise ValueError("predicted confidence must be in [0, 1]")
    if not np.isin(correct, [0, 1]).all():
        raise ValueError("correctness evidence must be binary")
    if bins < 2 or minimum_bin_samples < 1:
        raise ValueError("confidence bin settings are invalid")
    curves: list[ConfidenceCurve] = []
    for column, signal_id in enumerate(signal_ids):
        order = np.argsort(predicted[:, column], kind="stable")
        x = predicted[order, column]
        y = correct[order, column]
        bin_count = min(bins, len(x) // minimum_bin_samples)
        if bin_count < 2:
            raise ValueError(f"signal {signal_id} has insufficient confidence evidence")
        groups = np.array_split(np.arange(len(x)), bin_count)
        thresholds = np.array([float(x[group].mean()) for group in groups])
        empirical = np.array([float(y[group].mean()) for group in groups])
        weights = np.array([float(len(group)) for group in groups])
        fitted = _pav(empirical, weights)
        # Duplicate mean confidence can occur with quantized heads; merge deterministically.
        unique_thresholds: list[float] = []
        unique_values: list[float] = []
        for threshold, value in zip(thresholds, fitted, strict=True):
            if unique_thresholds and threshold <= unique_thresholds[-1]:
                unique_values[-1] = max(unique_values[-1], float(value))
            else:
                unique_thresholds.append(float(threshold))
                unique_values.append(float(value))
        if len(unique_thresholds) < 2:
            raise ValueError(f"signal {signal_id} confidence has no usable range")
        curves.append(
            ConfidenceCurve(
                signal_id=signal_id,
                thresholds=unique_thresholds,
                calibrated=unique_values,
            )
        )
    return ConfidenceCalibration(
        model_family=model_family,
        model_version=model_version,
        signal_registry_revision=signal_registry_revision,
        curves=curves,
    )
