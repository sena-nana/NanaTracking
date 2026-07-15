"""Frozen-F parameter-sequence expression baseline and required ablations."""

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

_REQUIRED_ABLATIONS = {
    "all_parameters",
    "single_frame_parameters",
    "parameters_velocity",
    "parameters_velocity_acceleration",
    "mouth_jaw_only",
    "without_mouth_viseme",
    "head_pose_only",
    "without_head_pose",
    "shuffled_time",
    "rgb_upper_bound",
}
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


class ExpressionModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AblationSpec(ExpressionModelConfig):
    name: str = Field(min_length=1)
    parameter_subset: Literal["all", "mouth-jaw", "non-mouth", "none"]
    temporal_features: Literal["single", "position", "velocity", "acceleration", "shuffled"]
    include_head_pose: bool
    rgb_upper_bound: bool = False


class ExpressionAblationConfig(ExpressionModelConfig):
    schema_version: Literal["nana-expression-ablation/1.0.0"]
    seed: int = Field(ge=0)
    steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    hidden_dims: int = Field(gt=0)
    emotion_labels: list[str] = Field(min_length=2)
    ablations: list[AblationSpec]
    smoke_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_complete_suite(self) -> Self:
        names = [ablation.name for ablation in self.ablations]
        if set(names) != _REQUIRED_ABLATIONS or len(names) != len(set(names)):
            raise ValueError("expression ablations must contain the complete unique required suite")
        if "neutral" not in self.emotion_labels:
            raise ValueError("expression labels require a neutral class")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class ParameterExpressionModel(nn.Module):
    """Small G baseline; it consumes F outputs and has no reference to model F weights."""

    def __init__(self, input_dims: int, hidden_dims: int, emotion_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.GELU(),
        )
        self.emotion = nn.Linear(hidden_dims, emotion_count)
        self.intensity = nn.Linear(hidden_dims, 1)
        self.confidence = nn.Linear(hidden_dims, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.encoder(features)
        return (
            self.emotion(encoded),
            torch.sigmoid(self.intensity(encoded)).squeeze(-1),
            torch.sigmoid(self.confidence(encoded)).squeeze(-1),
        )


def _synthetic_frozen_f_sequences(
    *, seed: int, emotion_count: int, neutral_index: int
) -> tuple[dict[str, FloatArray], IntArray, FloatArray, FloatArray, IntArray]:
    rng = np.random.default_rng(seed)
    actor_count = 15
    clips_per_actor = emotion_count * 2
    frame_count = 12
    sample_count = actor_count * clips_per_actor
    parameters = np.zeros((sample_count, frame_count, 36), dtype=np.float32)
    confidence = rng.uniform(0.82, 1.0, size=parameters.shape).astype(np.float32)
    visibility = rng.uniform(0.85, 1.0, size=(sample_count, frame_count, 1)).astype(np.float32)
    head_pose = rng.normal(0.0, 0.08, size=(sample_count, frame_count, 7)).astype(np.float32)
    rgb_proxy = rng.normal(0.0, 0.04, size=(sample_count, frame_count, 12)).astype(np.float32)
    labels = np.empty(sample_count, dtype=np.int64)
    intensity = np.empty(sample_count, dtype=np.float32)
    label_confidence = rng.uniform(0.78, 0.98, size=sample_count).astype(np.float32)
    actor_ids = np.empty(sample_count, dtype=np.int64)
    trajectory = np.sin(np.linspace(0.0, np.pi, frame_count, dtype=np.float32))
    prototypes = np.zeros((emotion_count, 36), dtype=np.float32)
    for label in range(emotion_count):
        if label == neutral_index:
            continue
        first = ((label - 1) * 5) % 16
        prototypes[label, first : first + 4] = np.array([0.8, -0.55, 0.65, -0.4])
        prototypes[label, 19 + ((label - 1) * 3) % 14 : 22 + ((label - 1) * 3) % 14] = (
            0.25 + 0.08 * label
        )
    index = 0
    for actor in range(actor_count):
        actor_bias = rng.normal(0.0, 0.035, size=36).astype(np.float32)
        for clip in range(clips_per_actor):
            label = clip % emotion_count
            strength = 0.0 if label == neutral_index else float(rng.uniform(0.55, 1.0))
            signal = trajectory[:, None] * prototypes[label][None, :] * strength
            viseme = rng.normal(0.0, 0.18, size=(frame_count, 20)).astype(np.float32)
            parameters[index] = signal + actor_bias + rng.normal(0.0, 0.035, size=(frame_count, 36))
            parameters[index, :, 16:] += viseme
            rgb_proxy[index, :, label] += 4.0
            labels[index] = label
            intensity[index] = strength
            actor_ids[index] = actor
            index += 1
    arrays = {
        "parameters": parameters,
        "confidence": confidence,
        "visibility": visibility,
        "head_pose": head_pose,
        "rgb_proxy": rgb_proxy,
    }
    return arrays, labels, intensity, label_confidence, actor_ids


def _features(arrays: dict[str, FloatArray], spec: AblationSpec, *, seed: int) -> FloatArray:
    if spec.rgb_upper_bound:
        values = arrays["rgb_proxy"]
    else:
        parameters = arrays["parameters"] * arrays["confidence"] * arrays["visibility"]
        if spec.parameter_subset == "mouth-jaw":
            values = parameters[:, :, 16:]
        elif spec.parameter_subset == "non-mouth":
            values = parameters[:, :, :16]
        elif spec.parameter_subset == "none":
            values = np.empty((parameters.shape[0], parameters.shape[1], 0), dtype=np.float32)
        else:
            values = parameters
    if spec.temporal_features == "shuffled":
        rng = np.random.default_rng(seed + 991)
        values = np.take_along_axis(
            values,
            np.stack([rng.permutation(values.shape[1]) for _ in range(values.shape[0])])[
                :, :, None
            ],
            axis=1,
        )
    if spec.temporal_features == "single":
        blocks = [values[:, values.shape[1] // 2]]
    else:
        blocks = [values.mean(axis=1), values.std(axis=1)]
        if spec.temporal_features in {"velocity", "acceleration", "shuffled"}:
            velocity = np.diff(values, axis=1)
            blocks.extend([velocity.mean(axis=1), np.abs(velocity).mean(axis=1)])
        if spec.temporal_features == "acceleration":
            acceleration = np.diff(values, n=2, axis=1)
            blocks.extend([acceleration.mean(axis=1), np.abs(acceleration).mean(axis=1)])
    if spec.include_head_pose:
        blocks.extend([arrays["head_pose"].mean(axis=1), arrays["head_pose"].std(axis=1)])
    blocks.append(arrays["visibility"].mean(axis=1))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _macro_metrics(
    labels: IntArray, predictions: IntArray, class_count: int
) -> tuple[float, float]:
    f1_values: list[float] = []
    recalls: list[float] = []
    actual_values = labels.tolist()
    predicted_values = predictions.tolist()
    for label in range(class_count):
        pairs = zip(actual_values, predicted_values, strict=True)
        true_positive = sum(actual == label and predicted == label for actual, predicted in pairs)
        pairs = zip(actual_values, predicted_values, strict=True)
        false_positive = sum(actual != label and predicted == label for actual, predicted in pairs)
        pairs = zip(actual_values, predicted_values, strict=True)
        false_negative = sum(actual == label and predicted != label for actual, predicted in pairs)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1_values.append(2.0 * precision * recall / max(1e-12, precision + recall))
        recalls.append(recall)
    return float(np.mean(f1_values)), float(np.mean(recalls))


def _calibration_error(probabilities: FloatArray, labels: IntArray) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    error = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= low) & (confidence < low + 0.1)
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(confidence[mask].mean()) - float(correct[mask].mean())
            )
    return error


def run_expression_ablation_smoke(
    config: ExpressionAblationConfig,
    output: Path,
) -> dict[str, object]:
    """Train model G only on synthetic frozen-F-like sequences and record all ablations."""

    torch.manual_seed(config.seed)
    arrays, labels, intensity, label_confidence, actors = _synthetic_frozen_f_sequences(
        seed=config.seed,
        emotion_count=len(config.emotion_labels),
        neutral_index=config.emotion_labels.index("neutral"),
    )
    unique_actors = np.unique(actors)
    test_actors = set(unique_actors[:3].tolist())
    validation_actors = set(unique_actors[3:6].tolist())
    train_mask = np.array(
        [actor not in test_actors | validation_actors for actor in actors], dtype=bool
    )
    validation_mask = np.array([actor in validation_actors for actor in actors], dtype=bool)
    test_mask = np.array([actor in test_actors for actor in actors], dtype=bool)
    results: dict[str, dict[str, float]] = {}
    suite_started = time.perf_counter()
    for offset, spec in enumerate(config.ablations):
        ablation_started = time.perf_counter()
        torch.manual_seed(config.seed + offset)
        features = _features(arrays, spec, seed=config.seed)
        mean = features[train_mask].mean(axis=0, keepdims=True)
        scale = features[train_mask].std(axis=0, keepdims=True).clip(min=1e-5)
        features = (features - mean) / scale
        x = torch.from_numpy(features)
        y = torch.from_numpy(labels)
        target_intensity = torch.from_numpy(intensity)
        target_confidence = torch.from_numpy(label_confidence)
        model = ParameterExpressionModel(
            features.shape[1], config.hidden_dims, len(config.emotion_labels)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        model.train()
        train_indices = torch.from_numpy(np.flatnonzero(train_mask))
        for _ in range(config.steps):
            logits, predicted_intensity, predicted_confidence = model(x[train_indices])
            loss = (
                nn.functional.cross_entropy(logits, y[train_indices])
                + 0.4
                * nn.functional.smooth_l1_loss(predicted_intensity, target_intensity[train_indices])
                + 0.1
                * nn.functional.binary_cross_entropy(
                    predicted_confidence, target_confidence[train_indices]
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits, predicted_intensity, _ = model(x[test_mask])
            probabilities = torch.softmax(logits, dim=-1).numpy()
        macro_f1, balanced_accuracy = _macro_metrics(
            labels[test_mask], probabilities.argmax(axis=1), len(config.emotion_labels)
        )
        results[spec.name] = {
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "intensity_mae": float(
                np.abs(predicted_intensity.numpy() - intensity[test_mask]).mean()
            ),
            "expected_calibration_error": _calibration_error(probabilities, labels[test_mask]),
            "train_eval_ms": (time.perf_counter() - ablation_started) * 1_000.0,
        }
    config_payload = config.model_dump(mode="json")
    report: dict[str, object] = {
        "schema_version": "nana-expression-ablation-report/1.0.0",
        "smoke_only": True,
        "warning": (
            "Synthetic smoke evidence does not validate CREMA-D or production expression quality."
        ),
        "frozen_f_source": "synthetic-frozen-f-predictions",
        "f_weights_trainable": False,
        "actor_split": {
            "train": sorted(set(actors[train_mask].tolist())),
            "validation": sorted(set(actors[validation_mask].tolist())),
            "test": sorted(set(actors[test_mask].tolist())),
        },
        "config_sha256": hashlib.sha256(
            json.dumps(config_payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "machine": platform.machine(),
            "platform": platform.platform(),
            "sample_count": int(labels.size),
            "suite_elapsed_ms": (time.perf_counter() - suite_started) * 1_000.0,
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
