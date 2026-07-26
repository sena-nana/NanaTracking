"""Identity-clustered evaluation for commercial first-party Stage A ablations."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import MultiViewTrackingBatch
from nana_tracking.data.loaders import create_loader
from nana_tracking.data.manifest import DatasetManifest
from nana_tracking.models import create_model, output_names
from nana_tracking.reproducibility import choose_device, sha256_file, sha256_json
from nana_tracking.training.checkpoint import load_checkpoint
from nana_tracking.training.stage_a import quaternion_geodesic, quaternion_to_matrix


def _forward(config: ExperimentConfig, model: torch.nn.Module, images: Tensor) -> dict[str, Tensor]:
    groups, views, channels, height, width = images.shape
    values = model(images.reshape(groups * views, channels, height, width))
    return {
        name: value.reshape(groups, views, *value.shape[1:])
        for name, value in zip(output_names(config.model), values, strict=True)
    }


def _project(
    geometry: Tensor,
    pose: Tensor,
    intrinsics: Tensor,
    *,
    width: int,
    height: int,
) -> Tensor:
    rotation = quaternion_to_matrix(pose[..., 3:])
    points = torch.einsum("bvij,bvnj->bvni", rotation, geometry) + pose[..., None, :3]
    homogeneous = torch.einsum("bvij,bvnj->bvni", intrinsics, points)
    pixels = homogeneous[..., :2] / homogeneous[..., 2:].clamp_min(1e-4)
    return torch.stack(
        (
            pixels[..., 0] / width * 2.0 - 1.0,
            pixels[..., 1] / height * 2.0 - 1.0,
        ),
        dim=-1,
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _temporal_metrics(
    sequences: dict[tuple[str, str, str], list[tuple[int, float, float]]],
) -> dict[str, dict[str, list[float]]]:
    per_identity: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (identity, _sequence, _expression), samples in sequences.items():
        ordered = sorted(samples)
        if len(ordered) < 3:
            continue
        timestamps = np.asarray([item[0] for item in ordered], dtype=np.int64)
        predicted = np.asarray([item[1] for item in ordered], dtype=np.float64)
        target = np.asarray([item[2] for item in ordered], dtype=np.float64)
        residual = predicted - target
        per_identity[identity]["temporal_jitter"].append(float(np.std(np.diff(residual))))
        target_range = float(np.ptp(target))
        if target_range <= 1e-6:
            continue
        predicted_range = float(np.ptp(predicted))
        per_identity[identity]["peak_attenuation"].append(
            abs(predicted_range - target_range) / target_range
        )
        neutral = float(target[0])
        peak = int(np.argmax(np.abs(target - neutral)))
        threshold = target_range * 0.1
        target_recovery = next(
            (
                index
                for index in range(peak + 1, len(target))
                if abs(target[index] - neutral) <= threshold
            ),
            None,
        )
        predicted_recovery = next(
            (
                index
                for index in range(peak + 1, len(predicted))
                if abs(predicted[index] - neutral) <= threshold
            ),
            None,
        )
        if target_recovery is not None:
            predicted_recovery = predicted_recovery or len(predicted) - 1
            delay_ns = max(
                0,
                int(timestamps[predicted_recovery] - timestamps[target_recovery]),
            )
            per_identity[identity]["recovery_ms"].append(delay_ns / 1_000_000.0)
    return per_identity


def _evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    device = choose_device(config.training.device)
    model = create_model(config.model).to(device)
    metadata = load_checkpoint(
        checkpoint,
        model=model,
        expected_usage_tier="commercial",
    )
    model.eval()
    loader = create_loader(config, split="test", shuffle=False, seed_offset=20_000)
    per_identity_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sequences: dict[tuple[str, str, str], list[tuple[int, float, float]]] = defaultdict(list)
    with torch.inference_mode():
        for raw_batch in loader:
            if not isinstance(raw_batch, MultiViewTrackingBatch):
                raise ValueError("Stage A evaluation requires MultiViewTrackingBatch")
            images = raw_batch.images.to(device)
            targets = {name: value.to(device) for name, value in raw_batch.targets.items()}
            intrinsics = raw_batch.camera_intrinsics.to(device)
            outputs = _forward(config, model, images)
            projected = _project(
                outputs["canonical_geometry"],
                targets["pose"],
                intrinsics,
                width=config.model.input_width,
                height=config.model.input_height,
            )
            for index, identity in enumerate(raw_batch.identity_ids):
                target_2d = targets["landmarks"][index]
                prediction_2d = outputs["landmarks"][index]
                interocular = (
                    torch.linalg.vector_norm(target_2d[:, 1] - target_2d[:, 3], dim=-1)
                    .mean()
                    .clamp_min(1e-6)
                )
                landmark_nme = (
                    torch.linalg.vector_norm(prediction_2d - target_2d, dim=-1).mean() / interocular
                )
                target_3d = targets["canonical_geometry"][index]
                prediction_3d = outputs["canonical_geometry"][index]
                geometry_scale = (
                    torch.linalg.vector_norm(target_3d[:, 1] - target_3d[:, 3], dim=-1)
                    .mean()
                    .clamp_min(1e-6)
                )
                geometry_nme = (
                    torch.linalg.vector_norm(prediction_3d - target_3d, dim=-1).mean()
                    / geometry_scale
                )
                pose_rotation = quaternion_geodesic(
                    outputs["pose"][index, :, 3:],
                    targets["pose"][index, :, 3:],
                ).mean()
                pose_translation = (
                    (outputs["pose"][index, :, :3] - targets["pose"][index, :, :3]).abs().mean()
                )
                reprojection_delta = projected[index] - target_2d
                reprojection_pixels = torch.linalg.vector_norm(
                    torch.stack(
                        (
                            reprojection_delta[..., 0] * config.model.input_width * 0.5,
                            reprojection_delta[..., 1] * config.model.input_height * 0.5,
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                ).mean()
                visibility_accuracy = (
                    (outputs["visibility"][index].argmax(dim=-1) == targets["visibility"][index])
                    .float()
                    .mean()
                )
                view_pairs: list[Tensor] = []
                for left in range(3):
                    for right in range(left + 1, 3):
                        view_pairs.append(
                            cast(
                                Tensor,
                                torch.linalg.vector_norm(
                                    prediction_3d[left] - prediction_3d[right], dim=-1
                                ).mean(),
                            )
                        )
                sample_metrics: dict[str, float] = {
                    "landmark_nme": cast(float, landmark_nme.item()),
                    "canonical_3d_nme": cast(float, geometry_nme.item()),
                    "pose_rotation_degrees": cast(float, torch.rad2deg(pose_rotation).item()),
                    "pose_translation_mae": cast(float, pose_translation.item()),
                    "reprojection_pixels": cast(float, reprojection_pixels.item()),
                    "visibility_accuracy_error": cast(float, (1.0 - visibility_accuracy).item()),
                    "multiview_geometry_consistency": cast(
                        float, torch.stack(view_pairs).mean().item()
                    ),
                }
                for name, value in sample_metrics.items():
                    per_identity_values[identity][name].append(value)
                mouth_prediction = torch.linalg.vector_norm(
                    prediction_2d[:, 13] - prediction_2d[:, 14], dim=-1
                ).mean()
                mouth_target = torch.linalg.vector_norm(
                    target_2d[:, 13] - target_2d[:, 14], dim=-1
                ).mean()
                sequences[
                    (
                        identity,
                        raw_batch.sequence_ids[index],
                        raw_batch.expressions[index],
                    )
                ].append(
                    (
                        cast(int, raw_batch.timestamps_ns[index].item()),
                        cast(float, mouth_prediction.item()),
                        cast(float, mouth_target.item()),
                    )
                )
    temporal = _temporal_metrics(sequences)
    for identity, temporal_metrics in temporal.items():
        for name, values in temporal_metrics.items():
            per_identity_values[identity][name].extend(values)
    reduced = {
        identity: {name: _mean(values) for name, values in metrics.items()}
        for identity, metrics in per_identity_values.items()
    }
    return reduced, metadata.model_dump(mode="json")


def _paired_bootstrap(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    *,
    seed: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    identities = sorted(set(baseline) & set(candidate))
    if not identities:
        raise ValueError("Stage A checkpoints have no common test identities")
    shared_metrics = set(baseline[identities[0]]) & set(candidate[identities[0]])
    for identity in identities[1:]:
        shared_metrics &= set(baseline[identity]) & set(candidate[identity])
    metrics = sorted(shared_metrics)
    rng = np.random.default_rng(seed)
    report: dict[str, dict[str, float]] = {}
    for metric in metrics:
        deltas = np.asarray(
            [candidate[identity][metric] - baseline[identity][metric] for identity in identities],
            dtype=np.float64,
        )
        draws = rng.integers(0, len(identities), size=(samples, len(identities)))
        estimates = deltas[draws].mean(axis=1)
        report[metric] = {
            "candidate_minus_baseline": float(deltas.mean()),
            "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)),
        }
    return report


def evaluate_stage_a_comparison(
    config: ExperimentConfig,
    *,
    single_view_checkpoint: Path,
    multiview_checkpoint: Path,
    output_path: Path,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if config.data.usage_tier != "commercial" or config.data.dataset not in {
        "manifest",
        "frozen_capture",
    }:
        raise ValueError("Stage A comparison requires a commercial first-party manifest")
    baseline, baseline_metadata = _evaluate_checkpoint(config, single_view_checkpoint)
    candidate, candidate_metadata = _evaluate_checkpoint(config, multiview_checkpoint)
    bootstrap = _paired_bootstrap(
        baseline,
        candidate,
        seed=config.training.seed,
        samples=bootstrap_samples,
    )
    if config.data.manifest is None:
        raise ValueError("Stage A evaluation requires a manifest")
    manifest = DatasetManifest.load(config.data.manifest)
    identity_count = len(
        {identity for split in manifest.splits.values() for identity in split.identities}
    )
    if identity_count < 8:
        classification = "pipeline-pilot"
    else:
        geometry_supported = all(
            bootstrap[name]["ci95_high"] < 0.0
            for name in ("canonical_3d_nme", "reprojection_pixels")
        )
        critical = [
            name
            for name in (
                "pose_rotation_degrees",
                "pose_translation_mae",
                "temporal_jitter",
                "peak_attenuation",
                "recovery_ms",
            )
            if name in bootstrap
        ]
        regression = any(bootstrap[name]["ci95_low"] > 0.0 for name in critical)
        classification = (
            "stage-a-regressed"
            if regression
            else "stage-a-supported"
            if geometry_supported
            else "stage-a-inconclusive"
        )
    report: dict[str, Any] = {
        "schema_version": "nana-stage-a-comparison/1.0.0",
        "usage_tier": "commercial",
        "commercial_release_evidence": False,
        "crema_d_used": False,
        "classification": classification,
        "identity_count": identity_count,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": config.training.seed,
            "cluster": "identity",
            "metrics": bootstrap,
        },
        "single_view": {
            "checkpoint_sha256": sha256_file(single_view_checkpoint),
            "metadata": baseline_metadata,
            "per_identity": baseline,
        },
        "multiview": {
            "checkpoint_sha256": sha256_file(multiview_checkpoint),
            "metadata": candidate_metadata,
            "per_identity": candidate,
        },
        "data_revision": config.reproducibility.data_revision,
        "manifest_digest": config.reproducibility.manifest_digest,
        "anchor_mapping_digest": config.reproducibility.anchor_mapping_digest,
        "training_recipe_digest": config.reproducibility.training_recipe_digest,
        "config_digest": sha256_json(config.model_dump(mode="json")),
        "limitations": (
            "Commercial first-party Stage A evidence only. No rig/confidence claim or release "
            "evidence is implied; Stage B/C, NanaLive A/B, ONNX parity, and target hardware "
            "acceptance remain required before Issue #7 can close."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
