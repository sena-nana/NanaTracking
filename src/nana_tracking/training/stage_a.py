"""Stage A real-geometry losses and optimizer isolation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import MultiViewTrackingBatch


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert normalized xyzw quaternions to rotation matrices."""

    q = nn.functional.normalize(quaternion, dim=-1, eps=1e-6)
    x, y, z, w = q.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotation_geodesic(rotation_a: Tensor, rotation_b: Tensor) -> Tensor:
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0 + 1e-6,
        1.0 - 1e-6,
    )
    return torch.acos(cosine)


def quaternion_geodesic(quaternion_a: Tensor, quaternion_b: Tensor) -> Tensor:
    left = nn.functional.normalize(quaternion_a, dim=-1, eps=1e-6)
    right = nn.functional.normalize(quaternion_b, dim=-1, eps=1e-6)
    cosine = (left * right).sum(dim=-1).abs().clamp(max=1.0)
    cosine = torch.where(cosine > 1.0 - 1e-6, torch.ones_like(cosine), cosine)
    return 2.0 * torch.acos(cosine)


def _pairwise(values: Tensor) -> Iterable[tuple[Tensor, Tensor]]:
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            yield values[:, left], values[:, right]


def _pairwise_smooth_l1(values: Tensor) -> Tensor:
    losses = [nn.functional.smooth_l1_loss(left, right) for left, right in _pairwise(values)]
    if not losses:
        return values.sum() * 0.0
    return torch.stack(losses).mean()


def _project_geometry(
    geometry: Tensor,
    target_pose: Tensor,
    intrinsics: Tensor,
    *,
    width: int,
    height: int,
) -> Tensor:
    rotation = quaternion_to_matrix(target_pose[..., 3:])
    camera_points = torch.einsum("bvij,bvnj->bvni", rotation, geometry)
    camera_points = camera_points + target_pose[..., None, :3]
    pixels_h = torch.einsum("bvij,bvnj->bvni", intrinsics, camera_points)
    depth = pixels_h[..., 2:].clamp_min(1e-4)
    pixels = pixels_h[..., :2] / depth
    normalized_x = pixels[..., 0] / width * 2.0 - 1.0
    normalized_y = pixels[..., 1] / height * 2.0 - 1.0
    return torch.stack((normalized_x, normalized_y), dim=-1)


def _capture_pose(pose: Tensor, camera_to_capture: Tensor) -> tuple[Tensor, Tensor]:
    camera_rotation = camera_to_capture[..., :3, :3]
    camera_translation = camera_to_capture[..., :3, 3]
    head_rotation = quaternion_to_matrix(pose[..., 3:])
    capture_rotation = camera_rotation @ head_rotation
    capture_translation = (
        torch.einsum("bvij,bvj->bvi", camera_rotation, pose[..., :3]) + camera_translation
    )
    return capture_translation, capture_rotation


def compute_stage_a_loss_components(
    config: ExperimentConfig,
    outputs: dict[str, Tensor],
    batch: MultiViewTrackingBatch,
) -> dict[str, Tensor]:
    if config.training.stage != "real-geometry-pretrain":
        raise ValueError("Stage A loss requires training.stage=real-geometry-pretrain")
    targets = batch.targets
    weights = batch.label_confidence
    landmark_error = nn.functional.smooth_l1_loss(
        outputs["landmarks"], targets["landmarks"], reduction="none"
    )
    geometry_error = nn.functional.smooth_l1_loss(
        outputs["canonical_geometry"],
        targets["canonical_geometry"],
        reduction="none",
    )
    translation_error = nn.functional.smooth_l1_loss(
        outputs["pose"][..., :3],
        targets["pose"][..., :3],
        reduction="none",
    )
    rotation_error = quaternion_geodesic(
        outputs["pose"][..., 3:],
        targets["pose"][..., 3:],
    )
    rotation_weight = weights["pose"][..., 3:].mean(dim=-1)
    projected = _project_geometry(
        outputs["canonical_geometry"],
        targets["pose"],
        batch.camera_intrinsics,
        width=config.model.input_width,
        height=config.model.input_height,
    )
    reprojection_error = nn.functional.smooth_l1_loss(
        projected,
        targets["landmarks"],
        reduction="none",
    )

    capture_translation, capture_rotation = _capture_pose(outputs["pose"], batch.camera_to_capture)
    pose_pair_losses: list[Tensor] = []
    for (left_t, right_t), (left_r, right_r) in zip(
        _pairwise(capture_translation),
        _pairwise(capture_rotation),
        strict=True,
    ):
        pose_pair_losses.append(
            nn.functional.smooth_l1_loss(left_t, right_t)
            + rotation_geodesic(left_r, right_r).mean()
        )
    multiview_pose = (
        torch.stack(pose_pair_losses).mean() if pose_pair_losses else outputs["pose"].sum() * 0.0
    )

    identity_targets = targets["identity"].reshape(-1)
    identity_logits = outputs["identity"].reshape(-1, outputs["identity"].shape[-1])
    identity_mask = identity_targets >= 0
    identity_loss = (
        nn.functional.cross_entropy(
            identity_logits[identity_mask],
            identity_targets[identity_mask],
        )
        if identity_mask.any()
        else identity_logits.sum() * 0.0
    )
    losses = {
        "landmarks": _weighted_mean(landmark_error, weights["landmarks"])
        * config.training.landmark_loss_weight,
        "canonical_geometry": _weighted_mean(geometry_error, weights["canonical_geometry"])
        * config.training.canonical_geometry_loss_weight,
        "pose_rotation": _weighted_mean(rotation_error, rotation_weight)
        * config.training.pose_rotation_loss_weight,
        "pose_translation": _weighted_mean(translation_error, weights["pose"][..., :3])
        * config.training.pose_translation_loss_weight,
        "reprojection": _weighted_mean(reprojection_error, weights["landmarks"])
        * config.training.reprojection_loss_weight,
        "multiview_geometry": _pairwise_smooth_l1(outputs["canonical_geometry"])
        * config.training.multiview_geometry_loss_weight,
        "multiview_pose": multiview_pose * config.training.multiview_pose_loss_weight,
        "visibility": nn.functional.cross_entropy(
            outputs["visibility"].reshape(-1, outputs["visibility"].shape[-1]),
            targets["visibility"].reshape(-1),
        )
        * config.training.visibility_loss_weight,
        "identity_adversary": identity_loss * config.training.identity_adversary_weight,
    }
    if any(not torch.isfinite(value).all() for value in losses.values()):
        raise FloatingPointError("Stage A produced a non-finite loss component")
    return losses


def stage_a_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Exclude rig/confidence heads structurally, including AdamW decay."""

    excluded = ("rig_head.", "confidence_head.")
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(excluded) and parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Stage A optimizer has no trainable parameters")
    return parameters


def learning_rate_at_step(config: ExperimentConfig, step: int) -> float:
    base = config.training.learning_rate
    minimum = config.training.minimum_learning_rate
    warmup = config.training.warmup_steps
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    remaining = max(config.training.max_steps - warmup, 1)
    progress = min(max((step - warmup) / remaining, 0.0), 1.0)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))
