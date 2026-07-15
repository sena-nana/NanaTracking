"""Minimal deterministic training engine used by real registries and smoke fixtures."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from nana_tracking.config import ExperimentConfig, save_config
from nana_tracking.contracts import CheckpointMetadata
from nana_tracking.data.capture import FrozenCaptureDataset
from nana_tracking.data.loaders import create_loader
from nana_tracking.data.manifest import DatasetManifest, SplitManifest
from nana_tracking.models import (
    create_model,
    mirror_basic_rig,
    mirror_full_rig,
    mirror_spatial_rig,
    output_names,
)
from nana_tracking.reproducibility import (
    choose_device,
    git_state,
    new_run_id,
    seed_everything,
    sha256_file,
    sha256_json,
)
from nana_tracking.training.checkpoint import load_checkpoint, save_checkpoint


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    checkpoint: Path
    final_step: int
    final_loss: float


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _losses(
    config: ExperimentConfig,
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    label_confidence: dict[str, Tensor],
    images: Tensor,
    model: nn.Module,
) -> dict[str, Tensor]:
    if config.model.name == "smoke":
        return {
            "rig": nn.functional.mse_loss(outputs["rig"], targets["rig"]),
            "pose": nn.functional.mse_loss(outputs["pose"], targets["pose"]),
            "confidence": nn.functional.binary_cross_entropy(
                outputs["confidence"], targets["confidence"]
            ),
        }
    raw = nn.functional.smooth_l1_loss(outputs["rig"], targets["rig"], reduction="none")
    pose_name = "torso_pose" if config.model.name == "full_set" else "pose"
    pose = nn.functional.smooth_l1_loss(outputs[pose_name], targets[pose_name], reduction="none")
    if config.model.name == "face_basic":
        geometry_names = ("landmarks",)
    elif config.model.name == "face_spatial":
        geometry_names = ("eye_origins", "eye_directions", "look_at_head", "face_geometry")
    else:
        geometry_names = (
            "joint_positions",
            "joint_rotations",
            "limb_directions",
            "limb_twists",
            "bone_lengths",
        )
    mirrored_outputs = dict(
        zip(output_names(config.model), model(torch.flip(images, dims=(-1,))), strict=True)
    )
    mirror_consistency = nn.functional.smooth_l1_loss(
        mirrored_outputs["rig"],
        mirror_basic_rig(outputs["rig"])
        if config.model.name == "face_basic"
        else mirror_spatial_rig(outputs["rig"])
        if config.model.name == "face_spatial"
        else mirror_full_rig(outputs["rig"]),
    )
    losses = {
        "rig": _weighted_mean(raw, label_confidence["rig"]) * config.training.rig_loss_weight,
        "pose": _weighted_mean(pose, label_confidence[pose_name])
        * config.training.pose_loss_weight,
        "visibility": (
            nn.functional.cross_entropy(
                outputs["visibility"].flatten(0, 1), targets["visibility"].flatten()
            )
            if config.model.name == "full_set"
            else nn.functional.cross_entropy(outputs["visibility"], targets["visibility"])
        )
        * config.training.visibility_loss_weight,
        "identity_adversary": nn.functional.cross_entropy(outputs["identity"], targets["identity"])
        * config.training.identity_adversary_weight,
        "confidence": nn.functional.binary_cross_entropy(
            outputs["confidence"], targets["confidence"]
        )
        * config.training.confidence_loss_weight,
        "mirror_consistency": mirror_consistency * config.training.mirror_consistency_weight,
    }
    for name in geometry_names:
        error = nn.functional.smooth_l1_loss(outputs[name], targets[name], reduction="none")
        weight = (
            config.training.face_geometry_loss_weight
            if name == "face_geometry"
            else config.training.eye_geometry_loss_weight
            if name in {"eye_origins", "eye_directions", "look_at_head"}
            else config.training.landmark_loss_weight
        )
        losses[name] = _weighted_mean(error, label_confidence[name]) * weight
    if config.model.name == "face_spatial":
        tongue_error = nn.functional.cross_entropy(
            outputs["tongue_visibility"], targets["tongue_visibility"], reduction="none"
        )
        losses["tongue_visibility"] = (
            _weighted_mean(tongue_error, label_confidence["tongue_visibility"].squeeze(-1))
            * config.training.tongue_visibility_loss_weight
        )
    return losses


def train(
    config: ExperimentConfig,
    *,
    resume: Path | None = None,
    repository_root: Path | None = None,
) -> TrainingResult:
    _verify_frozen_capture_input(config)
    seed_everything(config.training.seed)
    device = choose_device(config.training.device)
    if config.training.amp and device.type != "cuda":
        raise RuntimeError("AMP is currently supported only for CUDA training")
    amp_enabled = config.training.amp and device.type == "cuda"
    model = create_model(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scaler = torch.GradScaler("cuda", enabled=amp_enabled)

    if resume is None:
        run_id = new_run_id()
        run_dir = config.reproducibility.output_dir / run_id
        start_step = 0
    else:
        run_dir = resume.parent.parent
        restored = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            restore_rng=True,
        )
        run_id = restored.run_id
        start_step = restored.step

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.resolved.yaml")
    metrics_path = run_dir / "metrics.jsonl"
    loader = create_loader(config, split="train", shuffle=False)
    step = start_step
    final_loss = float("nan")
    model.train()

    while step < config.training.max_steps:
        made_progress = False
        for batch in loader:
            if step >= config.training.max_steps:
                break
            made_progress = True
            images = batch.images.to(device)
            targets = {name: value.to(device) for name, value in batch.targets.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = dict(zip(output_names(config.model), model(images), strict=True))
                label_confidence = {
                    name: value.to(device) for name, value in batch.label_confidence.items()
                }
                components = _losses(config, outputs, targets, label_confidence, images, model)
                loss = torch.stack(tuple(components.values())).sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            final_loss = float(loss.detach().cpu())
            with metrics_path.open("a", encoding="utf-8") as handle:
                metrics = {"step": step, "train/loss": final_loss}
                metrics.update(
                    {
                        f"train/{name}": float(value.detach().cpu())
                        for name, value in components.items()
                    }
                )
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        if not made_progress:
            raise RuntimeError("training loader produced no batches")

    root = repository_root or Path.cwd()
    git_commit, git_dirty = git_state(root)
    lock_path = root / "uv.lock"
    lock_digest = sha256_file(lock_path) if lock_path.exists() else "missing"
    metadata = CheckpointMetadata(
        run_id=run_id,
        epoch=max(0, (step - 1) // max(1, len(loader))),
        step=step,
        seed=config.training.seed,
        config_digest=sha256_json(config.model_dump(mode="json")),
        data_revision=config.reproducibility.data_revision,
        ntp_schema_revision=config.reproducibility.ntp_schema_revision,
        signal_registry_revision=config.reproducibility.signal_registry_revision,
        normalization_revision=config.reproducibility.normalization_revision,
        calibration_revision=config.reproducibility.calibration_revision,
        feature_revision=config.reproducibility.feature_revision,
        device=str(device),
        amp_enabled=amp_enabled,
        git_commit=git_commit,
        git_dirty=git_dirty,
        lock_digest=lock_digest,
        created_at=datetime.now(UTC),
    )
    checkpoint = run_dir / "checkpoints" / "last.pt"
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, metadata=metadata)
    return TrainingResult(run_dir, checkpoint, step, final_loss)


def _verify_frozen_capture_input(config: ExperimentConfig) -> None:
    frozen_path = config.data.frozen_capture
    if frozen_path is None:
        return
    manifest_path = config.data.manifest
    if manifest_path is None:
        raise ValueError("frozen capture training requires a DatasetManifest")
    frozen_path = frozen_path.resolve()
    manifest_path = manifest_path.resolve()
    frozen = FrozenCaptureDataset.load(frozen_path)
    frozen.verify(frozen_path)
    manifest = DatasetManifest.load(manifest_path)
    manifest.verify_files(manifest_path)
    if frozen.data_revision != config.reproducibility.data_revision:
        raise ValueError("training data revision does not match the frozen capture dataset")
    if manifest.data_revision != frozen.data_revision:
        raise ValueError("training manifest does not expose the frozen data revision")
    if manifest.smoke_only != frozen.smoke_only or config.export.smoke_only != frozen.smoke_only:
        raise ValueError("training, manifest, and frozen capture smoke status must match")
    if len(manifest.record_files) != 1 or (
        manifest.record_files[0].sha256 != frozen.capture_records.sha256
        or manifest.record_files[0].record_count != frozen.capture_records.record_count
    ):
        raise ValueError("training manifest records do not match the frozen capture records")
    expected_splits = {
        name: SplitManifest.model_validate(split.model_dump(mode="json"))
        for name, split in frozen.split_plan.splits.items()
    }
    if manifest.splits != expected_splits:
        raise ValueError("training manifest splits do not match the frozen capture splits")
    if manifest.license_record_ids != frozen.license_record_ids:
        raise ValueError("training manifest licenses do not match the frozen capture dataset")
    if {source.version for source in manifest.teacher_sources} != set(frozen.ntp_mapping_revisions):
        raise ValueError("training manifest teacher versions do not match frozen mappings")
    expected_revisions = {
        "ntp_schema_revision": config.reproducibility.ntp_schema_revision,
        "signal_registry_revision": config.reproducibility.signal_registry_revision,
        "normalization_revision": config.reproducibility.normalization_revision,
        "calibration_revision": config.reproducibility.calibration_revision,
        "feature_revision": config.reproducibility.feature_revision,
    }
    for field, expected in expected_revisions.items():
        if getattr(manifest, field) != expected:
            raise ValueError(f"training configuration {field} does not match the manifest")
