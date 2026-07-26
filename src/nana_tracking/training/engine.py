"""Minimal deterministic training engine used by real registries and smoke fixtures."""

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from nana_tracking.config import ExperimentConfig, save_config
from nana_tracking.contracts import CheckpointMetadata, MultiViewTrackingBatch, TrackingBatch
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
from nana_tracking.training.stage_a import (
    compute_stage_a_loss_components,
    learning_rate_at_step,
    stage_a_parameters,
)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    checkpoint: Path
    best_checkpoint: Path
    summary_report: Path
    final_step: int
    final_loss: float


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_loss_components(
    config: ExperimentConfig,
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    label_confidence: dict[str, Tensor],
    images: Tensor,
    model: nn.Module,
    multiview_batch: MultiViewTrackingBatch | None = None,
) -> dict[str, Tensor]:
    if config.training.stage == "real-geometry-pretrain":
        if multiview_batch is None:
            raise ValueError("Stage A loss requires a MultiViewTrackingBatch")
        return compute_stage_a_loss_components(config, outputs, multiview_batch)
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


def _resume_compatibility_digest(config: ExperimentConfig) -> str:
    payload = config.model_dump(mode="json")
    if config.training.stage != "real-geometry-pretrain":
        payload["training"].pop("max_steps", None)
    payload["training"].pop("validation_interval_steps", None)
    payload["training"].pop("checkpoint_interval_steps", None)
    payload["reproducibility"].pop("output_dir", None)
    return sha256_json(payload)


def _write_metrics(path: Path, metrics: dict[str, float | int | str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def _evaluate_loss_components(
    config: ExperimentConfig,
    model: nn.Module,
    loader: DataLoader[TrackingBatch],
    device: torch.device,
    *,
    amp_enabled: bool,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    sample_count = 0
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, device)
            images = moved.images
            targets = moved.targets
            label_confidence = moved.label_confidence
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = _forward_outputs(config, model, images)
                components = compute_loss_components(
                    config,
                    outputs,
                    targets,
                    label_confidence,
                    images,
                    model,
                    moved if isinstance(moved, MultiViewTrackingBatch) else None,
                )
            batch_size = images.shape[0]
            sample_count += batch_size
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
    if was_training:
        model.train()
    if sample_count == 0:
        raise RuntimeError("validation loader produced no batches")
    averaged = {name: total / sample_count for name, total in totals.items()}
    averaged["loss"] = sum(averaged.values())
    return averaged


def _move_batch(
    batch: TrackingBatch | MultiViewTrackingBatch,
    device: torch.device,
) -> TrackingBatch | MultiViewTrackingBatch:
    targets = {name: value.to(device) for name, value in batch.targets.items()}
    confidence = {name: value.to(device) for name, value in batch.label_confidence.items()}
    if isinstance(batch, MultiViewTrackingBatch):
        return MultiViewTrackingBatch(
            images=batch.images.to(device),
            targets=targets,
            label_confidence=confidence,
            camera_intrinsics=batch.camera_intrinsics.to(device),
            camera_to_capture=batch.camera_to_capture.to(device),
            sample_ids=batch.sample_ids,
            identity_ids=batch.identity_ids,
            sequence_ids=batch.sequence_ids,
            expressions=batch.expressions,
            timestamps_ns=batch.timestamps_ns.to(device),
        )
    return TrackingBatch(
        images=batch.images.to(device),
        targets=targets,
        label_confidence=confidence,
        sample_ids=batch.sample_ids,
    )


def _forward_outputs(
    config: ExperimentConfig,
    model: nn.Module,
    images: Tensor,
) -> dict[str, Tensor]:
    if images.ndim == 5:
        groups, views, channels, height, width = images.shape
        flat = images.reshape(groups * views, channels, height, width)
        values = model(flat)
        return {
            name: value.reshape(groups, views, *value.shape[1:])
            for name, value in zip(output_names(config.model), values, strict=True)
        }
    return dict(zip(output_names(config.model), model(images), strict=True))


def _best_logged_validation(metrics_path: Path) -> float:
    if not metrics_path.is_file():
        return math.inf
    best = math.inf
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        value = payload.get("validation/loss")
        if isinstance(value, int | float):
            best = min(best, float(value))
    return best


def _checkpoint_metadata(
    config: ExperimentConfig,
    *,
    run_id: str,
    step: int,
    batches_per_epoch: int,
    device: torch.device,
    amp_enabled: bool,
    git_commit: str,
    git_dirty: bool,
    lock_digest: str,
    parent_checkpoint_digest: str | None,
) -> CheckpointMetadata:
    consumed_batches = step * config.training.gradient_accumulation_steps
    return CheckpointMetadata(
        run_id=run_id,
        epoch=consumed_batches // batches_per_epoch,
        batch_in_epoch=consumed_batches % batches_per_epoch,
        step=step,
        seed=config.training.seed,
        config_digest=sha256_json(config.model_dump(mode="json")),
        resume_compatibility_digest=_resume_compatibility_digest(config),
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
        usage_tier=config.data.usage_tier,
        training_stage=config.training.stage,
        manifest_digest=config.reproducibility.manifest_digest,
        license_registry_digest=config.reproducibility.license_registry_digest,
        anchor_mapping_digest=config.reproducibility.anchor_mapping_digest,
        training_recipe_digest=config.reproducibility.training_recipe_digest,
        parent_checkpoint_digest=parent_checkpoint_digest,
    )


def _training_summary(
    config: ExperimentConfig,
    *,
    run_id: str,
    metrics_path: Path,
    checkpoint: Path,
    best_checkpoint: Path,
    output: Path,
) -> None:
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_losses = [
        float(event["train/loss"])
        for event in events
        if isinstance(event.get("train/loss"), int | float)
    ]
    validation_losses = [
        float(event["validation/loss"])
        for event in events
        if isinstance(event.get("validation/loss"), int | float)
    ]
    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    reduction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    payload: dict[str, Any] = {
        "schema_version": "nana-training-summary/1.0.0",
        "smoke_only": config.export.smoke_only,
        "usage_tier": config.data.usage_tier,
        "training_stage": config.training.stage,
        "crema_d_used": False,
        "run_id": run_id,
        "data_revision": config.reproducibility.data_revision,
        "ntp_schema_revision": config.reproducibility.ntp_schema_revision,
        "signal_registry_revision": config.reproducibility.signal_registry_revision,
        "normalization_revision": config.reproducibility.normalization_revision,
        "calibration_revision": config.reproducibility.calibration_revision,
        "feature_revision": config.reproducibility.feature_revision,
        "config_digest": sha256_json(config.model_dump(mode="json")),
        "training": {
            "steps": len(train_losses),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_fraction": reduction,
            "finite": all(math.isfinite(value) for value in train_losses),
        },
        "validation": {
            "evaluations": len(validation_losses),
            "best_loss": min(validation_losses) if validation_losses else None,
        },
        "checkpoints": {
            "last": str(checkpoint),
            "last_sha256": sha256_file(checkpoint),
            "best": str(best_checkpoint),
            "best_sha256": sha256_file(best_checkpoint),
        },
        "limitations": (
            "Synthetic smoke evidence only. This report validates training control flow and "
            "reproducibility; it does not establish real FaceBasic quality or production "
            "readiness."
            if config.data.usage_tier == "synthetic-smoke"
            else (
                "Commercial development evidence only. Independent locked-test, NanaLive A/B, "
                "ONNX parity, target-hardware, license, and release gates remain required."
            )
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train(
    config: ExperimentConfig,
    *,
    resume: Path | None = None,
    repository_root: Path | None = None,
) -> TrainingResult:
    _verify_frozen_capture_input(config)
    seed_everything(config.training.seed, deterministic=config.training.deterministic)
    device = choose_device(config.training.device)
    if config.training.amp and device.type != "cuda":
        raise RuntimeError("AMP is currently supported only for CUDA training")
    amp_enabled = config.training.amp and device.type == "cuda"
    model = create_model(config.model).to(device)
    optimizer_parameters = (
        stage_a_parameters(model)
        if config.training.stage == "real-geometry-pretrain"
        else list(model.parameters())
    )
    optimizer = (
        torch.optim.AdamW(
            optimizer_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        if config.training.optimizer == "adamw"
        else torch.optim.Adam(
            optimizer_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    )
    scaler = torch.GradScaler("cuda", enabled=amp_enabled)

    if resume is None:
        run_id = new_run_id()
        run_dir = config.reproducibility.output_dir / run_id
        start_step = 0
        parent_checkpoint_digest = None
    else:
        run_dir = resume.parent.parent
        restored = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            restore_rng=True,
            expected_usage_tier=config.data.usage_tier,
        )
        expected_resume_digest = _resume_compatibility_digest(config)
        if (
            restored.resume_compatibility_digest
            and restored.resume_compatibility_digest != expected_resume_digest
        ):
            raise ValueError("resume configuration is incompatible with the checkpoint")
        if restored.training_stage != config.training.stage:
            raise ValueError("checkpoint training stage does not match the configuration")
        run_id = restored.run_id
        start_step = restored.step
        parent_checkpoint_digest = sha256_file(resume)

    run_dir.mkdir(parents=True, exist_ok=True)
    if resume is None:
        save_config(config, run_dir / "config.resolved.yaml")
    else:
        save_config(config, run_dir / f"config.resume-{start_step}.yaml")
    metrics_path = run_dir / "metrics.jsonl"
    batches_per_epoch = len(
        create_loader(
            config,
            split="train",
            shuffle=config.training.shuffle,
            seed_offset=0,
        )
    )
    if batches_per_epoch == 0:
        raise RuntimeError("training loader produced no batches")
    step = start_step
    final_loss = float("nan")
    model.train()
    root = repository_root or Path.cwd()
    git_commit, git_dirty = git_state(root)
    lock_path = root / "uv.lock"
    lock_digest = sha256_file(lock_path) if lock_path.exists() else "missing"
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    best_validation_loss = _best_logged_validation(metrics_path)
    accumulation_steps = config.training.gradient_accumulation_steps
    consumed_batches = start_step * accumulation_steps
    start_epoch = consumed_batches // batches_per_epoch
    first_batch_in_epoch = consumed_batches % batches_per_epoch
    epoch = start_epoch
    validation_loader = (
        create_loader(
            config,
            split="validation",
            shuffle=False,
            seed_offset=10_000,
        )
        if config.training.validation_interval_steps is not None
        else None
    )

    def write_checkpoint(path: Path) -> None:
        metadata = _checkpoint_metadata(
            config,
            run_id=run_id,
            step=step,
            batches_per_epoch=batches_per_epoch,
            device=device,
            amp_enabled=amp_enabled,
            git_commit=git_commit,
            git_dirty=git_dirty,
            lock_digest=lock_digest,
            parent_checkpoint_digest=parent_checkpoint_digest,
        )
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            metadata=metadata,
        )

    optimizer.zero_grad(set_to_none=True)
    accumulated_batches = 0
    while step < config.training.max_steps:
        loader = create_loader(
            config,
            split="train",
            shuffle=config.training.shuffle,
            seed_offset=0,
            shuffle_seed_offset=epoch,
        )
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < first_batch_in_epoch:
                continue
            if step >= config.training.max_steps:
                break
            moved = _move_batch(batch, device)
            images = moved.images
            targets = moved.targets
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = _forward_outputs(config, model, images)
                label_confidence = moved.label_confidence
                components = compute_loss_components(
                    config,
                    outputs,
                    targets,
                    label_confidence,
                    images,
                    model,
                    moved if isinstance(moved, MultiViewTrackingBatch) else None,
                )
                loss = torch.stack(tuple(components.values())).sum()
            scaler.scale(loss / accumulation_steps).backward()
            accumulated_batches += 1
            if accumulated_batches < accumulation_steps:
                continue
            current_lr = (
                learning_rate_at_step(config, step)
                if config.training.stage == "real-geometry-pretrain"
                else config.training.learning_rate
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_batches = 0
            step += 1
            final_loss = float(loss.detach().cpu())
            metrics: dict[str, float | int | str] = {
                "event": "train",
                "step": step,
                "epoch": epoch,
                "batch_in_epoch": batch_index + 1,
                "train/loss": final_loss,
                "train/learning_rate": current_lr,
            }
            metrics.update(
                {f"train/{name}": float(value.detach().cpu()) for name, value in components.items()}
            )
            _write_metrics(metrics_path, metrics)

            validation_interval = config.training.validation_interval_steps
            if validation_interval is not None and step % validation_interval == 0:
                assert validation_loader is not None
                validation = _evaluate_loss_components(
                    config,
                    model,
                    validation_loader,
                    device,
                    amp_enabled=amp_enabled,
                )
                validation_event: dict[str, float | int | str] = {
                    "event": "validation",
                    "step": step,
                }
                validation_event.update(
                    {f"validation/{name}": value for name, value in validation.items()}
                )
                _write_metrics(metrics_path, validation_event)
                if validation["loss"] < best_validation_loss:
                    best_validation_loss = validation["loss"]
                    write_checkpoint(best_checkpoint)

            checkpoint_interval = config.training.checkpoint_interval_steps
            if checkpoint_interval is not None and step % checkpoint_interval == 0:
                write_checkpoint(run_dir / "checkpoints" / f"step-{step:08d}.pt")
        epoch += 1
        first_batch_in_epoch = 0

    checkpoint = run_dir / "checkpoints" / "last.pt"
    write_checkpoint(checkpoint)
    if not best_checkpoint.is_file():
        write_checkpoint(best_checkpoint)
    summary_report = run_dir / "training-summary.json"
    _training_summary(
        config,
        run_id=run_id,
        metrics_path=metrics_path,
        checkpoint=checkpoint,
        best_checkpoint=best_checkpoint,
        output=summary_report,
    )
    return TrainingResult(
        run_dir,
        checkpoint,
        best_checkpoint,
        summary_report,
        step,
        final_loss,
    )


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
