"""Minimal deterministic training engine used by real registries and smoke fixtures."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from nana_tracking.config import ExperimentConfig, save_config
from nana_tracking.contracts import CheckpointMetadata, TrackingModelOutput
from nana_tracking.data.synthetic import create_loader
from nana_tracking.models import create_model
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


def _loss(output: TrackingModelOutput, targets: dict[str, Tensor]) -> Tensor:
    return (
        nn.functional.mse_loss(output.rig, targets["rig"])
        + nn.functional.mse_loss(output.pose, targets["pose"])
        + nn.functional.binary_cross_entropy(output.confidence, targets["confidence"])
    )


def train(
    config: ExperimentConfig,
    *,
    resume: Path | None = None,
    repository_root: Path | None = None,
) -> TrainingResult:
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
    loader = create_loader(config, shuffle=False)
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
                output = TrackingModelOutput.from_tuple(model(images))
                loss = _loss(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            final_loss = float(loss.detach().cpu())
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "train/loss": final_loss}) + "\n")
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
