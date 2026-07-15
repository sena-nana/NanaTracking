"""Trusted local training checkpoint persistence."""

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from nana_tracking.contracts import CheckpointMetadata


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: CheckpointMetadata,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "metadata": metadata.model_dump(mode="json"),
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    restore_rng: bool = False,
) -> CheckpointMetadata:
    """Load a checkpoint created by this project; never use with untrusted files."""

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
    return CheckpointMetadata.model_validate(payload["metadata"])
