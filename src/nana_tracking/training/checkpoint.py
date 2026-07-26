"""Trusted local training checkpoint persistence."""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from nana_tracking.contracts import CheckpointMetadata
from nana_tracking.governance import ArtifactUsageTier


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.GradScaler | None,
    metadata: CheckpointMetadata,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
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
    scaler: torch.GradScaler | None = None,
    restore_rng: bool = False,
    expected_usage_tier: ArtifactUsageTier | None = None,
) -> CheckpointMetadata:
    """Load a checkpoint created by this project; never use with untrusted files."""

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    metadata = CheckpointMetadata.model_validate(payload["metadata"])
    if expected_usage_tier is not None and metadata.usage_tier != expected_usage_tier:
        raise ValueError(
            "checkpoint usage tier mismatch: "
            f"expected {expected_usage_tier}, got {metadata.usage_tier}"
        )
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        random.setstate(payload["python_rng"])
        if payload.get("numpy_rng") is not None:
            np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        cuda_rng = payload.get("cuda_rng")
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
    return metadata
