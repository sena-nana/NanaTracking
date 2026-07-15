"""Per-head evaluation with machine-readable reports."""

import json
from pathlib import Path

import torch

from nana_tracking.config import ExperimentConfig
from nana_tracking.data.loaders import create_loader
from nana_tracking.models import create_model, output_names
from nana_tracking.reproducibility import choose_device
from nana_tracking.training.checkpoint import load_checkpoint


def evaluate(
    config: ExperimentConfig,
    checkpoint: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    device = choose_device(config.training.device)
    model = create_model(config.model).to(device)
    load_checkpoint(checkpoint, model=model)
    model.eval()
    loader = create_loader(config, split="validation", shuffle=False, seed_offset=10_000)
    names = output_names(config.model)
    comparable = [
        name for name in names if name not in {"visibility", "tongue_visibility", "identity"}
    ]
    errors: dict[str, list[torch.Tensor]] = {name: [] for name in comparable}

    with torch.inference_mode():
        for batch in loader:
            outputs = dict(zip(names, model(batch.images.to(device)), strict=True))
            for name in comparable:
                prediction = outputs[name]
                target = batch.targets[name].to(device)
                errors[name].append((prediction - target).detach().abs().cpu())

    report: dict[str, dict[str, float]] = {}
    for name, chunks in errors.items():
        joined = torch.cat(chunks)
        report[name] = {
            "mae": float(joined.mean()),
            "mse": float(joined.square().mean()),
            "max_abs": float(joined.max()),
        }

    destination = output_path or checkpoint.parent.parent / "evaluation.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return report
