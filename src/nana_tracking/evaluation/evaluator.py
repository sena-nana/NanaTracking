"""Per-head evaluation with machine-readable reports."""

import json
import math
from pathlib import Path
from typing import Literal

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
    split: Literal["validation", "test"] = "validation",
) -> dict[str, dict[str, object]]:
    device = choose_device(config.training.device)
    model = create_model(config.model).to(device)
    load_checkpoint(checkpoint, model=model)
    model.eval()
    seed_offset = 10_000 if split == "validation" else 20_000
    loader = create_loader(config, split=split, shuffle=False, seed_offset=seed_offset)
    names = output_names(config.model)
    comparable = [
        name for name in names if name not in {"visibility", "tongue_visibility", "identity"}
    ]
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in comparable}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in comparable}
    classifications = [name for name in names if name in {"visibility", "tongue_visibility"}]
    classification_correct: dict[str, int] = {name: 0 for name in classifications}
    classification_count: dict[str, int] = {name: 0 for name in classifications}

    with torch.inference_mode():
        for batch in loader:
            outputs = dict(zip(names, model(batch.images.to(device)), strict=True))
            for name in comparable:
                prediction = outputs[name]
                target = batch.targets[name].to(device)
                weight = batch.label_confidence[name].to(device)
                mask = weight > 0
                predictions[name].append(prediction.detach()[mask].cpu())
                targets[name].append(target.detach()[mask].cpu())
            for name in classifications:
                prediction = outputs[name]
                target = batch.targets[name].to(device)
                predicted_class = prediction.argmax(dim=-1)
                mask = batch.label_confidence[name].to(device).squeeze(-1) > 0
                classification_correct[name] += int(
                    (predicted_class[mask] == target[mask]).sum().cpu()
                )
                classification_count[name] += int(mask.sum().cpu())

    report: dict[str, dict[str, object]] = {}
    for name in comparable:
        joined_prediction = torch.cat(predictions[name])
        joined_target = torch.cat(targets[name])
        if joined_target.numel() == 0:
            report[name] = {
                "status": "unavailable",
                "sample_count": 0,
                "reason": "no reliable labels are available for this output family",
            }
            continue
        joined = (joined_prediction - joined_target).abs()
        prediction_mean = joined_prediction.mean()
        target_mean = joined_target.mean()
        covariance = ((joined_prediction - prediction_mean) * (joined_target - target_mean)).mean()
        denominator = (
            joined_prediction.var(unbiased=False)
            + joined_target.var(unbiased=False)
            + (prediction_mean - target_mean).square()
        )
        ccc = (
            1.0
            if float(denominator) == 0.0 and torch.equal(joined_prediction, joined_target)
            else 0.0
            if float(denominator) == 0.0
            else float((2.0 * covariance / denominator).cpu())
        )
        mse = float(joined.square().mean())
        report[name] = {
            "status": "measured",
            "sample_count": joined.numel(),
            "mae": float(joined.mean()),
            "mse": mse,
            "rmse": math.sqrt(mse),
            "max_abs": float(joined.max()),
            "ccc": ccc,
        }
    for name in classifications:
        count = classification_count[name]
        report[name] = (
            {
                "status": "measured",
                "sample_count": count,
                "accuracy": classification_correct[name] / count,
            }
            if count
            else {
                "status": "unavailable",
                "sample_count": 0,
                "reason": "no reliable labels are available for this output family",
            }
        )

    filename = "evaluation.jsonl" if split == "validation" else f"evaluation-{split}.jsonl"
    destination = output_path or checkpoint.parent.parent / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return report
