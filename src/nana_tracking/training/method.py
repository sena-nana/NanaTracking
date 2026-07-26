"""Executable smoke-only checks that the training method learns and resumes exactly."""

import hashlib
import io
import json
import math
from pathlib import Path

import torch

from nana_tracking.config import ExperimentConfig
from nana_tracking.data.loaders import create_loader
from nana_tracking.evaluation import evaluate
from nana_tracking.models import create_model, output_names
from nana_tracking.reproducibility import choose_device, seed_everything, sha256_file
from nana_tracking.training.checkpoint import load_checkpoint
from nana_tracking.training.engine import compute_loss_components, train


def _training_state_digest(path: Path) -> str:
    payload: dict[str, object] = torch.load(path, map_location="cpu", weights_only=False)
    state = {
        name: payload.get(name)
        for name in (
            "model",
            "optimizer",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "scaler",
        )
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getbuffer()).hexdigest()


def _gradient_audit(config: ExperimentConfig) -> dict[str, object]:
    seed_everything(config.training.seed, deterministic=True)
    device = choose_device(config.training.device)
    model = create_model(config.model).to(device)
    batch = next(
        iter(
            create_loader(
                config,
                split="train",
                shuffle=False,
                seed_offset=0,
            )
        )
    )
    images = batch.images.to(device)
    targets = {name: value.to(device) for name, value in batch.targets.items()}
    confidence = {name: value.to(device) for name, value in batch.label_confidence.items()}
    outputs = dict(zip(output_names(config.model), model(images), strict=True))
    components = compute_loss_components(
        config,
        outputs,
        targets,
        confidence,
        images,
        model,
    )
    torch.stack(tuple(components.values())).sum().backward()
    groups: dict[str, dict[str, bool | int]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.split(".", maxsplit=1)[0]
        status = groups.setdefault(
            group,
            {"parameter_count": 0, "all_finite": True, "any_nonzero": False},
        )
        status["parameter_count"] = int(status["parameter_count"]) + 1
        gradient = parameter.grad
        status["all_finite"] = bool(status["all_finite"]) and (
            gradient is not None and bool(torch.isfinite(gradient).all())
        )
        status["any_nonzero"] = bool(status["any_nonzero"]) or (
            gradient is not None and bool(torch.count_nonzero(gradient))
        )
    inactive_groups: set[str] = (
        {"canonical_geometry_head"}
        if config.training.canonical_geometry_loss_weight == 0.0
        else set()
    )
    audited = {name: status for name, status in groups.items() if name not in inactive_groups}
    passed = bool(audited) and all(
        bool(status["all_finite"]) and bool(status["any_nonzero"]) for status in audited.values()
    )
    return {
        "passed": passed,
        "inactive_parameter_groups": sorted(inactive_groups),
        "parameter_groups": groups,
    }


def _probe_config(
    config: ExperimentConfig,
    *,
    output_dir: Path,
    max_steps: int,
) -> ExperimentConfig:
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "max_steps": max_steps,
                    "device": "cpu",
                    "amp": False,
                    "deterministic": True,
                    "validation_interval_steps": None,
                    "checkpoint_interval_steps": None,
                }
            ),
            "reproducibility": config.reproducibility.model_copy(update={"output_dir": output_dir}),
        }
    )


def _train_losses(metrics_path: Path) -> list[float]:
    losses: list[float] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        value = event.get("train/loss")
        if isinstance(value, int | float):
            losses.append(float(value))
    return losses


def validate_training_method(
    config: ExperimentConfig,
    output: Path,
    *,
    probe_steps: int = 40,
    minimum_loss_reduction: float = 0.3,
) -> dict[str, object]:
    """Run repeat, resume, gradient, and held-out smoke checks and write one report."""

    if not config.export.smoke_only:
        raise ValueError("training-method validation is smoke-only and cannot certify production")
    if probe_steps < 4:
        raise ValueError("training-method validation requires at least four probe steps")
    if not 0.0 < minimum_loss_reduction < 1.0:
        raise ValueError("minimum loss reduction must be between zero and one")

    work_dir = output.parent / "method-probes"
    direct_config = _probe_config(
        config,
        output_dir=work_dir / "direct",
        max_steps=probe_steps,
    )
    repeat_config = _probe_config(
        config,
        output_dir=work_dir / "repeat",
        max_steps=probe_steps,
    )
    staged_config = _probe_config(
        config,
        output_dir=work_dir / "resumed",
        max_steps=probe_steps // 2,
    )
    resumed_config = _probe_config(
        config,
        output_dir=work_dir / "resumed",
        max_steps=probe_steps,
    )

    direct = train(direct_config)
    repeat = train(repeat_config)
    staged = train(staged_config)
    resumed = train(resumed_config, resume=staged.checkpoint)

    direct_state = _training_state_digest(direct.checkpoint)
    repeat_state = _training_state_digest(repeat.checkpoint)
    resumed_state = _training_state_digest(resumed.checkpoint)
    direct_metrics = (direct.run_dir / "metrics.jsonl").read_bytes()
    repeat_metrics = (repeat.run_dir / "metrics.jsonl").read_bytes()
    resumed_metrics = (resumed.run_dir / "metrics.jsonl").read_bytes()
    deterministic = direct_state == repeat_state and direct_metrics == repeat_metrics
    resume_equivalent = direct_state == resumed_state and direct_metrics == resumed_metrics

    losses = _train_losses(direct.run_dir / "metrics.jsonl")
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    learning_passed = (
        all(math.isfinite(value) for value in losses) and reduction >= minimum_loss_reduction
    )
    gradients = _gradient_audit(direct_config)
    validation = evaluate(
        direct_config,
        direct.best_checkpoint,
        output_path=work_dir / "validation.json",
        split="validation",
    )
    test = evaluate(
        direct_config,
        direct.best_checkpoint,
        output_path=work_dir / "test.json",
        split="test",
    )
    checkpoint_model = create_model(direct_config.model)
    checkpoint_metadata = load_checkpoint(
        direct.checkpoint,
        model=checkpoint_model,
        expected_usage_tier=direct_config.data.usage_tier,
    )
    passed = deterministic and resume_equivalent and learning_passed and bool(gradients["passed"])
    report: dict[str, object] = {
        "schema_version": "nana-training-method-report/1.0.0",
        "passed": passed,
        "smoke_only": True,
        "crema_d_used": False,
        "probe": {
            "steps": probe_steps,
            "minimum_loss_reduction": minimum_loss_reduction,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_fraction": reduction,
            "learning_passed": learning_passed,
        },
        "determinism": {
            "passed": deterministic,
            "direct_state_sha256": direct_state,
            "repeat_state_sha256": repeat_state,
            "metrics_identical": direct_metrics == repeat_metrics,
        },
        "resume": {
            "passed": resume_equivalent,
            "uninterrupted_state_sha256": direct_state,
            "resumed_state_sha256": resumed_state,
            "metrics_identical": direct_metrics == resumed_metrics,
        },
        "gradients": gradients,
        "validation": validation,
        "test": test,
        "provenance": {
            "checkpoint": str(direct.checkpoint),
            "checkpoint_sha256": sha256_file(direct.checkpoint),
            "data_revision": checkpoint_metadata.data_revision,
            "config_digest": checkpoint_metadata.config_digest,
            "ntp_schema_revision": checkpoint_metadata.ntp_schema_revision,
            "signal_registry_revision": checkpoint_metadata.signal_registry_revision,
            "git_commit": checkpoint_metadata.git_commit,
            "git_dirty": checkpoint_metadata.git_dirty,
            "lock_digest": checkpoint_metadata.lock_digest,
            "device": checkpoint_metadata.device,
            "amp_enabled": checkpoint_metadata.amp_enabled,
        },
        "limitations": (
            "Repository-owned synthetic smoke data only. Passing this report proves deterministic "
            "learning and resume behavior, not real-camera accuracy, demographic quality, latency, "
            "or production readiness."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
