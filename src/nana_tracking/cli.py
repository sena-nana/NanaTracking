"""Command-line entrypoint for reproducible project workflows."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from nana_tracking.config import ExperimentConfig, load_config
from nana_tracking.data.executors import benchmark_backends
from nana_tracking.data.manifest import DatasetManifest
from nana_tracking.doctor import doctor_report
from nana_tracking.evaluation import evaluate as evaluate_model
from nana_tracking.export import create_model_package, verify_model_package
from nana_tracking.training import train as train_model

app = typer.Typer(no_args_is_help=True, help="NanaTracking training and ONNX tooling.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset manifest commands.")
app.add_typer(data_app, name="data")
console = Console()


def _print_json(payload: object) -> None:
    console.print_json(json.dumps(payload, default=str))


@app.command()
def doctor() -> None:
    """Report Python 3.14, GIL/JIT, accelerator, and ORT provider state."""

    _print_json(doctor_report())


@data_app.command("validate")
def validate_data(manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Validate schema fields and identity-safe dataset splits."""

    validated = DatasetManifest.load(manifest)
    _print_json(validated.model_dump(mode="json"))


@app.command("train")
def train_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    resume: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Train a registered model and save a reproducible checkpoint."""

    result = train_model(load_config(config), resume=resume)
    _print_json(
        {
            "run_dir": result.run_dir,
            "checkpoint": result.checkpoint,
            "final_step": result.final_step,
            "final_loss": result.final_loss,
        }
    )


@app.command("evaluate")
def evaluate_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Evaluate a checkpoint and write per-head metrics."""

    _print_json(evaluate_model(load_config(config), checkpoint, output_path=output))


@app.command("export")
def export_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Export and parity-check a complete ONNX model package."""

    _print_json(create_model_package(load_config(config), checkpoint, output))


@app.command("verify-export")
def verify_export_command(
    package: Annotated[Path, typer.Option("--package", exists=True, file_okay=False)],
) -> None:
    """Verify package contents, digest, and ORT fixed-vector parity."""

    _print_json(verify_model_package(package))


@app.command("smoke")
def smoke_command(
    work_dir: Annotated[Path, typer.Option("--work-dir", file_okay=False)] = Path("runs/smoke"),
) -> None:
    """Run the synthetic train/resume/evaluate/export/verify pipeline."""

    config = ExperimentConfig.model_validate(
        {
            "data": {"samples": 8, "batch_size": 4},
            "model": {"name": "smoke"},
            "training": {"seed": 7, "max_steps": 1, "device": "cpu"},
            "evaluation": {"atol": 1e-5, "rtol": 1e-4},
            "export": {"model_family": "nana-smoke", "model_version": "0.0.0-smoke"},
            "reproducibility": {
                "output_dir": work_dir / "runs",
                "data_revision": "synthetic-v1",
                "ntp_schema_revision": "smoke-ntp-v0",
                "signal_registry_revision": "smoke-signals-v0",
                "feature_revision": "smoke-features-v1",
            },
        }
    )
    initial = train_model(config)
    resumed_config = config.model_copy(
        update={"training": config.training.model_copy(update={"max_steps": 2})}
    )
    resumed = train_model(resumed_config, resume=initial.checkpoint)
    evaluation = evaluate_model(resumed_config, resumed.checkpoint)
    package_dir = work_dir / "model-package"
    parity = create_model_package(resumed_config, resumed.checkpoint, package_dir)
    verification = verify_model_package(package_dir)
    _print_json(
        {
            "checkpoint": resumed.checkpoint,
            "final_step": resumed.final_step,
            "evaluation": evaluation,
            "export_parity": parity,
            "package_verification": verification,
            "package": package_dir,
            "warning": "Synthetic smoke evidence is not face-tracking acceptance evidence.",
        }
    )


@app.command("benchmark-python")
def benchmark_python_command(
    output: Annotated[Path, typer.Option("--output")] = Path(
        "artifacts/benchmarks/python-executors.json"
    ),
    items: Annotated[int, typer.Option(min=1)] = 32,
    rounds: Annotated[int, typer.Option(min=1)] = 5_000,
    workers: Annotated[int, typer.Option(min=1)] = 2,
    buffersize: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    """Compare bounded Python 3.14 executor throughput without setting a pass threshold."""

    report = benchmark_backends(
        items=items,
        rounds=rounds,
        workers=workers,
        buffersize=buffersize,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json({"output": output, "results": report})


def main() -> None:
    app()


if __name__ == "__main__":
    main()
