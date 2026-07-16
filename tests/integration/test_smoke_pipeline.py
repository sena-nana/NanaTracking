import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nana_tracking.cli import app
from nana_tracking.config import load_config
from nana_tracking.evaluation import evaluate
from nana_tracking.evaluation.runtime import (
    benchmark_face_basic_package,
    benchmark_face_spatial_package,
    benchmark_full_set_package,
)
from nana_tracking.export import create_model_package, verify_model_package
from nana_tracking.training import train


@pytest.mark.integration
def test_train_evaluate_export_verify(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    config = config.model_copy(
        update={
            "reproducibility": config.reproducibility.model_copy(update={"output_dir": tmp_path}),
            "training": config.training.model_copy(update={"max_steps": 1}),
        }
    )
    result = train(config)
    metrics = evaluate(config, result.checkpoint)
    assert set(metrics) == {"rig", "pose", "confidence"}
    package = tmp_path / "model-package"
    parity = create_model_package(config, result.checkpoint, package)
    verified = verify_model_package(package)
    assert parity["rig"]["max_abs"] <= config.evaluation.atol
    assert verified["pose"]["max_abs"] <= config.evaluation.atol


@pytest.mark.integration
def test_face_basic_train_evaluate_export_verify(tmp_path: Path) -> None:
    config = load_config(Path("configs/face-basic-smoke.yaml"))
    config = config.model_copy(
        update={
            "reproducibility": config.reproducibility.model_copy(update={"output_dir": tmp_path}),
            "training": config.training.model_copy(update={"max_steps": 1}),
        }
    )
    result = train(config)
    metrics = evaluate(config, result.checkpoint)
    assert set(metrics) == {"rig", "pose", "landmarks", "confidence"}
    package = tmp_path / "face-basic-package"
    parity = create_model_package(config, result.checkpoint, package)
    verified = verify_model_package(package)
    assert set(parity) == {
        "rig",
        "pose",
        "landmarks",
        "visibility",
        "confidence",
    }
    assert verified["rig"]["max_abs"] <= config.evaluation.atol
    rust_consumer = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "nana-tracking-runtime-api",
            "--example",
            "verify-package",
            "--",
            str(package),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rust_consumer.returncode == 0, rust_consumer.stderr or rust_consumer.stdout
    benchmark = benchmark_face_basic_package(
        package,
        tmp_path / "runtime-benchmark.json",
        providers=["CPUExecutionProvider"],
        warmup=1,
        iterations=2,
    )
    assert benchmark["smoke_only"] is True
    runtime = benchmark["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["active_providers"] == ["CPUExecutionProvider"]
    stability_path = tmp_path / "runtime-stability.json"
    command = CliRunner().invoke(
        app,
        [
            "benchmark-face-stability",
            "--package",
            str(package),
            "--output",
            str(stability_path),
            "--duration-seconds",
            "0.25",
            "--target-fps",
            "20",
            "--resource-interval-seconds",
            "0.1",
            "--warmup",
            "1",
        ],
    )
    assert command.exit_code == 0, command.output
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    assert stability["smoke_only"] is True
    bounded = stability["bounded_sampling"]
    assert isinstance(bounded, dict)
    assert bounded["retained_samples_including_windows"] <= 65_536 + 8_192
    result = stability["stability"]
    assert isinstance(result, dict)
    assert result["passed"] is True


@pytest.mark.integration
def test_face_spatial_train_evaluate_export_verify(tmp_path: Path) -> None:
    config = load_config(Path("configs/face-spatial-smoke.yaml"))
    config = config.model_copy(
        update={
            "reproducibility": config.reproducibility.model_copy(update={"output_dir": tmp_path}),
            "training": config.training.model_copy(update={"max_steps": 1}),
        }
    )
    result = train(config)
    metrics = evaluate(config, result.checkpoint)
    assert set(metrics) == {
        "rig",
        "pose",
        "eye_origins",
        "eye_directions",
        "look_at_head",
        "face_geometry",
        "confidence",
    }
    package = tmp_path / "face-spatial-package"
    parity = create_model_package(config, result.checkpoint, package)
    verified = verify_model_package(package)
    assert set(parity) == {
        "rig",
        "pose",
        "eye_origins",
        "eye_directions",
        "look_at_head",
        "face_geometry",
        "visibility",
        "tongue_visibility",
        "confidence",
    }
    assert verified["rig"]["max_abs"] <= config.evaluation.atol
    benchmark = benchmark_face_spatial_package(
        package,
        tmp_path / "spatial-runtime-benchmark.json",
        providers=["CPUExecutionProvider"],
        warmup=1,
        iterations=2,
    )
    assert benchmark["schema_version"] == "face-spatial-runtime-benchmark/1.0.0"
    assert benchmark["geometry_topology_revision"] == "ntp-face-canonical/1.0.0-smoke"


@pytest.mark.integration
def test_full_set_train_evaluate_export_verify(tmp_path: Path) -> None:
    config = load_config(Path("configs/full-set-smoke.yaml"))
    config = config.model_copy(
        update={
            "reproducibility": config.reproducibility.model_copy(update={"output_dir": tmp_path}),
            "training": config.training.model_copy(update={"max_steps": 1}),
        }
    )
    result = train(config)
    metrics = evaluate(config, result.checkpoint)
    assert set(metrics) == {
        "rig",
        "torso_pose",
        "joint_positions",
        "joint_rotations",
        "limb_directions",
        "limb_twists",
        "bone_lengths",
        "confidence",
    }
    package = tmp_path / "full-set-package"
    parity = create_model_package(config, result.checkpoint, package)
    verified = verify_model_package(package)
    assert set(parity) == {
        "rig",
        "torso_pose",
        "joint_positions",
        "joint_rotations",
        "limb_directions",
        "limb_twists",
        "bone_lengths",
        "visibility",
        "confidence",
    }
    assert verified["rig"]["max_abs"] <= config.evaluation.atol
    benchmark = benchmark_full_set_package(
        package,
        tmp_path / "full-set-runtime-benchmark.json",
        providers=["CPUExecutionProvider"],
        warmup=1,
        iterations=2,
    )
    assert benchmark["schema_version"] == "full-set-upper-body-runtime-benchmark/1.0.0"
    assert benchmark["smoke_only"] is True
