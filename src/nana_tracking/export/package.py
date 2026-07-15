"""Portable ONNX package creation and verification."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import torch

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import (
    MODEL_OUTPUT_NAMES,
    AdapterContract,
    ModelPackageMetadata,
)
from nana_tracking.models import create_model
from nana_tracking.reproducibility import sha256_file
from nana_tracking.training.checkpoint import load_checkpoint

REQUIRED_PACKAGE_PATHS = (
    "model.onnx",
    "schema.json",
    "signal-registry-revision.json",
    "normalization.json",
    "runtime-metadata.json",
    "calibration-schema.json",
    "adapter-contract.json",
    "test-vectors/input.npz",
    "test-vectors/expected.npz",
    "test-vectors/parity.json",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ort_outputs(model_path: Path, input_array: np.ndarray) -> list[np.ndarray]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    values = session.run(list(MODEL_OUTPUT_NAMES), {"image": input_array})
    return [cast(np.ndarray, value) for value in values]


def create_model_package(
    config: ExperimentConfig,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output_dir / "test-vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(config.model)
    load_checkpoint(checkpoint, model=model)
    model.eval()
    shape = (
        1,
        config.model.input_channels,
        config.model.input_height,
        config.model.input_width,
    )
    generator = torch.Generator().manual_seed(config.training.seed + 50_000)
    example = torch.rand(shape, generator=generator)
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        (example,),
        onnx_path,
        input_names=["image"],
        output_names=list(MODEL_OUTPUT_NAMES),
        opset_version=config.export.opset,
        dynamo=True,
    )

    with torch.inference_mode():
        eager_values = [value.detach().cpu().numpy() for value in model(example)]
    input_array = example.numpy()
    runtime_values = _ort_outputs(onnx_path, input_array)
    parity: dict[str, dict[str, float]] = {}
    for name, eager, runtime in zip(MODEL_OUTPUT_NAMES, eager_values, runtime_values, strict=True):
        difference = np.abs(eager - runtime)
        parity[name] = {
            "mae": float(difference.mean()),
            "max_abs": float(difference.max(initial=0.0)),
        }
        np.testing.assert_allclose(
            runtime,
            eager,
            atol=config.evaluation.atol,
            rtol=config.evaluation.rtol,
        )

    np.savez(vector_dir / "input.npz", image=input_array)
    np.savez(
        vector_dir / "expected.npz",
        **dict(zip(MODEL_OUTPUT_NAMES, eager_values, strict=True)),
    )
    _write_json(vector_dir / "parity.json", parity)
    _write_json(
        output_dir / "schema.json",
        {
            "schema_version": "smoke-1",
            "input": {"name": "image", "shape": list(shape), "dtype": "float32"},
            "outputs": list(MODEL_OUTPUT_NAMES),
            "smoke_only": True,
        },
    )
    _write_json(
        output_dir / "signal-registry-revision.json",
        {"revision": config.reproducibility.signal_registry_revision},
    )
    _write_json(output_dir / "normalization.json", {"input_range": [0.0, 1.0]})
    _write_json(
        output_dir / "calibration-schema.json",
        {"schema_version": "1", "default_path": "level-a", "smoke_only": True},
    )
    adapter = AdapterContract(
        adapter_type="affine-residual",
        base_model_family=config.export.model_family,
        base_model_version=config.export.model_version,
        feature_revision=config.reproducibility.feature_revision,
    )
    _write_json(output_dir / "adapter-contract.json", adapter.model_dump(mode="json"))
    metadata = ModelPackageMetadata(
        model_family=config.export.model_family,
        model_version=config.export.model_version,
        source_checkpoint_digest=sha256_file(checkpoint),
        ntp_schema_revision=config.reproducibility.ntp_schema_revision,
        signal_registry_revision=config.reproducibility.signal_registry_revision,
        feature_revision=config.reproducibility.feature_revision,
        onnx_opset=config.export.opset,
        input_shape=list(shape),
        output_names=list(MODEL_OUTPUT_NAMES),
        model_digest=sha256_file(onnx_path),
    )
    _write_json(output_dir / "runtime-metadata.json", metadata.model_dump(mode="json"))
    return parity


def verify_model_package(
    package_dir: Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, dict[str, float]]:
    missing = [
        relative for relative in REQUIRED_PACKAGE_PATHS if not (package_dir / relative).is_file()
    ]
    if missing:
        raise ValueError(f"model package is missing required files: {missing}")
    metadata = ModelPackageMetadata.model_validate_json(
        (package_dir / "runtime-metadata.json").read_text(encoding="utf-8")
    )
    actual_digest = sha256_file(package_dir / "model.onnx")
    if actual_digest != metadata.model_digest:
        raise ValueError("model.onnx digest does not match runtime metadata")
    with np.load(package_dir / "test-vectors" / "input.npz") as input_file:
        image = input_file["image"]
    with np.load(package_dir / "test-vectors" / "expected.npz") as expected_file:
        expected = {name: expected_file[name] for name in MODEL_OUTPUT_NAMES}
    actual = _ort_outputs(package_dir / "model.onnx", image)
    report: dict[str, dict[str, float]] = {}
    for name, runtime in zip(MODEL_OUTPUT_NAMES, actual, strict=True):
        difference = np.abs(runtime - expected[name])
        report[name] = {
            "mae": float(difference.mean()),
            "max_abs": float(difference.max(initial=0.0)),
        }
        np.testing.assert_allclose(runtime, expected[name], atol=atol, rtol=rtol)
    return report
