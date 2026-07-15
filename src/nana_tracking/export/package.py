"""Portable ONNX package creation and verification."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import AdapterContract, ModelPackageMetadata
from nana_tracking.data.manifest import DatasetManifest
from nana_tracking.models import (
    create_deployment_model,
    create_model,
    deployment_output_names,
)
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


def _operator_contract(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    operators: set[str] = set()
    custom_domains: set[str] = set()

    def visit(nodes: Iterable[onnx.NodeProto]) -> None:
        for node in nodes:
            domain = node.domain or "ai.onnx"
            operators.add(f"{domain}::{node.op_type}")
            if node.domain not in {"", "ai.onnx"}:
                custom_domains.add(node.domain)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g.node)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested_graph in attribute.graphs:
                        visit(nested_graph.node)

    visit(model.graph.node)
    for function in model.functions:
        visit(function.node)
    return sorted(operators), sorted(custom_domains)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ort_outputs(
    model_path: Path,
    input_array: np.ndarray,
    names: tuple[str, ...],
) -> list[np.ndarray]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    values = session.run(list(names), {"image": input_array})
    return [cast(np.ndarray, value) for value in values]


def create_model_package(
    config: ExperimentConfig,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    if not config.export.smoke_only:
        if config.data.dataset == "synthetic" or config.data.manifest is None:
            raise ValueError("non-smoke packages require a reviewed manifest dataset")
        if DatasetManifest.load(config.data.manifest).smoke_only:
            raise ValueError("a smoke-only manifest cannot produce a non-smoke package")
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output_dir / "test-vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    training_model = create_model(config.model)
    load_checkpoint(checkpoint, model=training_model)
    model = create_deployment_model(config.model, training_model)
    model.eval()
    names = deployment_output_names(config.model)
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
        output_names=list(names),
        opset_version=config.export.opset,
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    onnx.checker.check_model(onnx_path)
    graph = onnx.load(onnx_path, load_external_data=False)
    required_operators, custom_domains = _operator_contract(graph)
    if custom_domains:
        raise ValueError(f"export contains undeclared custom operator domains: {custom_domains}")
    if not required_operators:
        raise ValueError("exported ONNX graph contains no executable operators")

    with torch.inference_mode():
        eager_values = [value.detach().cpu().numpy() for value in model(example)]
    input_array = example.numpy()
    runtime_values = _ort_outputs(onnx_path, input_array, names)
    parity: dict[str, dict[str, float]] = {}
    for name, eager, runtime in zip(names, eager_values, runtime_values, strict=True):
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
        **dict(zip(names, eager_values, strict=True)),
    )
    _write_json(vector_dir / "parity.json", parity)
    is_basic = config.model.name == "face_basic"
    is_spatial = config.model.name == "face_spatial"
    is_full = config.model.name == "full_set"
    if is_basic:
        guaranteed_profile = "Basic"
        schema_version = "face-basic-1"
        output_roles = {
            "rig": "ntp-basic-36",
            "pose": "ntp-head-camera-pose",
            "landmarks": "auxiliary-training-diagnostic",
            "visibility": "runtime-state-classification",
            "confidence": "per-signal-confidence",
        }
        supported_signals = list(range(1, 37))
        supported_structures = ["head_geometry"]
        geometry_topology_revision = None
    elif is_spatial:
        guaranteed_profile = "Spatial"
        schema_version = "face-spatial-1"
        output_roles = {
            "rig": "ntp-spatial-41",
            "pose": "ntp-head-camera-pose",
            "eye_origins": "ntp-head-relative-eye-origins",
            "eye_directions": "ntp-continuous-eye-directions",
            "look_at_head": "head-relative-look-at-before-pose-transform",
            "face_geometry": config.reproducibility.geometry_topology_revision,
            "visibility": "runtime-state-classification",
            "tongue_visibility": "tongue-observation-state-classification",
            "confidence": "per-signal-confidence",
        }
        supported_signals = list(range(1, 42))
        supported_structures = [
            "head_geometry",
            "eye_geometry",
            "look_at_point",
            "face_geometry",
        ]
        geometry_topology_revision = config.reproducibility.geometry_topology_revision
    elif is_full:
        guaranteed_profile = "Partial"
        schema_version = "full-set-extension-1"
        output_roles = {
            "rig": "ntp-full-signals-42-through-76",
            "torso_pose": "ntp-camera-relative-torso-pose",
            "joint_positions": "ntp-torso-local-shoulder-elbow-wrist",
            "joint_rotations": "ntp-torso-local-shoulder-elbow-wrist-xyzw",
            "limb_directions": "ntp-torso-local-upper-arm-and-forearm-directions",
            "limb_twists": "ntp-upper-arm-and-forearm-twist",
            "bone_lengths": "normalized-upper-arm-and-forearm-lengths",
            "visibility": "internal-region-state-classification",
            "confidence": "per-full-signal-confidence",
        }
        supported_signals = list(range(42, 77))
        supported_structures = ["body_skeleton"]
        geometry_topology_revision = None
    else:
        guaranteed_profile = "Partial"
        schema_version = "smoke-1"
        output_roles = {
            "rig": "smoke-rig",
            "pose": "smoke-pose",
            "confidence": "smoke-confidence",
        }
        supported_signals = []
        supported_structures = []
        geometry_topology_revision = None
    _write_json(
        output_dir / "schema.json",
        {
            "schema_version": schema_version,
            "input": {"name": "image", "shape": list(shape), "dtype": "float32"},
            "outputs": list(names),
            "output_roles": output_roles,
            "dynamic_dimensions": [],
            "required_operators": required_operators,
            "custom_operator_domains": custom_domains,
            "smoke_only": config.export.smoke_only,
        },
    )
    _write_json(
        output_dir / "signal-registry-revision.json",
        {"revision": config.reproducibility.signal_registry_revision},
    )
    _write_json(
        output_dir / "normalization.json",
        {
            "revision": config.reproducibility.normalization_revision,
            "input_range": [0.0, 1.0],
            "layout": "NCHW",
            "color": "RGB",
        },
    )
    _write_json(
        output_dir / "calibration-schema.json",
        {
            "schema_version": "1",
            "revision": config.reproducibility.calibration_revision,
            "default_path": "level-a",
            "smoke_only": config.export.smoke_only,
        },
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
        normalization_revision=config.reproducibility.normalization_revision,
        calibration_revision=config.reproducibility.calibration_revision,
        feature_revision=config.reproducibility.feature_revision,
        onnx_opset=config.export.opset,
        input_shape=list(shape),
        output_names=list(names),
        model_digest=sha256_file(onnx_path),
        smoke_only=config.export.smoke_only,
        precision_support=["fp32"],
        guaranteed_profile=guaranteed_profile,
        supported_signals=supported_signals,
        supported_structures=supported_structures,
        supported_features=[],
        temporal_state="external-causal-refiner/1.0.0-compatible",
        allowed_backends=["onnxruntime"],
        runtime_modes={
            "Performance": {
                "precision": "fp32",
                "scheduling": "latest-frame-only",
            },
            "Quality": {
                "precision": "fp32",
                "scheduling": "latest-frame-only-with-causal-refiner",
            },
        },
        dynamic_dimensions=[],
        required_operators=required_operators,
        custom_operator_domains=custom_domains,
        geometry_topology_revision=geometry_topology_revision,
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
    onnx.checker.check_model(package_dir / "model.onnx")
    graph = onnx.load(package_dir / "model.onnx", load_external_data=False)
    actual_operators, actual_custom_domains = _operator_contract(graph)
    if actual_custom_domains or metadata.custom_operator_domains:
        raise ValueError("custom ONNX operator domains are not allowed in portable packages")
    if not metadata.required_operators or actual_operators != metadata.required_operators:
        raise ValueError("ONNX operator metadata does not match the packaged graph")
    with np.load(package_dir / "test-vectors" / "input.npz") as input_file:
        image = input_file["image"]
    names = tuple(metadata.output_names)
    with np.load(package_dir / "test-vectors" / "expected.npz") as expected_file:
        expected = {name: expected_file[name] for name in names}
    actual = _ort_outputs(package_dir / "model.onnx", image, names)
    report: dict[str, dict[str, float]] = {}
    for name, runtime in zip(names, actual, strict=True):
        difference = np.abs(runtime - expected[name])
        report[name] = {
            "mae": float(difference.mean()),
            "max_abs": float(difference.max(initial=0.0)),
        }
        np.testing.assert_allclose(runtime, expected[name], atol=atol, rtol=rtol)
    return report
