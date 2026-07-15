"""Offline Level B residual adapter training, packaging, and compatibility."""

import json
from pathlib import Path
from typing import Self, cast

import numpy as np
import onnxruntime as ort
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from nana_tracking.contracts import AdapterContract
from nana_tracking.reproducibility import sha256_file


class AffineResidualAdapter(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(Tensor(features).fill_(1.0))
        self.offset = nn.Parameter(Tensor(features).zero_())

    def forward(self, values: Tensor) -> Tensor:
        return values + (values * (self.scale - 1.0) + self.offset)


class _ResidualDeployment(nn.Module):
    def __init__(self, adapter: AffineResidualAdapter) -> None:
        super().__init__()
        self.adapter = adapter

    def forward(self, values: Tensor) -> Tensor:
        return self.adapter(values) - values


class AdapterPackageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "ntp-level-b-adapter-package/1.0.0"
    user_slot: str = Field(min_length=1)
    adapter_type: str = "affine-residual"
    base_model_family: str = Field(min_length=1)
    base_model_version: str = Field(min_length=1)
    base_model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    calibration_revision: str = Field(min_length=1)
    signal_ids: list[int]
    output_mode: str = "residual"
    adapter_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_data_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    smoke_only: bool = True

    @model_validator(mode="after")
    def validate_signal_order(self) -> Self:
        if not self.signal_ids or self.signal_ids != sorted(set(self.signal_ids)):
            raise ValueError("adapter Signal IDs must be unique and increasing")
        return self


def _sha256_arrays(values: dict[str, np.ndarray]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def train_level_b_adapter(
    base_values: np.ndarray,
    target_values: np.ndarray,
    confidence: np.ndarray,
    output_dir: Path,
    *,
    user_slot: str,
    base_model_family: str,
    base_model_version: str,
    base_model_digest: str,
    feature_revision: str,
    signal_registry_revision: str,
    normalization_revision: str,
    calibration_revision: str,
    signal_ids: list[int],
    seed: int = 7,
    steps: int = 200,
    learning_rate: float = 0.01,
    smoke_only: bool = True,
) -> AdapterPackageMetadata:
    """Train only bounded affine residual parameters; the base encoder is never updated."""

    if base_values.shape != target_values.shape or base_values.shape != confidence.shape:
        raise ValueError("adapter arrays must have identical [samples, signals] shapes")
    if base_values.ndim != 2 or base_values.shape[1] != len(signal_ids):
        raise ValueError("adapter signal_ids must match the training columns")
    if len(base_values) < 8 or steps < 1 or learning_rate <= 0.0:
        raise ValueError("adapter training requires bounded positive settings and enough samples")
    if signal_ids != sorted(set(signal_ids)):
        raise ValueError("adapter signal_ids must be unique and increasing")
    arrays = {"base": base_values, "target": target_values, "confidence": confidence}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("adapter evidence must be finite")
    if not ((confidence >= 0.0).all() and (confidence <= 1.0).all()):
        raise ValueError("adapter confidence must be in [0, 1]")
    torch.manual_seed(seed)
    base = torch.from_numpy(base_values.astype(np.float32, copy=False))
    target = torch.from_numpy(target_values.astype(np.float32, copy=False))
    weights = torch.from_numpy(confidence.astype(np.float32, copy=False))
    adapter = AffineResidualAdapter(len(signal_ids))
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    adapter.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = adapter(base)
        error = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
        data_loss = (error * weights).sum() / weights.sum().clamp_min(1.0)
        regularization = (adapter.scale - 1.0).square().mean() + adapter.offset.square().mean()
        loss = data_loss + 0.001 * regularization
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            adapter.scale.clamp_(0.75, 1.25)
            adapter.offset.clamp_(-0.25, 0.25)
    deployment = _ResidualDeployment(adapter.eval()).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = output_dir / "test-vectors"
    vectors.mkdir(parents=True, exist_ok=True)
    example = base[:1]
    adapter_path = output_dir / "adapter.onnx"
    torch.onnx.export(
        deployment,
        (example,),
        adapter_path,
        input_names=["base_values"],
        output_names=["residual"],
        opset_version=18,
        dynamo=True,
        verbose=False,
    )
    with torch.inference_mode():
        expected = deployment(example).numpy()
    session = ort.InferenceSession(str(adapter_path), providers=["CPUExecutionProvider"])
    actual = cast(list[np.ndarray], session.run(["residual"], {"base_values": example.numpy()}))[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    np.savez(vectors / "input.npz", base_values=example.numpy())
    np.savez(vectors / "expected.npz", residual=expected)
    metadata = AdapterPackageMetadata(
        user_slot=user_slot,
        base_model_family=base_model_family,
        base_model_version=base_model_version,
        base_model_digest=base_model_digest,
        feature_revision=feature_revision,
        signal_registry_revision=signal_registry_revision,
        normalization_revision=normalization_revision,
        calibration_revision=calibration_revision,
        signal_ids=signal_ids,
        adapter_digest=sha256_file(adapter_path),
        source_data_digest=_sha256_arrays(arrays),
        smoke_only=smoke_only,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def verify_level_b_adapter(
    package_dir: Path,
    *,
    user_slot: str,
    base_model_family: str,
    base_model_version: str,
    base_model_digest: str,
    feature_revision: str,
) -> dict[str, float]:
    required = [
        package_dir / "adapter.onnx",
        package_dir / "metadata.json",
        package_dir / "test-vectors" / "input.npz",
        package_dir / "test-vectors" / "expected.npz",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise ValueError(f"adapter package is incomplete: {missing}")
    metadata = AdapterPackageMetadata.model_validate_json(
        (package_dir / "metadata.json").read_text(encoding="utf-8")
    )
    expected_contract = (
        user_slot,
        base_model_family,
        base_model_version,
        base_model_digest,
        feature_revision,
    )
    actual_contract = (
        metadata.user_slot,
        metadata.base_model_family,
        metadata.base_model_version,
        metadata.base_model_digest,
        metadata.feature_revision,
    )
    if actual_contract != expected_contract:
        raise ValueError("adapter is incompatible with the active user or base model contract")
    if sha256_file(package_dir / "adapter.onnx") != metadata.adapter_digest:
        raise ValueError("adapter ONNX digest does not match metadata")
    with np.load(package_dir / "test-vectors" / "input.npz") as inputs:
        base = inputs["base_values"]
    with np.load(package_dir / "test-vectors" / "expected.npz") as expected_file:
        expected = expected_file["residual"]
    session = ort.InferenceSession(
        str(package_dir / "adapter.onnx"), providers=["CPUExecutionProvider"]
    )
    actual = cast(list[np.ndarray], session.run(["residual"], {"base_values": base}))[0]
    difference = np.abs(actual - expected)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    return {"mae": float(difference.mean()), "max_abs": float(difference.max(initial=0.0))}


class OrtLevelBAdapter:
    """User-bound ONNX residual adapter returning only framework-neutral values."""

    def __init__(
        self,
        package_dir: Path,
        *,
        user_slot: str,
        base_model_family: str,
        base_model_version: str,
        base_model_digest: str,
        feature_revision: str,
    ) -> None:
        verify_level_b_adapter(
            package_dir,
            user_slot=user_slot,
            base_model_family=base_model_family,
            base_model_version=base_model_version,
            base_model_digest=base_model_digest,
            feature_revision=feature_revision,
        )
        self.metadata = AdapterPackageMetadata.model_validate_json(
            (package_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self._session = ort.InferenceSession(
            str(package_dir / "adapter.onnx"), providers=["CPUExecutionProvider"]
        )

    def apply(self, base_values: tuple[float, ...]) -> tuple[float, ...]:
        if len(base_values) != len(self.metadata.signal_ids):
            raise ValueError("base values do not match the adapter Signal IDs")
        values = np.asarray([base_values], dtype=np.float32)
        residual = cast(list[np.ndarray], self._session.run(["residual"], {"base_values": values}))[
            0
        ]
        adapted = values + residual
        if not np.isfinite(adapted).all():
            raise ValueError("adapter produced non-finite values")
        return tuple(float(value) for value in adapted[0])


def ensure_adapter_compatible(
    contract: AdapterContract,
    *,
    model_family: str,
    model_version: str,
    feature_revision: str,
) -> None:
    expected = (model_family, model_version, feature_revision)
    actual = (
        contract.base_model_family,
        contract.base_model_version,
        contract.feature_revision,
    )
    if actual != expected:
        raise ValueError(
            "adapter is incompatible with the active base model or feature contract: "
            f"expected={expected!r}, actual={actual!r}"
        )
