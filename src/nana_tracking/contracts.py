"""Framework-neutral training and artifact contracts."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TrackingBatch:
    images: Tensor
    targets: dict[str, Tensor]
    label_confidence: dict[str, Tensor]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrackingModelOutput:
    rig: Tensor
    pose: Tensor
    confidence: Tensor

    @classmethod
    def from_tuple(cls, values: tuple[Tensor, Tensor, Tensor]) -> TrackingModelOutput:
        return cls(rig=values[0], pose=values[1], confidence=values[2])

    def as_dict(self) -> dict[str, Tensor]:
        return {"rig": self.rig, "pose": self.pose, "confidence": self.confidence}


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckpointMetadata(ContractModel):
    run_id: str
    epoch: int = Field(ge=0)
    step: int = Field(ge=0)
    seed: int = Field(ge=0)
    config_digest: str
    data_revision: str
    ntp_schema_revision: str
    signal_registry_revision: str
    normalization_revision: str
    calibration_revision: str
    feature_revision: str
    device: str
    amp_enabled: bool
    git_commit: str
    git_dirty: bool
    lock_digest: str
    created_at: datetime


class ModelPackageMetadata(ContractModel):
    package_schema_version: str = "1"
    model_family: str
    model_version: str
    source_checkpoint_digest: str
    ntp_schema_revision: str
    signal_registry_revision: str
    normalization_revision: str
    calibration_revision: str
    feature_revision: str
    onnx_opset: int
    input_shape: list[int]
    output_names: list[str]
    model_digest: str
    smoke_only: bool = True
    input_layout: str = "NCHW"
    input_color: str = "RGB"
    input_range: tuple[float, float] = (0.0, 1.0)
    precision_support: list[str] = Field(default_factory=lambda: ["fp32"])
    supported_signals: list[int] = Field(default_factory=list)
    supported_structures: list[str] = Field(default_factory=list)
    temporal_state: str = "single-frame"
    geometry_topology_revision: str | None = None


class AdapterContract(ContractModel):
    adapter_schema_version: str = "1"
    adapter_type: str
    base_model_family: str
    base_model_version: str
    feature_revision: str
    output_mode: str = "residual"
    resettable: bool = True
