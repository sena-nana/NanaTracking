"""Strongly typed experiment configuration."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled or stale configuration fields."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    dataset: Literal["synthetic", "manifest"] = "synthetic"
    manifest: Path | None = None
    samples: PositiveInt = 16
    batch_size: PositiveInt = 4
    executor: Literal["inline", "multiprocessing", "interpreter"] = "inline"
    workers: int = Field(default=0, ge=0)
    buffersize: PositiveInt = 2
    require_complete_basic: bool = True

    @model_validator(mode="after")
    def validate_executor(self) -> DataConfig:
        if self.executor != "inline" and self.workers < 1:
            raise ValueError("parallel executors require workers >= 1")
        if self.dataset == "manifest" and self.manifest is None:
            raise ValueError("manifest datasets require data.manifest")
        if self.dataset == "synthetic" and self.manifest is not None:
            raise ValueError("synthetic datasets do not accept data.manifest")
        return self


class ModelConfig(StrictModel):
    name: Literal["smoke", "face_basic"] = "smoke"
    input_channels: PositiveInt = 3
    input_height: PositiveInt = 8
    input_width: PositiveInt = 8
    hidden_dims: PositiveInt = 16
    rig_dims: PositiveInt = 4
    pose_dims: PositiveInt = 3
    landmark_count: PositiveInt = 68
    identity_classes: PositiveInt = 2
    identity_dims: PositiveInt = 16
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_face_basic_contract(self) -> ModelConfig:
        if self.name == "face_basic":
            if self.rig_dims != 36:
                raise ValueError("face_basic requires the complete 36-signal BasicSet")
            if self.pose_dims != 7:
                raise ValueError("face_basic pose is xyz plus an xyzw quaternion (7 values)")
            if min(self.input_height, self.input_width) < 64:
                raise ValueError("face_basic ROI inputs must be at least 64x64")
        return self


class TrainingConfig(StrictModel):
    seed: int = Field(default=7, ge=0)
    max_steps: PositiveInt = 2
    learning_rate: float = Field(default=0.01, gt=0)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    amp: bool = False
    rig_loss_weight: float = Field(default=1.0, ge=0)
    pose_loss_weight: float = Field(default=0.5, ge=0)
    landmark_loss_weight: float = Field(default=0.25, ge=0)
    visibility_loss_weight: float = Field(default=0.1, ge=0)
    confidence_loss_weight: float = Field(default=0.2, ge=0)
    identity_adversary_weight: float = Field(default=0.05, ge=0)
    mirror_consistency_weight: float = Field(default=0.1, ge=0)


class EvaluationConfig(StrictModel):
    atol: float = Field(default=1e-5, gt=0)
    rtol: float = Field(default=1e-4, gt=0)


class ExportConfig(StrictModel):
    opset: int = Field(default=18, ge=18)
    model_family: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    smoke_only: bool = True


class ReproducibilityConfig(StrictModel):
    output_dir: Path = Path("runs")
    data_revision: str = Field(min_length=1)
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    normalization_revision: str = "ntp-normalization/1.0.0"
    calibration_revision: str = "ntp-calibration/1.0.0"
    feature_revision: str = Field(min_length=1)


class ExperimentConfig(StrictModel):
    """Complete input to a reproducible train/evaluate/export run."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    export: ExportConfig
    reproducibility: ReproducibilityConfig

    @model_validator(mode="after")
    def validate_artifact_status(self) -> ExperimentConfig:
        if not self.export.smoke_only and self.data.dataset != "manifest":
            raise ValueError("non-smoke exports require a reviewed manifest dataset")
        return self


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return ExperimentConfig.model_validate(raw)


def save_config(config: ExperimentConfig, path: Path) -> None:
    """Persist a resolved configuration in a stable human-readable form."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
