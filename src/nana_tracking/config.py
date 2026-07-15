"""Strongly typed experiment configuration."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled or stale configuration fields."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    dataset: Literal["synthetic"] = "synthetic"
    manifest: Path | None = None
    samples: PositiveInt = 16
    batch_size: PositiveInt = 4
    executor: Literal["inline", "multiprocessing", "interpreter"] = "inline"
    workers: int = Field(default=0, ge=0)
    buffersize: PositiveInt = 2

    @model_validator(mode="after")
    def validate_executor(self) -> DataConfig:
        if self.executor != "inline" and self.workers < 1:
            raise ValueError("parallel executors require workers >= 1")
        return self


class ModelConfig(StrictModel):
    name: Literal["smoke"] = "smoke"
    input_channels: PositiveInt = 3
    input_height: PositiveInt = 8
    input_width: PositiveInt = 8
    hidden_dims: PositiveInt = 16
    rig_dims: PositiveInt = 4
    pose_dims: PositiveInt = 3


class TrainingConfig(StrictModel):
    seed: int = Field(default=7, ge=0)
    max_steps: PositiveInt = 2
    learning_rate: float = Field(default=0.01, gt=0)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    amp: bool = False


class EvaluationConfig(StrictModel):
    atol: float = Field(default=1e-5, gt=0)
    rtol: float = Field(default=1e-4, gt=0)


class ExportConfig(StrictModel):
    opset: int = Field(default=18, ge=18)
    model_family: str = Field(min_length=1)
    model_version: str = Field(min_length=1)


class ReproducibilityConfig(StrictModel):
    output_dir: Path = Path("runs")
    data_revision: str = Field(min_length=1)
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    feature_revision: str = Field(min_length=1)


class ExperimentConfig(StrictModel):
    """Complete input to a reproducible train/evaluate/export run."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    export: ExportConfig
    reproducibility: ReproducibilityConfig


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
