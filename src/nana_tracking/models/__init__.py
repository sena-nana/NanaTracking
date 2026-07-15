"""Model registry."""

from torch import nn

from nana_tracking.config import ModelConfig
from nana_tracking.models.face_basic import (
    FACE_BASIC_OUTPUT_NAMES,
    FaceBasicModel,
    mirror_basic_rig,
)
from nana_tracking.models.smoke import SmokeTrackingModel

SMOKE_OUTPUT_NAMES = ("rig", "pose", "confidence")


def create_model(config: ModelConfig) -> nn.Module:
    if config.name == "smoke":
        return SmokeTrackingModel(config)
    if config.name == "face_basic":
        return FaceBasicModel(config)
    raise ValueError(f"unknown model: {config.name}")


def output_names(config: ModelConfig) -> tuple[str, ...]:
    return SMOKE_OUTPUT_NAMES if config.name == "smoke" else FACE_BASIC_OUTPUT_NAMES


__all__ = [
    "FACE_BASIC_OUTPUT_NAMES",
    "FaceBasicModel",
    "SmokeTrackingModel",
    "create_model",
    "mirror_basic_rig",
    "output_names",
]
