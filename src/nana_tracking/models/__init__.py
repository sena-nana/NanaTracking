"""Model registry."""

from torch import Tensor, nn

from nana_tracking.config import ModelConfig
from nana_tracking.models.face_basic import (
    FACE_BASIC_OUTPUT_NAMES,
    FaceBasicModel,
    mirror_basic_rig,
)
from nana_tracking.models.face_spatial import (
    FACE_SPATIAL_OUTPUT_NAMES,
    FaceSpatialModel,
    mirror_spatial_rig,
)
from nana_tracking.models.full_set import FULL_SET_OUTPUT_NAMES, FullSetModel, mirror_full_rig
from nana_tracking.models.smoke import SmokeTrackingModel

SMOKE_OUTPUT_NAMES = ("rig", "pose", "confidence")


def create_model(config: ModelConfig) -> nn.Module:
    if config.name == "smoke":
        return SmokeTrackingModel(config)
    if config.name == "face_basic":
        return FaceBasicModel(config)
    if config.name == "face_spatial":
        return FaceSpatialModel(config)
    if config.name == "full_set":
        return FullSetModel(config)
    raise ValueError(f"unknown model: {config.name}")


def output_names(config: ModelConfig) -> tuple[str, ...]:
    if config.name == "smoke":
        return SMOKE_OUTPUT_NAMES
    if config.name == "face_basic":
        return FACE_BASIC_OUTPUT_NAMES
    return FACE_SPATIAL_OUTPUT_NAMES if config.name == "face_spatial" else FULL_SET_OUTPUT_NAMES


class FaceBasicDeploymentModel(nn.Module):
    """Remove the training-only identity adversary from deployable artifacts."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        rig, pose, landmarks, visibility, _identity, confidence = self.model(image)
        return rig, pose, landmarks, visibility, confidence


def create_deployment_model(config: ModelConfig, model: nn.Module) -> nn.Module:
    if config.name == "face_basic":
        return FaceBasicDeploymentModel(model)
    if config.name == "face_spatial":
        return FaceSpatialDeploymentModel(model)
    if config.name == "full_set":
        return FullSetDeploymentModel(model)
    return model


class FaceSpatialDeploymentModel(nn.Module):
    """Remove the training-only identity adversary from FaceSpatial packages."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, image: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        outputs = self.model(image)
        return (*outputs[:8], outputs[9])


class FullSetDeploymentModel(nn.Module):
    """Remove the training-only identity adversary from FullSet packages."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, image: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        outputs = self.model(image)
        return (*outputs[:8], outputs[9])


def deployment_output_names(config: ModelConfig) -> tuple[str, ...]:
    if config.name == "smoke":
        return SMOKE_OUTPUT_NAMES
    names = output_names(config)
    return tuple(name for name in names if name != "identity")


__all__ = [
    "FACE_BASIC_OUTPUT_NAMES",
    "FACE_SPATIAL_OUTPUT_NAMES",
    "FULL_SET_OUTPUT_NAMES",
    "FaceBasicDeploymentModel",
    "FaceBasicModel",
    "FaceSpatialDeploymentModel",
    "FaceSpatialModel",
    "FullSetDeploymentModel",
    "FullSetModel",
    "SmokeTrackingModel",
    "create_deployment_model",
    "create_model",
    "deployment_output_names",
    "mirror_basic_rig",
    "mirror_full_rig",
    "mirror_spatial_rig",
    "output_names",
]
