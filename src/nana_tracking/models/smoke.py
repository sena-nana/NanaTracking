"""Tiny non-production multi-head model for framework verification."""

import torch
from torch import Tensor, nn

from nana_tracking.config import ModelConfig


class SmokeTrackingModel(nn.Module):
    """Exercise rig, pose, and confidence heads without claiming tracking quality."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_channels, config.hidden_dims),
            nn.Tanh(),
        )
        self.rig_head = nn.Linear(config.hidden_dims, config.rig_dims)
        self.pose_head = nn.Linear(config.hidden_dims, config.pose_dims)
        self.confidence_head = nn.Linear(config.hidden_dims, config.rig_dims)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.encoder(self.pool(image))
        return (
            self.rig_head(features),
            self.pose_head(features),
            torch.sigmoid(self.confidence_head(features)),
        )


def create_model(config: ModelConfig) -> SmokeTrackingModel:
    if config.name != "smoke":
        raise ValueError(f"unknown model: {config.name}")
    return SmokeTrackingModel(config)
