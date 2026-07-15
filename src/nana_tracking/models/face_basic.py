"""Single-pass multi-task FaceBasic model.

The encoder runs exactly once. Geometry and identity heads are auxiliary training signals; only
the rig, pose, visibility, and confidence outputs cross the NTP producer boundary.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from nana_tracking.config import ModelConfig

FACE_BASIC_OUTPUT_NAMES = (
    "rig",
    "pose",
    "landmarks",
    "visibility",
    "identity",
    "confidence",
)

_UNSIGNED_BASIC_SLOTS = (8, 9, 12, 13, 14, 15, 16, 27, 29, 32, 33, 34, 35)


class DepthwiseBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.block(image)


def create_face_encoder(config: ModelConfig) -> nn.Sequential:
    """Create the single shared image encoder used by both face profiles."""

    widths = (24, 40, 64, config.hidden_dims)
    return nn.Sequential(
        nn.Conv2d(config.input_channels, widths[0], 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(widths[0]),
        nn.SiLU(inplace=True),
        DepthwiseBlock(widths[0], widths[1], 2),
        DepthwiseBlock(widths[1], widths[2], 2),
        DepthwiseBlock(widths[2], widths[3], 2),
        DepthwiseBlock(widths[3], widths[3], 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(config.dropout),
    )


class FaceBasicModel(nn.Module):
    """Lightweight shared encoder with explicit orthogonal multi-task heads."""

    unsigned_slots: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = create_face_encoder(config)
        self.rig_head = nn.Linear(config.hidden_dims, 36)
        self.pose_head = nn.Linear(config.hidden_dims, 7)
        with torch.no_grad():
            self.pose_head.bias[6] = 1.0
        self.landmark_head = nn.Linear(config.hidden_dims, config.landmark_count * 2)
        self.visibility_head = nn.Linear(config.hidden_dims, 3)
        self.identity_head = nn.Sequential(
            nn.Linear(config.hidden_dims, config.identity_dims),
            nn.SiLU(),
            nn.Linear(config.identity_dims, config.identity_classes),
        )
        self.confidence_head = nn.Linear(config.hidden_dims, 36)
        self.landmark_count = config.landmark_count
        unsigned = torch.zeros(36, dtype=torch.bool)
        unsigned[list(_UNSIGNED_BASIC_SLOTS)] = True
        self.register_buffer("unsigned_slots", unsigned, persistent=False)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.encoder(image)
        raw_rig = self.rig_head(features)
        rig = torch.where(self.unsigned_slots, torch.sigmoid(raw_rig), torch.tanh(raw_rig))
        raw_pose = self.pose_head(features)
        position = torch.tanh(raw_pose[:, :3])
        quaternion = nn.functional.normalize(raw_pose[:, 3:], dim=-1, eps=1e-6)
        pose = torch.cat((position, quaternion), dim=-1)
        landmarks = torch.tanh(self.landmark_head(features)).reshape(-1, self.landmark_count, 2)
        visibility = self.visibility_head(features)
        reversed_features = -features + (2.0 * features).detach()
        identity = self.identity_head(reversed_features)
        confidence = torch.sigmoid(self.confidence_head(features))
        return rig, pose, landmarks, visibility, identity, confidence


def mirror_basic_rig(values: Tensor) -> Tensor:
    """Reflect BasicSet values anatomically for image-mirror consistency training."""

    if values.shape[-1] != 36:
        raise ValueError("BasicSet tensors require 36 values")
    mapping: Sequence[int] = (
        1,
        0,
        3,
        2,
        5,
        4,
        7,
        6,
        9,
        8,
        11,
        10,
        13,
        12,
        15,
        14,
        16,
        17,
        18,
        20,
        19,
        22,
        21,
        24,
        23,
        26,
        25,
        27,
        28,
        29,
        30,
        31,
        33,
        32,
        35,
        34,
    )
    mirrored = values[..., list(mapping)]
    # Lateral jaw translation changes sign under anatomical reflection.
    signs = torch.ones(36, dtype=values.dtype, device=values.device)
    signs[17] = -1.0
    return mirrored * signs
