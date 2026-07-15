"""Efficient FullSet extension model for torso, arms, tongue detail, and auricles."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from nana_tracking.config import ModelConfig
from nana_tracking.models.face_basic import create_face_encoder

FULL_SET_OUTPUT_NAMES = (
    "rig",
    "torso_pose",
    "joint_positions",
    "joint_rotations",
    "limb_directions",
    "limb_twists",
    "bone_lengths",
    "visibility",
    "identity",
    "confidence",
)

_UNSIGNED_FULL_SLOTS = (28, 33)


class FullSetModel(nn.Module):
    """Single-pass upper-body network with geometry-consistent multi-task heads."""

    unsigned_slots: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = create_face_encoder(config)
        self.rig_head = nn.Linear(config.hidden_dims, 35)
        self.torso_pose_head = nn.Linear(config.hidden_dims, 7)
        self.joint_position_head = nn.Linear(config.hidden_dims, 18)
        self.joint_rotation_head = nn.Linear(config.hidden_dims, 24)
        self.limb_direction_head = nn.Linear(config.hidden_dims, 12)
        self.limb_twist_head = nn.Linear(config.hidden_dims, 4)
        self.bone_length_head = nn.Linear(config.hidden_dims, 4)
        self.visibility_head = nn.Linear(config.hidden_dims, 30)
        self.identity_head = nn.Sequential(
            nn.Linear(config.hidden_dims, config.identity_dims),
            nn.SiLU(),
            nn.Linear(config.identity_dims, config.identity_classes),
        )
        self.confidence_head = nn.Linear(config.hidden_dims, 35)
        with torch.no_grad():
            self.torso_pose_head.bias[6] = 1.0
            self.joint_rotation_head.bias.reshape(2, 3, 4)[..., 3] = 1.0
            self.limb_direction_head.bias.reshape(2, 2, 3)[..., 1] = 1.0
        unsigned = torch.zeros(35, dtype=torch.bool)
        unsigned[list(_UNSIGNED_FULL_SLOTS)] = True
        self.register_buffer("unsigned_slots", unsigned, persistent=False)

    def forward(
        self, image: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.encoder(image)
        raw_rig = self.rig_head(features)
        rig = torch.where(self.unsigned_slots, torch.sigmoid(raw_rig), torch.tanh(raw_rig))
        raw_torso = self.torso_pose_head(features)
        torso_pose = torch.cat(
            (
                torch.tanh(raw_torso[:, :3]),
                nn.functional.normalize(raw_torso[:, 3:], dim=-1, eps=1e-6),
            ),
            dim=-1,
        )
        joint_positions = torch.tanh(self.joint_position_head(features)).reshape(-1, 2, 3, 3)
        joint_rotations = nn.functional.normalize(
            self.joint_rotation_head(features).reshape(-1, 2, 3, 4), dim=-1, eps=1e-6
        )
        limb_directions = nn.functional.normalize(
            self.limb_direction_head(features).reshape(-1, 2, 2, 3), dim=-1, eps=1e-6
        )
        limb_twists = torch.tanh(self.limb_twist_head(features)).reshape(-1, 2, 2)
        bone_lengths = torch.sigmoid(self.bone_length_head(features)).reshape(-1, 2, 2)
        visibility = self.visibility_head(features).reshape(-1, 5, 6)
        reversed_features = -features + (2.0 * features).detach()
        identity = self.identity_head(reversed_features)
        confidence = torch.sigmoid(self.confidence_head(features))
        return (
            rig,
            torso_pose,
            joint_positions,
            joint_rotations,
            limb_directions,
            limb_twists,
            bone_lengths,
            visibility,
            identity,
            confidence,
        )


def mirror_full_rig(values: Tensor) -> Tensor:
    """Reflect the stable Full-only block (signals 42..76) anatomically."""

    if values.shape[-1] != 35:
        raise ValueError("FullSet extension tensors require 35 values")
    mapping: Sequence[int] = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        16,
        15,
        18,
        17,
        20,
        19,
        22,
        21,
        24,
        23,
        30,
        31,
        32,
        33,
        34,
        25,
        26,
        27,
        28,
        29,
    )
    signs = torch.ones(35, dtype=values.dtype, device=values.device)
    signs[[0, 4, 5, 6, 10, 11, 12, 27, 29, 32, 34]] = -1.0
    return values[..., list(mapping)] * signs
