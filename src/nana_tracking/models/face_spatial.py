"""Single-pass FaceSpatial model with continuous gaze and canonical geometry heads."""

import torch
from torch import Tensor, nn

from nana_tracking.config import ModelConfig
from nana_tracking.models.face_basic import create_face_encoder, mirror_basic_rig

FACE_SPATIAL_OUTPUT_NAMES = (
    "rig",
    "pose",
    "eye_origins",
    "eye_directions",
    "look_at_head",
    "face_geometry",
    "visibility",
    "tongue_visibility",
    "identity",
    "confidence",
)

_UNSIGNED_SPATIAL_SLOTS = (8, 9, 12, 13, 14, 15, 16, 27, 29, 32, 33, 34, 35, 40)


class FaceSpatialModel(nn.Module):
    """One shared encoder with explicit SpatialSet and normalized-geometry heads."""

    unsigned_slots: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = create_face_encoder(config)
        self.rig_head = nn.Linear(config.hidden_dims, 41)
        self.pose_head = nn.Linear(config.hidden_dims, 7)
        self.eye_origin_head = nn.Linear(config.hidden_dims, 6)
        self.eye_direction_head = nn.Linear(config.hidden_dims, 6)
        self.look_at_head = nn.Linear(config.hidden_dims, 3)
        self.face_geometry_head = nn.Linear(config.hidden_dims, config.landmark_count * 3)
        self.visibility_head = nn.Linear(config.hidden_dims, 3)
        self.tongue_visibility_head = nn.Linear(config.hidden_dims, 2)
        self.identity_head = nn.Sequential(
            nn.Linear(config.hidden_dims, config.identity_dims),
            nn.SiLU(),
            nn.Linear(config.identity_dims, config.identity_classes),
        )
        self.confidence_head = nn.Linear(config.hidden_dims, 41)
        self.landmark_count = config.landmark_count
        with torch.no_grad():
            self.pose_head.bias[6] = 1.0
            self.eye_direction_head.bias[2] = 1.0
            self.eye_direction_head.bias[5] = 1.0
            self.look_at_head.bias[2] = 1.0
        unsigned = torch.zeros(41, dtype=torch.bool)
        unsigned[list(_UNSIGNED_SPATIAL_SLOTS)] = True
        self.register_buffer("unsigned_slots", unsigned, persistent=False)

    def forward(
        self, image: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.encoder(image)
        raw_rig = self.rig_head(features)
        normalized = torch.where(self.unsigned_slots, torch.sigmoid(raw_rig), torch.tanh(raw_rig))
        rig = normalized.clone()
        rig[:, 36] = torch.tanh(raw_rig[:, 36]) * 1.2
        rig[:, 37] = torch.tanh(raw_rig[:, 37]) * 0.8
        rig[:, 38] = torch.tanh(raw_rig[:, 38]) * 1.2
        rig[:, 39] = torch.tanh(raw_rig[:, 39]) * 0.8

        raw_pose = self.pose_head(features)
        pose = torch.cat(
            (
                torch.tanh(raw_pose[:, :3]),
                nn.functional.normalize(raw_pose[:, 3:], dim=-1, eps=1e-6),
            ),
            dim=-1,
        )
        eye_origins = torch.tanh(self.eye_origin_head(features)).reshape(-1, 2, 3)
        eye_directions = nn.functional.normalize(
            self.eye_direction_head(features).reshape(-1, 2, 3), dim=-1, eps=1e-6
        )
        look_at_head = torch.tanh(self.look_at_head(features))
        face_geometry = torch.tanh(self.face_geometry_head(features)).reshape(
            -1, self.landmark_count, 3
        )
        visibility = self.visibility_head(features)
        tongue_visibility = self.tongue_visibility_head(features)
        reversed_features = -features + (2.0 * features).detach()
        identity = self.identity_head(reversed_features)
        confidence = torch.sigmoid(self.confidence_head(features))
        return (
            rig,
            pose,
            eye_origins,
            eye_directions,
            look_at_head,
            face_geometry,
            visibility,
            tongue_visibility,
            identity,
            confidence,
        )


def mirror_spatial_rig(values: Tensor) -> Tensor:
    """Reflect continuous binocular gaze and the nested BasicSet anatomically."""

    if values.shape[-1] != 41:
        raise ValueError("SpatialSet tensors require 41 values")
    mirrored = torch.empty_like(values)
    mirrored[..., :36] = mirror_basic_rig(values[..., :36])
    mirrored[..., 36] = -values[..., 38]
    mirrored[..., 37] = values[..., 39]
    mirrored[..., 38] = -values[..., 36]
    mirrored[..., 39] = values[..., 37]
    mirrored[..., 40] = values[..., 40]
    return mirrored
