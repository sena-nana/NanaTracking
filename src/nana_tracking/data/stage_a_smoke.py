"""Deterministic synthetic three-view fixture for Stage A control-flow tests only."""

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import MultiViewTrackingBatch


@dataclass(frozen=True, slots=True)
class _SmokeGroup:
    images: Tensor
    targets: dict[str, Tensor]
    confidence: dict[str, Tensor]
    intrinsics: Tensor
    camera_to_capture: Tensor
    sample_id: str
    identity_id: str
    timestamp_ns: int


class StageASmokeDataset(Dataset[_SmokeGroup]):
    def __init__(self, config: ExperimentConfig, *, seed_offset: int) -> None:
        self._config = config
        self._seed_offset = seed_offset

    def __len__(self) -> int:
        return self._config.data.samples

    def __getitem__(self, index: int) -> _SmokeGroup:
        generator = torch.Generator().manual_seed(
            self._config.training.seed + self._seed_offset + index
        )
        anchor_count = self._config.model.landmark_count
        geometry = torch.rand((anchor_count, 3), generator=generator) * 0.4 - 0.2
        geometry[:, 2] *= 0.25
        pose = torch.tensor([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0])
        intrinsics = torch.tensor(
            [
                [48.0, 0.0, self._config.model.input_width / 2],
                [0.0, 48.0, self._config.model.input_height / 2],
                [0.0, 0.0, 1.0],
            ]
        )
        points = geometry + pose[:3]
        pixels = (intrinsics @ points.T).T
        pixels = pixels[:, :2] / pixels[:, 2:]
        landmarks = torch.stack(
            (
                pixels[:, 0] / self._config.model.input_width * 2.0 - 1.0,
                pixels[:, 1] / self._config.model.input_height * 2.0 - 1.0,
            ),
            dim=-1,
        )
        identity = index % self._config.model.identity_classes
        base = torch.rand(
            (
                3,
                self._config.model.input_height,
                self._config.model.input_width,
            ),
            generator=generator,
        )
        base = base * 0.05 + (identity + 1) / (self._config.model.identity_classes + 1)
        images = torch.stack(
            [
                (base + view * 0.01).clamp(0.0, 1.0)
                for view in (-1.0, 0.0, 1.0)
            ]
        )
        return _SmokeGroup(
            images=images,
            targets={
                "landmarks": landmarks.repeat(3, 1, 1),
                "canonical_geometry": geometry.repeat(3, 1, 1),
                "pose": pose.repeat(3, 1),
                "visibility": torch.zeros(3, dtype=torch.long),
                "identity": torch.full((3,), identity, dtype=torch.long),
            },
            confidence={
                "landmarks": torch.ones(3, anchor_count, 2),
                "canonical_geometry": torch.ones(3, anchor_count, 3),
                "pose": torch.ones(3, 7),
                "visibility": torch.ones(3, 1),
                "identity": torch.ones(3, 1),
            },
            intrinsics=intrinsics.repeat(3, 1, 1),
            camera_to_capture=torch.eye(4).repeat(3, 1, 1),
            sample_id=f"synthetic-stage-a-{index:05d}",
            identity_id=f"synthetic-{identity}",
            timestamp_ns=index * 33_333_333,
        )


def _collate(groups: list[_SmokeGroup]) -> MultiViewTrackingBatch:
    return MultiViewTrackingBatch(
        images=torch.stack([group.images for group in groups]),
        targets={
            name: torch.stack([group.targets[name] for group in groups])
            for name in groups[0].targets
        },
        label_confidence={
            name: torch.stack([group.confidence[name] for group in groups])
            for name in groups[0].confidence
        },
        camera_intrinsics=torch.stack([group.intrinsics for group in groups]),
        camera_to_capture=torch.stack([group.camera_to_capture for group in groups]),
        sample_ids=tuple(group.sample_id for group in groups),
        identity_ids=tuple(group.identity_id for group in groups),
        sequence_ids=tuple("synthetic-sequence" for _ in groups),
        expressions=tuple("synthetic-dynamic-smoke" for _ in groups),
        timestamps_ns=torch.tensor([group.timestamp_ns for group in groups]),
    )


def create_stage_a_smoke_loader(
    config: ExperimentConfig,
    *,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader[MultiViewTrackingBatch]:
    dataset = StageASmokeDataset(config, seed_offset=seed_offset)
    generator = torch.Generator().manual_seed(config.training.seed + seed_offset)
    return cast(
        DataLoader[MultiViewTrackingBatch],
        DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=_collate,
        ),
    )
