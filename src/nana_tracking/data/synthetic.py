"""Deterministic non-production data used to exercise the framework."""

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import TrackingBatch


@dataclass(frozen=True, slots=True)
class SyntheticSample:
    image: Tensor
    rig: Tensor
    pose: Tensor
    confidence: Tensor
    sample_id: str


class SyntheticTrackingDataset(Dataset[SyntheticSample]):
    """Tiny deterministic dataset; never a proxy for face-tracking quality."""

    def __init__(self, config: ExperimentConfig, *, seed_offset: int = 0) -> None:
        self._config = config
        self._seed_offset = seed_offset

    def __len__(self) -> int:
        return self._config.data.samples

    def __getitem__(self, index: int) -> SyntheticSample:
        model = self._config.model
        generator = torch.Generator().manual_seed(
            self._config.training.seed + self._seed_offset + index
        )
        image = torch.rand(
            model.input_channels,
            model.input_height,
            model.input_width,
            generator=generator,
        )
        channel_means = image.mean(dim=(1, 2))
        rig_seed = torch.cat((channel_means, image.mean().view(1)))
        repeats = (model.rig_dims + rig_seed.numel() - 1) // rig_seed.numel()
        rig = rig_seed.repeat(repeats)[: model.rig_dims]
        pose_seed = torch.stack((image.mean(), image.std(), image.max()))
        pose_repeats = (model.pose_dims + pose_seed.numel() - 1) // pose_seed.numel()
        pose = pose_seed.repeat(pose_repeats)[: model.pose_dims]
        confidence = torch.ones(model.rig_dims)
        return SyntheticSample(image, rig, pose, confidence, f"synthetic-{index:05d}")


def collate_tracking(samples: list[SyntheticSample]) -> TrackingBatch:
    return TrackingBatch(
        images=torch.stack([sample.image for sample in samples]),
        targets={
            "rig": torch.stack([sample.rig for sample in samples]),
            "pose": torch.stack([sample.pose for sample in samples]),
            "confidence": torch.stack([sample.confidence for sample in samples]),
        },
        label_confidence={
            "rig": torch.ones(len(samples), samples[0].rig.numel()),
            "pose": torch.ones(len(samples), samples[0].pose.numel()),
        },
        sample_ids=tuple(sample.sample_id for sample in samples),
    )


def create_loader(
    config: ExperimentConfig,
    *,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader[TrackingBatch]:
    dataset = SyntheticTrackingDataset(config, seed_offset=seed_offset)
    generator = torch.Generator().manual_seed(config.training.seed + seed_offset)
    return cast(
        DataLoader[TrackingBatch],
        DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=collate_tracking,
            num_workers=0,
        ),
    )
