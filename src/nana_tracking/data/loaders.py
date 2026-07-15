"""Dataset loader registry."""

from torch.utils.data import DataLoader

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import TrackingBatch
from nana_tracking.data.face_basic import create_manifest_loader
from nana_tracking.data.full_set import create_full_set_loader
from nana_tracking.data.synthetic import create_loader as create_synthetic_loader


def create_loader(
    config: ExperimentConfig,
    *,
    split: str,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader[TrackingBatch]:
    if config.data.dataset == "synthetic":
        return create_synthetic_loader(config, shuffle=shuffle, seed_offset=seed_offset)
    if config.model.name == "full_set":
        return create_full_set_loader(config, split=split, shuffle=shuffle)
    return create_manifest_loader(config, split=split, shuffle=shuffle)
