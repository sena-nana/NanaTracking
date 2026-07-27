"""Dataset loader registry."""

from torch.utils.data import DataLoader

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import TrackingBatch
from nana_tracking.data.face_basic import create_manifest_loader
from nana_tracking.data.full_set import create_full_set_loader
from nana_tracking.data.manifest import DatasetManifest
from nana_tracking.data.stage_a_smoke import create_stage_a_smoke_loader
from nana_tracking.data.synthetic import create_loader as create_synthetic_loader


def create_loader(
    config: ExperimentConfig,
    *,
    split: str,
    shuffle: bool,
    seed_offset: int = 0,
    shuffle_seed_offset: int | None = None,
) -> DataLoader[TrackingBatch]:
    shuffle_offset = seed_offset if shuffle_seed_offset is None else shuffle_seed_offset
    if config.data.dataset == "synthetic":
        return create_synthetic_loader(
            config,
            shuffle=shuffle,
            seed_offset=seed_offset,
            shuffle_seed_offset=shuffle_offset,
        )
    if config.data.dataset == "multiview_smoke":
        return create_stage_a_smoke_loader(  # type: ignore[return-value]
            config,
            shuffle=shuffle,
            seed_offset=shuffle_offset,
        )
    if config.data.dataset == "manifest" and config.data.manifest is not None:
        manifest = DatasetManifest.load(config.data.manifest)
        if manifest.capture_schema_version == "nana-stage-a-materialization/1.0.0":
            raise ValueError(
                "Canonical core-16 candidate manifests require the future HR-Canonical loader; "
                "the legacy FaceBasic manifest loader is smoke-only"
            )
    if config.model.name == "full_set":
        return create_full_set_loader(
            config,
            split=split,
            shuffle=shuffle,
            seed_offset=shuffle_offset,
        )
    return create_manifest_loader(
        config,
        split=split,
        shuffle=shuffle,
        seed_offset=shuffle_offset,
    )
