"""Manifest-backed FullSet upper-body loader with identity-safe admission gates."""

from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2.functional import resize

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import TrackingBatch
from nana_tracking.data.face_basic import (
    FaceBasicSample,
    auxiliary_vector,
    collate_face_basic,
    resolve_image_uri,
)
from nana_tracking.data.labeling import MaterializedRecord, materialize_dataset
from nana_tracking.data.manifest import DatasetManifest
from nana_tracking.data.schema import CaptureRecord


def _visibility_targets(record: CaptureRecord) -> torch.Tensor:
    tags = set(record.conditions.occlusions)
    if "out_of_frame" in tags:
        state = 3
    elif "self_occlusion" in tags:
        state = 2
    elif tags:
        state = 1
    else:
        state = 0
    return torch.full((5,), state, dtype=torch.long)


class FullSetDataset(Dataset[FaceBasicSample]):
    """Load complete Full-only scalar truth and masked torso-local geometry truth."""

    def __init__(self, config: ExperimentConfig, *, split: str) -> None:
        if config.data.manifest is None:
            raise ValueError("FullSetDataset requires data.manifest")
        self._config = config
        self._manifest_path = config.data.manifest.resolve()
        manifest = DatasetManifest.load(self._manifest_path)
        manifest.verify_files(self._manifest_path)
        if split not in manifest.splits:
            raise ValueError(f"manifest has no {split!r} split")
        identities = set(manifest.splits[split].identities)
        self._identity_indices = {
            identity: index for index, identity in enumerate(sorted(identities))
        }
        if len(self._identity_indices) > config.model.identity_classes:
            raise ValueError("model.identity_classes is smaller than the manifest identity count")
        self._max_teacher_skew_ns = manifest.synchronization.max_teacher_skew_ns
        raw_records: dict[str, CaptureRecord] = {}
        for reference in manifest.record_files:
            path = manifest.resolve(self._manifest_path, reference)
            for record in CaptureRecord.load_jsonl(path):
                raw_records[record.record_id] = record
        materialized = materialize_dataset(self._manifest_path)
        if materialized.quality.error_count:
            raise ValueError("manifest failed data quality gates")
        self._records: list[tuple[CaptureRecord, MaterializedRecord]] = []
        for labels in materialized.records:
            full_labels = labels.labels[41:76]
            if labels.identity_id not in identities:
                continue
            if config.data.require_complete_full and not all(
                label.state == "available" for label in full_labels
            ):
                continue
            self._records.append((raw_records[labels.record_id], labels))
        if not self._records:
            raise ValueError(f"no usable FullSet records remain in split {split!r}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> FaceBasicSample:
        record, materialized = self._records[index]
        image_path = resolve_image_uri(self._manifest_path, record.rgb.uri)
        image = decode_image(str(image_path), mode=ImageReadMode.RGB).to(torch.float32) / 255.0
        if image.shape[1:] != (record.rgb.height, record.rgb.width):
            raise ValueError(f"RGB dimensions for {record.record_id!r} do not match the record")
        image = resize(
            image,
            [self._config.model.input_height, self._config.model.input_width],
            antialias=True,
        )
        labels = materialized.labels[41:76]
        rig = torch.tensor([label.value or 0.0 for label in labels])
        rig_confidence = torch.tensor([label.confidence for label in labels])
        torso_pose, torso_confidence = auxiliary_vector(
            record,
            [f"torso.pose.position.{axis}" for axis in "xyz"]
            + [f"torso.pose.orientation.{axis}" for axis in "xyzw"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            self._max_teacher_skew_ns,
        )
        positions, positions_confidence = auxiliary_vector(
            record,
            [
                f"body.{side}.{joint}.position.{axis}"
                for side in ("left", "right")
                for joint in ("shoulder", "elbow", "wrist")
                for axis in "xyz"
            ],
            [0.0] * 18,
            self._max_teacher_skew_ns,
        )
        rotations, rotations_confidence = auxiliary_vector(
            record,
            [
                f"body.{side}.{joint}.orientation.{axis}"
                for side in ("left", "right")
                for joint in ("shoulder", "elbow", "wrist")
                for axis in "xyzw"
            ],
            [0.0, 0.0, 0.0, 1.0] * 6,
            self._max_teacher_skew_ns,
        )
        directions, directions_confidence = auxiliary_vector(
            record,
            [
                f"body.{side}.{limb}.direction.{axis}"
                for side in ("left", "right")
                for limb in ("upper_arm", "forearm")
                for axis in "xyz"
            ],
            [0.0, 1.0, 0.0] * 4,
            self._max_teacher_skew_ns,
        )
        twists, twists_confidence = auxiliary_vector(
            record,
            [
                f"body.{side}.{limb}.twist"
                for side in ("left", "right")
                for limb in ("upper_arm", "forearm")
            ],
            [0.0] * 4,
            self._max_teacher_skew_ns,
        )
        lengths, lengths_confidence = auxiliary_vector(
            record,
            [
                f"body.{side}.{limb}.normalized_length"
                for side in ("left", "right")
                for limb in ("upper_arm", "forearm")
            ],
            [0.5] * 4,
            self._max_teacher_skew_ns,
        )
        targets = {
            "rig": rig,
            "torso_pose": torso_pose,
            "joint_positions": positions.reshape(2, 3, 3),
            "joint_rotations": rotations.reshape(2, 3, 4),
            "limb_directions": directions.reshape(2, 2, 3),
            "limb_twists": twists.reshape(2, 2),
            "bone_lengths": lengths.reshape(2, 2),
            "visibility": _visibility_targets(record),
            "identity": torch.tensor(self._identity_indices[record.identity_id]),
            "confidence": rig_confidence.clone(),
        }
        weights = {
            "rig": rig_confidence,
            "torso_pose": torso_confidence,
            "joint_positions": positions_confidence.reshape(2, 3, 3),
            "joint_rotations": rotations_confidence.reshape(2, 3, 4),
            "limb_directions": directions_confidence.reshape(2, 2, 3),
            "limb_twists": twists_confidence.reshape(2, 2),
            "bone_lengths": lengths_confidence.reshape(2, 2),
            "visibility": torch.ones(5),
            "identity": torch.ones(1),
            "confidence": torch.ones_like(rig_confidence),
        }
        return FaceBasicSample(image, targets, weights, record.record_id)


def create_full_set_loader(
    config: ExperimentConfig, *, split: str, shuffle: bool
) -> DataLoader[TrackingBatch]:
    dataset = FullSetDataset(config, split=split)
    generator = torch.Generator().manual_seed(config.training.seed)
    multiprocessing = config.data.executor == "multiprocessing"
    return cast(
        DataLoader[TrackingBatch],
        DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=collate_face_basic,
            num_workers=config.data.workers if multiprocessing else 0,
            prefetch_factor=config.data.buffersize if multiprocessing else None,
        ),
    )
