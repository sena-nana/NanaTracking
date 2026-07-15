"""Manifest-backed FaceBasic training loader.

Raw images remain outside Git. Only approved, identity-safe manifest records with complete Basic
truth enter the default training set; missing auxiliary geometry is masked instead of fabricated.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2.functional import resize

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import TrackingBatch
from nana_tracking.data.labeling import MaterializedRecord, materialize_dataset
from nana_tracking.data.manifest import DatasetManifest
from nana_tracking.data.schema import CaptureRecord, LabelObservation


@dataclass(frozen=True, slots=True)
class FaceBasicSample:
    image: Tensor
    targets: dict[str, Tensor]
    label_confidence: dict[str, Tensor]
    sample_id: str


def _resolve_image_uri(manifest_path: Path, uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"FaceBasic loader supports local or file:// RGB URIs, got {uri!r}")
    raw = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(raw)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _best_observation(
    record: CaptureRecord,
    name: str,
    max_teacher_skew_ns: int,
) -> LabelObservation | None:
    candidates = [
        observation
        for teacher in record.teachers
        if abs(teacher.capture_timestamp_ns - record.capture_timestamp_ns) <= max_teacher_skew_ns
        and (observation := teacher.labels.get(name)) is not None
        and observation.state in {"observed", "fused"}
        and observation.value is not None
    ]
    return max(candidates, key=lambda item: item.confidence, default=None)


def _auxiliary_vector(
    record: CaptureRecord,
    names: list[str],
    defaults: list[float],
    max_teacher_skew_ns: int,
) -> tuple[Tensor, Tensor]:
    values: list[float] = []
    confidence: list[float] = []
    for name, default in zip(names, defaults, strict=True):
        observation = _best_observation(record, name, max_teacher_skew_ns)
        value = None if observation is None else observation.value
        values.append(default if value is None else float(value))
        confidence.append(0.0 if observation is None else observation.confidence)
    return torch.tensor(values), torch.tensor(confidence)


class FaceBasicDataset(Dataset[FaceBasicSample]):
    def __init__(self, config: ExperimentConfig, *, split: str) -> None:
        if config.data.manifest is None:
            raise ValueError("FaceBasicDataset requires data.manifest")
        self._config = config
        self._manifest_path = config.data.manifest.resolve()
        manifest = DatasetManifest.load(self._manifest_path)
        manifest.verify_files(self._manifest_path)
        self._max_teacher_skew_ns = manifest.synchronization.max_teacher_skew_ns
        if split not in manifest.splits:
            raise ValueError(f"manifest has no {split!r} split")
        identities = set(manifest.splits[split].identities)
        self._identity_indices = {
            identity: index
            for index, identity in enumerate(sorted(manifest.splits[split].identities))
        }
        if len(self._identity_indices) > config.model.identity_classes:
            raise ValueError(
                "model.identity_classes is smaller than the selected manifest identity count"
            )

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
            if labels.identity_id not in identities:
                continue
            signal_count = 41 if config.model.name == "face_spatial" else 36
            complete = all(label.state == "available" for label in labels.labels[:signal_count])
            require_complete = (
                config.data.require_complete_spatial
                if config.model.name == "face_spatial"
                else config.data.require_complete_basic
            )
            if require_complete and not complete:
                continue
            self._records.append((raw_records[labels.record_id], labels))
        if not self._records:
            raise ValueError(f"no usable FaceBasic records remain in split {split!r}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> FaceBasicSample:
        record, materialized = self._records[index]
        image_path = _resolve_image_uri(self._manifest_path, record.rgb.uri)
        image = decode_image(str(image_path), mode=ImageReadMode.RGB).to(torch.float32) / 255.0
        if image.shape[1:] != (record.rgb.height, record.rgb.width):
            raise ValueError(
                f"RGB dimensions for {record.record_id!r} do not match its capture record"
            )
        image = resize(
            image,
            [self._config.model.input_height, self._config.model.input_width],
            antialias=True,
        )

        signal_count = 41 if self._config.model.name == "face_spatial" else 36
        profile_labels = materialized.labels[:signal_count]
        rig = torch.tensor([label.value or 0.0 for label in profile_labels])
        rig_confidence = torch.tensor([label.confidence for label in profile_labels])
        pose_names = [
            "head.pose.position.x",
            "head.pose.position.y",
            "head.pose.position.z",
            "head.pose.orientation.x",
            "head.pose.orientation.y",
            "head.pose.orientation.z",
            "head.pose.orientation.w",
        ]
        pose, pose_confidence = _auxiliary_vector(
            record,
            pose_names,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            self._max_teacher_skew_ns,
        )
        landmark_axes = ("x", "y", "z") if self._config.model.name == "face_spatial" else ("x", "y")
        landmark_prefix = (
            "face.canonical" if self._config.model.name == "face_spatial" else "face.landmark"
        )
        landmark_names = [
            f"{landmark_prefix}.{point}.{axis}"
            for point in range(self._config.model.landmark_count)
            for axis in landmark_axes
        ]
        landmarks, landmark_confidence = _auxiliary_vector(
            record,
            landmark_names,
            [0.0] * len(landmark_names),
            self._max_teacher_skew_ns,
        )
        coordinate_count = len(landmark_axes)
        landmarks = landmarks.reshape(self._config.model.landmark_count, coordinate_count)
        landmark_confidence = landmark_confidence.reshape(
            self._config.model.landmark_count, coordinate_count
        )
        if "out_of_frame" in record.conditions.occlusions:
            visibility = 2
        elif record.conditions.occlusions:
            visibility = 1
        else:
            visibility = 0
        confidence_target = rig_confidence.clone()
        geometry_target = (
            "face_geometry" if self._config.model.name == "face_spatial" else "landmarks"
        )
        targets = {
            "rig": rig,
            "pose": pose,
            geometry_target: landmarks,
            "visibility": torch.tensor(visibility, dtype=torch.long),
            "identity": torch.tensor(self._identity_indices[record.identity_id], dtype=torch.long),
            "confidence": confidence_target,
        }
        weights = {
            "rig": rig_confidence,
            "pose": pose_confidence,
            geometry_target: landmark_confidence,
            "visibility": torch.ones(1),
            "identity": torch.ones(1),
            "confidence": torch.ones_like(confidence_target),
        }
        if self._config.model.name == "face_spatial":
            eye_origin_names = [
                f"face.eye.{side}.origin.{axis}"
                for side in ("left", "right")
                for axis in ("x", "y", "z")
            ]
            eye_direction_names = [
                f"face.eye.{side}.direction.{axis}"
                for side in ("left", "right")
                for axis in ("x", "y", "z")
            ]
            eye_origins, eye_origin_confidence = _auxiliary_vector(
                record,
                eye_origin_names,
                [-0.15, 0.05, 0.0, 0.15, 0.05, 0.0],
                self._max_teacher_skew_ns,
            )
            eye_directions, eye_direction_confidence = _auxiliary_vector(
                record,
                eye_direction_names,
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                self._max_teacher_skew_ns,
            )
            look_at, look_at_confidence = _auxiliary_vector(
                record,
                [f"face.look_at_head.{axis}" for axis in ("x", "y", "z")],
                [0.0, 0.0, 1.0],
                self._max_teacher_skew_ns,
            )
            tongue_observation = _best_observation(
                record, "tongue.visible", self._max_teacher_skew_ns
            )
            tongue_visible = (
                0
                if tongue_observation is None or float(tongue_observation.value or 0.0) < 0.5
                else 1
            )
            tongue_confidence = (
                0.0 if tongue_observation is None else tongue_observation.confidence
            )
            targets.update(
                {
                    "eye_origins": eye_origins.reshape(2, 3),
                    "eye_directions": eye_directions.reshape(2, 3),
                    "look_at_head": look_at,
                    "tongue_visibility": torch.tensor(tongue_visible, dtype=torch.long),
                }
            )
            weights.update(
                {
                    "eye_origins": eye_origin_confidence.reshape(2, 3),
                    "eye_directions": eye_direction_confidence.reshape(2, 3),
                    "look_at_head": look_at_confidence,
                    "tongue_visibility": torch.tensor([tongue_confidence]),
                }
            )
        return FaceBasicSample(image, targets, weights, record.record_id)


def collate_face_basic(samples: list[FaceBasicSample]) -> TrackingBatch:
    return TrackingBatch(
        images=torch.stack([sample.image for sample in samples]),
        targets={
            name: torch.stack([sample.targets[name] for sample in samples])
            for name in samples[0].targets
        },
        label_confidence={
            name: torch.stack([sample.label_confidence[name] for sample in samples])
            for name in samples[0].label_confidence
        },
        sample_ids=tuple(sample.sample_id for sample in samples),
    )


def create_manifest_loader(
    config: ExperimentConfig,
    *,
    split: str,
    shuffle: bool,
) -> DataLoader[TrackingBatch]:
    dataset = FaceBasicDataset(config, split=split)
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
