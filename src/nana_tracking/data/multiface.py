"""Fail-closed Multiface Stage A contracts, preflight, and three-view loader."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Literal, Self, cast
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2.functional import crop, resize

from nana_tracking.config import ExperimentConfig, save_config
from nana_tracking.contracts import MultiViewTrackingBatch
from nana_tracking.data.face_basic import resolve_image_uri
from nana_tracking.data.manifest import (
    DatasetManifest,
    FileReference,
    LicensePermissions,
    LicenseReview,
    RecordFile,
    SplitManifest,
    SynchronizationPolicy,
    TeacherSource,
    dataset_digest,
)
from nana_tracking.data.strategy import LicenseRegistry
from nana_tracking.reproducibility import sha256_file

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

SEMANTIC_ANCHORS = (
    "eye.left.inner",
    "eye.left.outer",
    "eye.right.inner",
    "eye.right.outer",
    "brow.left.inner",
    "brow.left.outer",
    "brow.right.inner",
    "brow.right.outer",
    "nose.tip",
    "nose.left.ala",
    "nose.right.ala",
    "mouth.left.corner",
    "mouth.right.corner",
    "lip.upper.center",
    "lip.lower.center",
    "chin.center",
)

CORE_EXPRESSIONS = (
    "E001_Neutral_Eyes_Open",
    "E003_Neutral_Eyes_Closed",
    "E004_Relaxed_Mouth_Open",
    "E008_Smile_Mouth_Closed",
)

DEFAULT_IDENTITY_PRIORITY = (
    "7889059",
    "2183941",
    "5372021",
    "8870559",
    "6674443",
    "5067077",
    "6795937",
    "002421669",
    "002539136",
    "002643814",
    "002645310",
    "002757580",
    "002914589",
)

_MULTIFACE_ROOT = (
    "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/"
    "MugsyDataRelease/v0.0/identities/"
)


class StageAModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnchorCorrespondence(StageAModel):
    semantic_name: str = Field(min_length=1)
    mediapipe_index: int = Field(ge=0)
    multiface_vertex_id: int = Field(ge=0, lt=7_306)


class AnchorMappingReview(StageAModel):
    approved_by: str = Field(min_length=1)
    approved_at: datetime
    identities: list[str] = Field(min_length=3)
    expressions: list[str] = Field(min_length=4)
    camera_ids: list[str] = Field(min_length=3)
    reviewed_sample_ids: list[str] = Field(min_length=12)
    overlay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SemanticAnchorMapping(StageAModel):
    schema_version: Literal["nana-semantic-anchor-map/1.0.0"]
    mapping_revision: str = Field(min_length=1)
    source_topology: Literal["multiface-tracked-mesh-7306"]
    mediapipe_version: str = Field(min_length=1)
    mediapipe_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "approved"]
    correspondences: list[AnchorCorrespondence]
    review: AnchorMappingReview | None
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_review_gate(self) -> Self:
        if self.status == "pending":
            return self
        names = tuple(item.semantic_name for item in self.correspondences)
        if names != SEMANTIC_ANCHORS:
            raise ValueError("approved mapping must contain the ordered 16 semantic anchors")
        vertex_ids = [item.multiface_vertex_id for item in self.correspondences]
        if len(vertex_ids) != len(set(vertex_ids)):
            raise ValueError("approved semantic anchors require unique Multiface vertices")
        if self.mediapipe_model_sha256 == "0" * 64:
            raise ValueError("approved mapping requires the real MediaPipe model digest")
        if self.review is None:
            raise ValueError("approved mapping requires human review evidence")
        if len(set(self.review.identities)) < 3:
            raise ValueError("anchor review requires at least three identities")
        if len(set(self.review.expressions)) < 4:
            raise ValueError("anchor review requires at least four expressions")
        if len(set(self.review.camera_ids)) < 3:
            raise ValueError("anchor review requires at least three camera views")
        return self

    @classmethod
    def load(cls, path: Path, *, require_approved: bool = False) -> Self:
        mapping = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if require_approved and mapping.status != "approved":
            raise ValueError("Stage A training requires an approved semantic anchor mapping")
        return mapping


class AnchorVote(StageAModel):
    semantic_name: str
    mediapipe_index: int = Field(ge=0)
    multiface_vertex_id: int = Field(ge=0, lt=7_306)
    distance: FiniteFloat = Field(ge=0.0)
    identity_id: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)

    @classmethod
    def load_jsonl(cls, path: Path) -> list[Self]:
        return [
            cls.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def build_anchor_mapping_proposal(
    votes: Iterable[AnchorVote],
    *,
    mapping_revision: str,
    mediapipe_version: str,
    mediapipe_model_sha256: str,
) -> SemanticAnchorMapping:
    """Reduce MediaPipe-assisted nearest-vertex votes; output remains unapproved."""

    grouped: dict[str, list[AnchorVote]] = {name: [] for name in SEMANTIC_ANCHORS}
    for vote in votes:
        if vote.semantic_name not in grouped:
            raise ValueError(f"unknown semantic anchor vote: {vote.semantic_name}")
        grouped[vote.semantic_name].append(vote)
    correspondences: list[AnchorCorrespondence] = []
    for name in SEMANTIC_ANCHORS:
        candidates = grouped[name]
        identities = {item.identity_id for item in candidates}
        expressions = {item.expression for item in candidates}
        cameras = {item.camera_id for item in candidates}
        if len(identities) < 3 or len(expressions) < 4 or len(cameras) < 3:
            raise ValueError(
                f"{name} lacks 3-identity/4-expression/3-camera proposal coverage"
            )
        counts = Counter(item.multiface_vertex_id for item in candidates)
        maximum = max(counts.values())
        finalists = [vertex_id for vertex_id, count in counts.items() if count == maximum]
        vertex_id = min(
            finalists,
            key=lambda candidate: (
                sorted(
                    item.distance
                    for item in candidates
                    if item.multiface_vertex_id == candidate
                )[counts[candidate] // 2],
                candidate,
            ),
        )
        mediapipe_indices = Counter(item.mediapipe_index for item in candidates)
        mediapipe_index = min(
            index
            for index, count in mediapipe_indices.items()
            if count == max(mediapipe_indices.values())
        )
        correspondences.append(
            AnchorCorrespondence(
                semantic_name=name,
                mediapipe_index=mediapipe_index,
                multiface_vertex_id=vertex_id,
            )
        )
    return SemanticAnchorMapping(
        schema_version="nana-semantic-anchor-map/1.0.0",
        mapping_revision=mapping_revision,
        source_topology="multiface-tracked-mesh-7306",
        mediapipe_version=mediapipe_version,
        mediapipe_model_sha256=mediapipe_model_sha256,
        status="pending",
        correspondences=correspondences,
        review=None,
        note=(
            "MediaPipe-assisted candidate only. Human overlay review must create a separate "
            "approved mapping before training."
        ),
    )


Matrix3 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
]
Matrix4 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
]


class StageAViewRecord(StageAModel):
    camera_id: str = Field(min_length=1)
    image: FileReference
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    face_bbox_xyxy: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    intrinsics: Matrix3
    camera_to_capture: Matrix4
    landmarks_2d_pixels: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=16, max_length=16)
    landmark_confidence: list[tuple[FiniteFloat, FiniteFloat]] = Field(
        min_length=16, max_length=16
    )
    canonical_geometry: list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]] = Field(
        min_length=16, max_length=16
    )
    geometry_confidence: list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]] = Field(
        min_length=16, max_length=16
    )
    pose: tuple[
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
    ]
    pose_confidence: tuple[
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
    ]
    visibility: int = Field(ge=0, le=2)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        left, top, right, bottom = self.face_bbox_xyxy
        if right <= left or bottom <= top:
            raise ValueError("face bbox must have positive area")
        if any(not 0.0 <= value <= 1.0 for pair in self.landmark_confidence for value in pair):
            raise ValueError("landmark confidence must remain in [0, 1]")
        if any(not 0.0 <= value <= 1.0 for row in self.geometry_confidence for value in row):
            raise ValueError("geometry confidence must remain in [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in self.pose_confidence):
            raise ValueError("pose confidence must remain in [0, 1]")
        return self


class StageAFrameGroup(StageAModel):
    record_id: str = Field(min_length=1)
    identity_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    timestamp_ns: int = Field(ge=0)
    sequence_index: int = Field(ge=0)
    source_fps: FiniteFloat = Field(ge=15.0, le=30.0)
    effective_fps: FiniteFloat = Field(ge=4.5, le=30.0)
    views: list[StageAViewRecord] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_views(self) -> Self:
        camera_ids = [view.camera_id for view in self.views]
        if len(set(camera_ids)) != 3:
            raise ValueError("Stage A groups require three distinct cameras")
        return self

    @classmethod
    def load_jsonl(cls, path: Path) -> list[Self]:
        records = [
            cls.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"Stage A record shard is empty: {path}")
        return records


@dataclass(frozen=True, slots=True)
class StageASample:
    images: Tensor
    targets: dict[str, Tensor]
    label_confidence: dict[str, Tensor]
    camera_intrinsics: Tensor
    camera_to_capture: Tensor
    sample_id: str
    identity_id: str
    sequence_id: str
    expression: str
    timestamp_ns: int


def _crop_geometry(
    image: Tensor,
    view: StageAViewRecord,
    *,
    output_height: int,
    output_width: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    left, top, right, bottom = view.face_bbox_xyxy
    side = max(right - left, bottom - top) * 1.25
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    crop_left = math.floor(center_x - side * 0.5)
    crop_top = math.floor(center_y - side * 0.5)
    crop_side = max(1, math.ceil(side))
    cropped = crop(image, crop_top, crop_left, crop_side, crop_side)
    resized = resize(cropped, [output_height, output_width], antialias=True)

    scale_x = output_width / crop_side
    scale_y = output_height / crop_side
    intrinsics = torch.tensor(view.intrinsics, dtype=torch.float32)
    intrinsics[0, 0] *= scale_x
    intrinsics[1, 1] *= scale_y
    intrinsics[0, 2] = (intrinsics[0, 2] - crop_left) * scale_x
    intrinsics[1, 2] = (intrinsics[1, 2] - crop_top) * scale_y

    landmarks = torch.tensor(view.landmarks_2d_pixels, dtype=torch.float32)
    landmarks[:, 0] = ((landmarks[:, 0] - crop_left) * scale_x) / output_width * 2.0 - 1.0
    landmarks[:, 1] = ((landmarks[:, 1] - crop_top) * scale_y) / output_height * 2.0 - 1.0
    inside = (landmarks.abs() <= 1.0).all(dim=-1, keepdim=True)
    landmark_confidence = torch.tensor(view.landmark_confidence, dtype=torch.float32)
    landmark_confidence = landmark_confidence * inside
    return resized, intrinsics, landmarks, landmark_confidence


class MultifaceStageADataset(Dataset[StageASample]):
    def __init__(self, config: ExperimentConfig, *, split: str) -> None:
        if config.data.manifest is None or config.data.anchor_mapping is None:
            raise ValueError("Multiface Stage A requires manifest and anchor mapping paths")
        self._config = config
        self._manifest_path = config.data.manifest.resolve()
        manifest = DatasetManifest.load(self._manifest_path)
        manifest.verify_files(self._manifest_path)
        if manifest.schema_version != "ntp-dataset/3.0.0":
            raise ValueError("Multiface Stage A requires ntp-dataset/3.0.0")
        if manifest.usage_tier != "noncommercial-research":
            raise ValueError("Multiface Stage A requires noncommercial-research data")
        SemanticAnchorMapping.load(config.data.anchor_mapping.resolve(), require_approved=True)
        if split not in manifest.splits:
            raise ValueError(f"manifest has no {split!r} split")
        split_identities = set(manifest.splits[split].identities)
        split_cameras = set(manifest.splits[split].devices)
        camera_sets = {
            name: set(item.devices)
            for name, item in manifest.splits.items()
            if name in {"train", "validation", "test"}
        }
        if any(
            camera_sets[left] & camera_sets[right]
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        ):
            raise ValueError("Stage A train, validation, and test cameras must be disjoint")

        train_identities = sorted(manifest.splits["train"].identities)
        if len(train_identities) > config.model.identity_classes:
            raise ValueError("model.identity_classes is smaller than Stage A train identities")
        identity_indices = {identity: index for index, identity in enumerate(train_identities)}
        all_records = [
            record
            for reference in manifest.record_files
            for record in StageAFrameGroup.load_jsonl(
                manifest.resolve(self._manifest_path, reference)
            )
        ]
        seen_ids: set[str] = set()
        selected: list[StageAFrameGroup] = []
        for record in all_records:
            if record.record_id in seen_ids:
                raise ValueError(f"duplicate Stage A record ID: {record.record_id}")
            seen_ids.add(record.record_id)
            if record.identity_id not in split_identities:
                continue
            if {view.camera_id for view in record.views} != split_cameras:
                continue
            if split == "train" and not 4.5 <= record.effective_fps <= 5.5:
                raise ValueError("Stage A train records must be sampled at approximately 5 FPS")
            if split != "train" and record.effective_fps not in {15.0, 30.0}:
                raise ValueError("Stage A validation/test records must preserve 15 or 30 FPS")
            selected.append(record)
        if not selected:
            raise ValueError(f"no usable Multiface Stage A records remain in split {split!r}")
        selected.sort(
            key=lambda record: (
                record.identity_id,
                record.session_id,
                record.expression,
                record.timestamp_ns,
                record.sequence_index,
            )
        )
        previous: dict[tuple[str, str, str], tuple[int, int]] = {}
        for record in selected:
            key = (record.identity_id, record.session_id, record.expression)
            if key in previous:
                timestamp, sequence = previous[key]
                if record.timestamp_ns <= timestamp or record.sequence_index <= sequence:
                    raise ValueError("Stage A sequence timestamps and indices must increase")
            previous[key] = (record.timestamp_ns, record.sequence_index)
        self._records = selected
        self._identity_indices = identity_indices
        self._verified_images: set[Path] = set()

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> StageASample:
        record = self._records[index]
        images: list[Tensor] = []
        intrinsics: list[Tensor] = []
        landmarks: list[Tensor] = []
        landmark_confidence: list[Tensor] = []
        geometry: list[Tensor] = []
        geometry_confidence: list[Tensor] = []
        poses: list[Tensor] = []
        pose_confidence: list[Tensor] = []
        visibility: list[int] = []
        camera_to_capture: list[Tensor] = []
        for view in record.views:
            image_path = resolve_image_uri(self._manifest_path, view.image.path)
            if image_path not in self._verified_images:
                actual = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if actual != view.image.sha256:
                    raise ValueError(f"Stage A image digest mismatch: {image_path}")
                self._verified_images.add(image_path)
            image = decode_image(str(image_path), mode=ImageReadMode.RGB).to(torch.float32) / 255.0
            if image.shape[1:] != (view.height, view.width):
                raise ValueError(f"Stage A RGB dimensions mismatch: {record.record_id}")
            processed, adjusted_intrinsics, points, point_confidence = _crop_geometry(
                image,
                view,
                output_height=self._config.model.input_height,
                output_width=self._config.model.input_width,
            )
            images.append(processed)
            intrinsics.append(adjusted_intrinsics)
            landmarks.append(points)
            landmark_confidence.append(point_confidence)
            geometry.append(torch.tensor(view.canonical_geometry, dtype=torch.float32))
            geometry_confidence.append(
                torch.tensor(view.geometry_confidence, dtype=torch.float32)
            )
            poses.append(torch.tensor(view.pose, dtype=torch.float32))
            pose_confidence.append(torch.tensor(view.pose_confidence, dtype=torch.float32))
            visibility.append(view.visibility)
            camera_to_capture.append(
                torch.tensor(view.camera_to_capture, dtype=torch.float32)
            )
        identity_index = self._identity_indices.get(record.identity_id, -1)
        identity_weight = 1.0 if identity_index >= 0 else 0.0
        return StageASample(
            images=torch.stack(images),
            targets={
                "landmarks": torch.stack(landmarks),
                "canonical_geometry": torch.stack(geometry),
                "pose": torch.stack(poses),
                "visibility": torch.tensor(visibility, dtype=torch.long),
                "identity": torch.full((3,), identity_index, dtype=torch.long),
            },
            label_confidence={
                "landmarks": torch.stack(landmark_confidence),
                "canonical_geometry": torch.stack(geometry_confidence),
                "pose": torch.stack(pose_confidence),
                "visibility": torch.ones(3, 1),
                "identity": torch.full((3, 1), identity_weight),
            },
            camera_intrinsics=torch.stack(intrinsics),
            camera_to_capture=torch.stack(camera_to_capture),
            sample_id=record.record_id,
            identity_id=record.identity_id,
            sequence_id=record.session_id,
            expression=record.expression,
            timestamp_ns=record.timestamp_ns,
        )


def collate_multiview(samples: list[StageASample]) -> MultiViewTrackingBatch:
    return MultiViewTrackingBatch(
        images=torch.stack([sample.images for sample in samples]),
        targets={
            name: torch.stack([sample.targets[name] for sample in samples])
            for name in samples[0].targets
        },
        label_confidence={
            name: torch.stack([sample.label_confidence[name] for sample in samples])
            for name in samples[0].label_confidence
        },
        camera_intrinsics=torch.stack([sample.camera_intrinsics for sample in samples]),
        camera_to_capture=torch.stack([sample.camera_to_capture for sample in samples]),
        sample_ids=tuple(sample.sample_id for sample in samples),
        identity_ids=tuple(sample.identity_id for sample in samples),
        sequence_ids=tuple(sample.sequence_id for sample in samples),
        expressions=tuple(sample.expression for sample in samples),
        timestamps_ns=torch.tensor(
            [sample.timestamp_ns for sample in samples], dtype=torch.int64
        ),
    )


def create_multiface_loader(
    config: ExperimentConfig,
    *,
    split: str,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader[MultiViewTrackingBatch]:
    dataset = MultifaceStageADataset(config, split=split)
    generator = torch.Generator().manual_seed(config.training.seed + seed_offset)
    multiprocessing = config.data.executor == "multiprocessing"
    return cast(
        DataLoader[MultiViewTrackingBatch],
        DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=collate_multiview,
            num_workers=config.data.workers if multiprocessing else 0,
            prefetch_factor=config.data.buffersize if multiprocessing else None,
        ),
    )


def validate_stage_a_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = DatasetManifest.load(manifest_path)
    manifest.verify_files(manifest_path)
    if manifest.schema_version != "ntp-dataset/3.0.0":
        raise ValueError("Stage A validation requires ntp-dataset/3.0.0")
    if manifest.anchor_mapping is None:
        raise ValueError("Stage A manifest is missing anchor_mapping")
    mapping_path = manifest.resolve(manifest_path, manifest.anchor_mapping)
    mapping = SemanticAnchorMapping.load(mapping_path, require_approved=True)
    records = [
        record
        for reference in manifest.record_files
        for record in StageAFrameGroup.load_jsonl(manifest.resolve(manifest_path, reference))
    ]
    seen: set[str] = set()
    split_counts: dict[str, int] = {}
    camera_sets = {name: set(split.devices) for name, split in manifest.splits.items()}
    if any(
        camera_sets[left] & camera_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("Stage A split cameras must be pairwise disjoint")
    for split_name, split in manifest.splits.items():
        identities = set(split.identities)
        cameras = set(split.devices)
        selected = [
            record
            for record in records
            if record.identity_id in identities
            and {view.camera_id for view in record.views} == cameras
        ]
        if not selected:
            raise ValueError(f"Stage A split has no matching records: {split_name}")
        for record in selected:
            if record.record_id in seen:
                raise ValueError(f"Stage A record leaks across splits: {record.record_id}")
            seen.add(record.record_id)
            if split_name == "train" and not 4.5 <= record.effective_fps <= 5.5:
                raise ValueError("Stage A training records must be sampled at approximately 5 FPS")
            if split_name != "train" and record.effective_fps not in {15.0, 30.0}:
                raise ValueError("Stage A validation/test must retain continuous 15 or 30 FPS")
        split_counts[split_name] = len(selected)
    identity_count = len(
        {identity for split in manifest.splits.values() for identity in split.identities}
    )
    return {
        "schema_version": manifest.schema_version,
        "usage_tier": manifest.usage_tier,
        "pipeline_stage": manifest.pipeline_stage,
        "mapping_revision": mapping.mapping_revision,
        "identity_count": identity_count,
        "classification": "research-cohort" if identity_count >= 8 else "pipeline-pilot",
        "split_record_counts": split_counts,
    }


def create_stage_a_configs(
    *,
    manifest_path: Path,
    anchor_mapping_path: Path,
    recipe_path: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Create digest-pinned single-view and multiview configs after every gate passes."""

    from nana_tracking.training.recipe import TrainingRecipe

    manifest = DatasetManifest.load(manifest_path)
    manifest.verify_files(manifest_path)
    if manifest.usage_tier != "noncommercial-research":
        raise ValueError("Stage A config generation requires a research manifest")
    SemanticAnchorMapping.load(anchor_mapping_path, require_approved=True)
    TrainingRecipe.load(recipe_path)
    if manifest.license_registry is None:
        raise ValueError("Stage A manifest requires a license registry")
    registry_path = manifest.resolve(manifest_path, manifest.license_registry)
    common: dict[str, object] = {
        "data": {
            "dataset": "multiface",
            "usage_tier": "noncommercial-research",
            "manifest": manifest_path,
            "anchor_mapping": anchor_mapping_path,
            "samples": 1,
            "batch_size": 8,
            "executor": "inline",
            "workers": 0,
            "buffersize": 2,
            "require_complete_basic": False,
        },
        "model": {
            "name": "face_basic",
            "input_channels": 3,
            "input_height": 128,
            "input_width": 128,
            "hidden_dims": 64,
            "rig_dims": 36,
            "pose_dims": 7,
            "landmark_count": 16,
            "identity_classes": len(manifest.splits["train"].identities),
            "identity_dims": 16,
            "dropout": 0.1,
        },
        "training": {
            "stage": "real-geometry-pretrain",
            "seed": 17,
            "max_steps": 3000,
            "learning_rate": 0.0003,
            "minimum_learning_rate": 0.000003,
            "warmup_steps": 200,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "gradient_accumulation_steps": 2,
            "device": "cuda",
            "amp": True,
            "shuffle": True,
            "deterministic": True,
            "validation_interval_steps": 100,
            "checkpoint_interval_steps": 500,
            "rig_loss_weight": 0.0,
            "pose_loss_weight": 0.0,
            "landmark_loss_weight": 1.0,
            "visibility_loss_weight": 0.25,
            "confidence_loss_weight": 0.0,
            "identity_adversary_weight": 0.1,
            "mirror_consistency_weight": 0.0,
            "canonical_geometry_loss_weight": 1.0,
            "pose_rotation_loss_weight": 1.0,
            "pose_translation_loss_weight": 1.0,
            "reprojection_loss_weight": 1.0,
        },
        "evaluation": {"atol": 0.00001, "rtol": 0.0001},
        "export": {
            "opset": 18,
            "model_family": "nana-face-basic-stage-a-research",
            "model_version": "1.0.0-research",
            "smoke_only": False,
            "enabled": False,
        },
        "reproducibility": {
            "data_revision": manifest.data_revision,
            "ntp_schema_revision": manifest.ntp_schema_revision,
            "signal_registry_revision": manifest.signal_registry_revision,
            "normalization_revision": manifest.normalization_revision,
            "calibration_revision": manifest.calibration_revision,
            "feature_revision": manifest.feature_revision,
            "geometry_topology_revision": "multiface-semantic-anchors/1.0.0-research",
            "manifest_digest": sha256_file(manifest_path),
            "license_registry_digest": sha256_file(registry_path),
            "anchor_mapping_digest": sha256_file(anchor_mapping_path),
            "training_recipe_digest": sha256_file(recipe_path),
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, geometry_weight, pose_weight in (
        ("single-view", 0.0, 0.0),
        ("multiview", 0.5, 0.5),
    ):
        payload = cast(dict[str, object], json.loads(json.dumps(common, default=str)))
        training = cast(dict[str, object], payload["training"])
        training["multiview_geometry_loss_weight"] = geometry_weight
        training["multiview_pose_loss_weight"] = pose_weight
        reproducibility = cast(dict[str, object], payload["reproducibility"])
        reproducibility["output_dir"] = f"runs/face-basic-stage-a-{name}"
        config = ExperimentConfig.model_validate(payload)
        path = output_directory / f"face-basic-stage-a-{name}.yaml"
        save_config(config, path)
        paths.append(path)
    return paths[0], paths[1]


def _relative_reference(path: Path, *, parent: Path) -> FileReference:
    return FileReference(
        path=Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix(),
        sha256=sha256_file(path),
    )


def build_stage_a_manifest(
    *,
    record_files: list[Path],
    split_plan_path: Path,
    label_catalog_path: Path,
    license_registry_path: Path,
    anchor_mapping_path: Path,
    recipe_path: Path,
    data_revision: str,
    output_path: Path,
    license_record_id: str = "research-multiface-cc-by-nc",
) -> DatasetManifest:
    """Freeze reviewed Stage A records behind all research admission digests."""

    from nana_tracking.training.recipe import TrainingRecipe

    if not record_files:
        raise ValueError("Stage A manifest requires at least one record shard")
    registry = LicenseRegistry.load(license_registry_path)
    registry.verify_local_license_texts(license_registry_path)
    admitted = registry.admit(
        [license_record_id],
        stage="research-model-training",
        production=False,
        usage_tier="noncommercial-research",
    )
    license_record = admitted[0]
    SemanticAnchorMapping.load(anchor_mapping_path, require_approved=True)
    TrainingRecipe.load(recipe_path)
    split_plan = MultifaceSplitPlan.model_validate_json(
        split_plan_path.read_text(encoding="utf-8")
    )
    record_references: list[RecordFile] = []
    for path in record_files:
        records = StageAFrameGroup.load_jsonl(path)
        record_references.append(
            RecordFile(
                path=Path(
                    os.path.relpath(path.resolve(), output_path.parent.resolve())
                ).as_posix(),
                sha256=sha256_file(path),
                record_count=len(records),
            )
        )
    manifest = DatasetManifest(
        schema_version="ntp-dataset/3.0.0",
        capture_schema_version="ntp-capture/1.0.0",
        data_revision=data_revision,
        digest="0" * 64,
        ntp_schema_revision="ntp/1.0",
        signal_registry_revision="ntp-signals/1.0.0",
        normalization_revision="ntp-normalization/1.0.0",
        calibration_revision="ntp-calibration/1.0.0",
        feature_revision="ntp-features/1.0.0",
        pipeline_stage="research-model-training",
        usage_tier="noncommercial-research",
        label_catalog=_relative_reference(
            label_catalog_path, parent=output_path.parent
        ),
        license_registry=_relative_reference(
            license_registry_path, parent=output_path.parent
        ),
        license_record_ids=[license_record_id],
        anchor_mapping=_relative_reference(
            anchor_mapping_path, parent=output_path.parent
        ),
        training_recipe=_relative_reference(recipe_path, parent=output_path.parent),
        split_plan=_relative_reference(split_plan_path, parent=output_path.parent),
        record_files=record_references,
        teacher_sources=[
            TeacherSource(
                source_id="multiface-tracked-mesh-headpose",
                source_type="multiview_pose",
                version=license_record.version,
                license_id=license_record_id,
            )
        ],
        license_reviews=[
            LicenseReview(
                license_id=license_record_id,
                scope="third_party",
                status="approved",
                evidence=license_record.evidence,
                permissions=LicensePermissions(
                    collection=True,
                    distillation=False,
                    pseudo_labeling=False,
                    commercial_training=False,
                    noncommercial_research_training=True,
                    research_derivative_labels=True,
                    research_checkpoint_local_use=True,
                ),
            )
        ],
        synchronization=SynchronizationPolicy(
            max_teacher_skew_ns=1,
            max_depth_skew_ns=1,
        ),
        splits=cast(dict[str, SplitManifest], split_plan.splits),
        smoke_only=False,
    )
    manifest = manifest.model_copy(update={"digest": dataset_digest(manifest)})
    manifest.save(output_path)
    return manifest


class MultifaceRemoteAsset(StageAModel):
    identity_id: str
    filename: str
    url: str
    bytes: int = Field(gt=0)


class MultifacePreflightPlan(StageAModel):
    schema_version: Literal["nana-multiface-preflight/1.0.0"] = (
        "nana-multiface-preflight/1.0.0"
    )
    license_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_record_id: str
    budget_bytes: int = Field(gt=0)
    selected_bytes: int = Field(gt=0)
    selected_identities: list[str] = Field(min_length=5, max_length=10)
    expressions: list[str] = Field(min_length=4, max_length=4)
    assets: list[MultifaceRemoteAsset] = Field(min_length=1)
    classification: Literal["pipeline-pilot", "research-cohort"]


class MultifaceCamera(StageAModel):
    camera_id: str = Field(min_length=1)
    camera_to_capture: Matrix4


class MultifaceSplitPlan(StageAModel):
    schema_version: Literal["nana-multiface-splits/1.0.0"] = (
        "nana-multiface-splits/1.0.0"
    )
    seed: Literal[17] = 17
    camera_forward_axis: Literal["+z"] = "+z"
    classification: Literal["pipeline-pilot", "research-cohort"]
    splits: dict[Literal["train", "validation", "test"], SplitManifest]
    selected_camera_yaw_degrees: dict[str, dict[str, float]]


def _camera_yaw(camera: MultifaceCamera) -> float:
    forward_x = camera.camera_to_capture[0][2]
    forward_z = camera.camera_to_capture[2][2]
    return math.degrees(math.atan2(forward_x, forward_z))


def build_multiface_split_plan(
    identities: Iterable[str],
    cameras: Iterable[MultifaceCamera],
) -> MultifaceSplitPlan:
    unique_identities = sorted(set(identities))
    if not 5 <= len(unique_identities) <= 10:
        raise ValueError("Stage A split requires between five and ten identities")
    ordered = sorted(
        unique_identities,
        key=lambda identity: hashlib.sha256(f"17:{identity}".encode()).hexdigest(),
    )
    if len(ordered) >= 8:
        assignments = {
            "validation": ordered[:1],
            "train": ordered[1:-2],
            "test": ordered[-2:],
        }
        classification: Literal["pipeline-pilot", "research-cohort"] = "research-cohort"
    else:
        assignments = {
            "validation": ordered[:1],
            "train": ordered[1:-1],
            "test": ordered[-1:],
        }
        classification = "pipeline-pilot"
    available = {camera.camera_id: _camera_yaw(camera) for camera in cameras}
    if len(available) < 9:
        raise ValueError("Stage A needs at least nine distinct calibrated cameras")
    targets = {
        "train": (-30.0, 0.0, 30.0),
        "validation": (-45.0, 15.0, 45.0),
        "test": (-60.0, -15.0, 60.0),
    }
    used: set[str] = set()
    selected: dict[str, dict[str, float]] = {}
    for split_name in ("train", "validation", "test"):
        chosen: dict[str, float] = {}
        for target in targets[split_name]:
            candidates = [
                (abs(yaw - target), camera_id, yaw)
                for camera_id, yaw in available.items()
                if camera_id not in used
            ]
            if not candidates:
                raise ValueError("camera selection exhausted before all split triplets")
            _distance, camera_id, yaw = min(candidates)
            used.add(camera_id)
            chosen[camera_id] = yaw
        selected[split_name] = chosen
    return MultifaceSplitPlan(
        classification=classification,
        splits={
            split_name: SplitManifest(
                identities=sorted(assignments[split_name]),
                devices=sorted(selected[split_name]),
            )
            for split_name in ("train", "validation", "test")
        },
        selected_camera_yaw_degrees=selected,
    )


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = next((value for key, value in attrs if key == "href"), None)
        if href:
            self.links.append(href)


def _remote_size(url: str) -> int:
    request = Request(url, method="HEAD", headers={"User-Agent": "NanaTracking-stage-a/1.0"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed reviewed HTTPS origin
        raw = response.headers.get("Content-Length")
    if raw is None or int(raw) <= 0:
        raise ValueError(f"remote asset has no positive Content-Length: {url}")
    return int(raw)


def _identity_assets(identity_id: str, expressions: tuple[str, ...]) -> list[MultifaceRemoteAsset]:
    index_url = urljoin(_MULTIFACE_ROOT, f"{identity_id}/index.html")
    request = Request(
        index_url,
        headers={"User-Agent": "NanaTracking-stage-a/1.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed reviewed HTTPS origin
        html = response.read().decode("utf-8")
    parser = _LinkParser()
    parser.feed(html)
    assets: list[MultifaceRemoteAsset] = []
    for href in parser.links:
        filename = href.rsplit("/", 1)[-1]
        lower = filename.lower()
        if filename in {"CHECKSUM", "index.html"}:
            continue
        metadata = "metadata" in lower
        supervised_asset = any(
            marker in lower for marker in ("images", "tracked_mesh", "headpose")
        )
        expression_match = any(expression in filename for expression in expressions)
        if not metadata and not (supervised_asset and expression_match):
            continue
        if any(marker in lower for marker in ("audio", "unwrapped_uv", "texture")):
            continue
        url = urljoin(index_url, href)
        assets.append(
            MultifaceRemoteAsset(
                identity_id=identity_id,
                filename=filename,
                url=url,
                bytes=_remote_size(url),
            )
        )
    if not assets:
        raise ValueError(f"no selected Multiface assets found for identity {identity_id}")
    return assets


def preflight_multiface(
    *,
    registry_path: Path,
    identity_priority: tuple[str, ...] = DEFAULT_IDENTITY_PRIORITY,
    expressions: tuple[str, ...] = CORE_EXPRESSIONS,
    budget_gib: int = 200,
    license_record_id: str = "research-multiface-cc-by-nc",
) -> MultifacePreflightPlan:
    """Admit first, then query exact archive sizes without downloading dataset bytes."""

    registry = LicenseRegistry.load(registry_path)
    registry.verify_local_license_texts(registry_path)
    registry.admit(
        [license_record_id],
        stage="research-model-training",
        production=False,
        usage_tier="noncommercial-research",
    )
    budget_bytes = budget_gib * 1024**3
    selected: list[MultifaceRemoteAsset] = []
    selected_identities: list[str] = []
    selected_bytes = 0
    for identity_id in identity_priority:
        assets = _identity_assets(identity_id, expressions)
        identity_bytes = sum(asset.bytes for asset in assets)
        if selected_bytes + identity_bytes > budget_bytes:
            continue
        selected.extend(assets)
        selected_identities.append(identity_id)
        selected_bytes += identity_bytes
        if len(selected_identities) == 10:
            break
    if len(selected_identities) < 5:
        raise ValueError("200 GiB preflight cannot fit the minimum five Multiface identities")
    return MultifacePreflightPlan(
        license_registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        license_record_id=license_record_id,
        budget_bytes=budget_bytes,
        selected_bytes=selected_bytes,
        selected_identities=selected_identities,
        expressions=list(expressions),
        assets=selected,
        classification=(
            "research-cohort" if len(selected_identities) >= 8 else "pipeline-pilot"
        ),
    )


def save_stage_a_model(model: StageAModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
