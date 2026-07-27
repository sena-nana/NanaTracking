"""Versioned dataset manifest, provenance, and identity-safe split validation."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nana_tracking.governance import ArtifactUsageTier, PipelineStage


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileReference(ManifestModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RecordFile(FileReference):
    record_count: int = Field(gt=0)


class SplitManifest(ManifestModel):
    identities: list[str] = Field(min_length=1)
    sessions: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    camera_ids: list[str] = Field(default_factory=list)


class LicensePermissions(ManifestModel):
    collection: bool
    distillation: bool
    pseudo_labeling: bool
    commercial_training: bool


class LicenseReview(ManifestModel):
    license_id: str = Field(min_length=1)
    scope: Literal["first_party", "third_party", "synthetic"]
    status: Literal["approved", "rejected", "pending"]
    evidence: str = Field(min_length=1)
    permissions: LicensePermissions


TeacherSourceType = Literal[
    "truedepth",
    "offline_face",
    "multiview_pose",
    "human_review",
    "synthetic",
]


class TeacherSource(ManifestModel):
    source_id: str = Field(min_length=1)
    source_type: TeacherSourceType
    version: str = Field(min_length=1)
    license_id: str = Field(min_length=1)


class SynchronizationPolicy(ManifestModel):
    max_teacher_skew_ns: int = Field(gt=0, le=5_000_000)
    max_depth_skew_ns: int = Field(gt=0, le=2_000_000)
    require_monotonic_timestamps: bool = True
    require_increasing_sequence: bool = True


class DatasetManifest(ManifestModel):
    schema_version: Literal["ntp-dataset/2.0.0", "ntp-dataset/3.0.0"]
    capture_schema_version: Literal[
        "ntp-capture/1.0.0",
        "nana-stage-a-materialization/1.0.0",
    ]
    data_revision: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    calibration_revision: str = Field(min_length=1)
    feature_revision: str = Field(min_length=1)
    pipeline_stage: PipelineStage = "base-model-training"
    usage_tier: ArtifactUsageTier | None = None
    label_catalog: FileReference
    license_registry: FileReference | None = None
    license_record_ids: list[str] = Field(default_factory=list)
    anchor_mapping: FileReference | None = None
    teacher_model: FileReference | None = None
    calibration_bundle: FileReference | None = None
    quality_profile: FileReference | None = None
    overlay_review: FileReference | None = None
    training_recipe: FileReference | None = None
    split_plan: FileReference | None = None
    record_files: list[RecordFile] = Field(min_length=1)
    teacher_sources: list[TeacherSource] = Field(min_length=1)
    license_reviews: list[LicenseReview] = Field(min_length=1)
    synchronization: SynchronizationPolicy
    splits: dict[str, SplitManifest]
    smoke_only: bool

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version == "ntp-dataset/2.0.0":
            inferred: ArtifactUsageTier = "synthetic-smoke" if self.smoke_only else "commercial"
            if self.usage_tier is not None and self.usage_tier != inferred:
                raise ValueError("v2 usage_tier conflicts with smoke_only")
            self.usage_tier = inferred
        elif self.usage_tier is None:
            raise ValueError("v3 manifests require usage_tier")
        if self.usage_tier == "commercial" and self.smoke_only:
            raise ValueError("commercial manifests cannot be smoke-only")
        required = {"train", "validation", "test"}
        missing = required.difference(self.splits)
        if missing:
            raise ValueError(f"missing required splits: {sorted(missing)}")

        owners: dict[str, str] = {}
        session_owners: dict[str, str] = {}
        for split_name, split in self.splits.items():
            for identity in split.identities:
                previous = owners.setdefault(identity, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"identity {identity!r} appears in both {previous!r} and {split_name!r}"
                    )
            for session in split.sessions:
                previous = session_owners.setdefault(session, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"session {session!r} appears in both {previous!r} and {split_name!r}"
                    )

        license_by_id = {review.license_id: review for review in self.license_reviews}
        if len(license_by_id) != len(self.license_reviews):
            raise ValueError("license review IDs must be unique")
        source_ids = {source.source_id for source in self.teacher_sources}
        if len(source_ids) != len(self.teacher_sources):
            raise ValueError("teacher source IDs must be unique")
        for source in self.teacher_sources:
            review = license_by_id.get(source.license_id)
            if review is None:
                raise ValueError(f"teacher {source.source_id!r} has no license review")
            permissions = review.permissions
            complete = all(
                (
                    permissions.collection,
                    permissions.distillation,
                    permissions.pseudo_labeling,
                    permissions.commercial_training,
                )
            )
            if review.status != "approved" or not complete:
                raise ValueError(
                    f"teacher {source.source_id!r} is not approved for the complete training use"
                )
        if self.usage_tier != "synthetic-smoke" and (
            self.license_registry is None or not self.license_record_ids
        ):
            raise ValueError("non-smoke manifests require a license registry and admitted records")
        if self.license_record_ids and self.license_record_ids != sorted(
            set(self.license_record_ids)
        ):
            raise ValueError("license record IDs must be unique and increasing")
        if not {source.license_id for source in self.teacher_sources}.issubset(
            self.license_record_ids
        ):
            raise ValueError("every teacher license must be present in license_record_ids")

        test_devices = set(self.splits["test"].devices)
        development_devices = set(self.splits["train"].devices) | set(
            self.splits["validation"].devices
        )
        overlap = sorted(test_devices & development_devices)
        if overlap:
            raise ValueError(f"held-out test devices leak into development splits: {overlap}")
        if self.capture_schema_version == "nana-stage-a-materialization/1.0.0":
            self._validate_stage_a_contract()
        return self

    def _validate_stage_a_contract(self) -> None:
        if self.schema_version != "ntp-dataset/3.0.0":
            raise ValueError("Stage A materialization requires a commercial v3 manifest")
        if self.usage_tier != "commercial" or self.smoke_only:
            raise ValueError("Stage A materialization manifests must be commercial and non-smoke")
        required_refs = {
            "anchor_mapping": self.anchor_mapping,
            "teacher_model": self.teacher_model,
            "calibration_bundle": self.calibration_bundle,
            "quality_profile": self.quality_profile,
            "overlay_review": self.overlay_review,
            "training_recipe": self.training_recipe,
            "split_plan": self.split_plan,
        }
        missing_refs = sorted(
            name for name, reference in required_refs.items() if reference is None
        )
        if missing_refs:
            raise ValueError(f"Stage A manifest is missing required references: {missing_refs}")
        source_ids = {source.source_id for source in self.teacher_sources}
        expected_sources = {
            "mediapipe-face-landmarker-v1",
            "opencv-calibrated-geometry-v1",
        }
        if source_ids != expected_sources:
            raise ValueError(
                "Stage A teacher sources must be exactly MediaPipe landmarks and OpenCV geometry"
            )

        for dimension in ("identities", "sessions", "devices", "camera_ids"):
            owners: dict[str, str] = {}
            for split_name, split in self.splits.items():
                for value in getattr(split, dimension):
                    previous = owners.setdefault(value, split_name)
                    if previous != split_name:
                        raise ValueError(
                            f"{dimension[:-1]} {value!r} appears in both "
                            f"{previous!r} and {split_name!r}"
                        )
        for split_name, split in self.splits.items():
            for field in ("identities", "sessions", "devices", "camera_ids"):
                values = getattr(split, field)
                if not values:
                    raise ValueError(f"Stage A {split_name} split requires {field}")
                if len(values) != len(set(values)):
                    raise ValueError(f"Stage A {split_name} split contains duplicate {field}")
            if len(split.devices) < 3 or len(split.camera_ids) < 3:
                raise ValueError(
                    f"Stage A {split_name} split requires at least three devices and cameras"
                )

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def resolve(self, manifest_path: Path, reference: FileReference) -> Path:
        return (manifest_path.parent / reference.path).resolve()

    def verify_files(self, manifest_path: Path) -> None:
        references: list[FileReference] = [self.label_catalog, *self.record_files]
        if self.license_registry is not None:
            references.append(self.license_registry)
        if self.anchor_mapping is not None:
            references.append(self.anchor_mapping)
        if self.teacher_model is not None:
            references.append(self.teacher_model)
        if self.calibration_bundle is not None:
            references.append(self.calibration_bundle)
        if self.quality_profile is not None:
            references.append(self.quality_profile)
        if self.overlay_review is not None:
            references.append(self.overlay_review)
        if self.training_recipe is not None:
            references.append(self.training_recipe)
        if self.split_plan is not None:
            references.append(self.split_plan)
        for reference in references:
            path = self.resolve(manifest_path, reference)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != reference.sha256:
                raise ValueError(
                    f"digest mismatch for {reference.path}: "
                    f"expected {reference.sha256}, got {actual}"
                )
        if self.license_registry is not None:
            from nana_tracking.data.strategy import LicenseRegistry

            registry_path = self.resolve(manifest_path, self.license_registry)
            registry = LicenseRegistry.load(registry_path)
            registry.verify_local_license_texts(registry_path)
            registry.admit(
                self.license_record_ids,
                stage=self.pipeline_stage,
                production=self.usage_tier == "commercial",
                usage_tier=self.usage_tier,
            )
        actual_dataset_digest = dataset_digest(self)
        if actual_dataset_digest != self.digest:
            raise ValueError(
                f"dataset digest mismatch: expected {self.digest}, got {actual_dataset_digest}"
            )


def dataset_digest(manifest: DatasetManifest) -> str:
    excluded = {"digest"}
    if manifest.schema_version == "ntp-dataset/2.0.0":
        excluded.update(
            {
                "usage_tier",
                "anchor_mapping",
                "teacher_model",
                "calibration_bundle",
                "quality_profile",
                "overlay_review",
                "training_recipe",
                "split_plan",
            }
        )
    payload = manifest.model_dump(mode="json", exclude=excluded)
    if manifest.schema_version == "ntp-dataset/2.0.0":
        for split in payload["splits"].values():
            split.pop("camera_ids", None)
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()
