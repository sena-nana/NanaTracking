"""Versioned dataset manifest, provenance, and identity-safe split validation."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    schema_version: Literal["ntp-dataset/2.0.0"]
    capture_schema_version: Literal["ntp-capture/1.0.0"]
    data_revision: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    calibration_revision: str = Field(min_length=1)
    feature_revision: str = Field(min_length=1)
    label_catalog: FileReference
    record_files: list[RecordFile] = Field(min_length=1)
    teacher_sources: list[TeacherSource] = Field(min_length=1)
    license_reviews: list[LicenseReview] = Field(min_length=1)
    synchronization: SynchronizationPolicy
    splits: dict[str, SplitManifest]
    smoke_only: bool

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
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
            if review.status != "approved" or not all(
                (
                    permissions.collection,
                    permissions.distillation,
                    permissions.pseudo_labeling,
                    permissions.commercial_training,
                )
            ):
                raise ValueError(
                    f"teacher {source.source_id!r} is not approved for the complete training use"
                )
        return self

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
        for reference in references:
            path = self.resolve(manifest_path, reference)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != reference.sha256:
                raise ValueError(
                    f"digest mismatch for {reference.path}: "
                    f"expected {reference.sha256}, got {actual}"
                )
        actual_dataset_digest = dataset_digest(self)
        if actual_dataset_digest != self.digest:
            raise ValueError(
                f"dataset digest mismatch: expected {self.digest}, got {actual_dataset_digest}"
            )


def dataset_digest(manifest: DatasetManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"digest"})
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()
