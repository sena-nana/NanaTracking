"""Commercial data admission, group-safe splitting, and frozen-F cache contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nana_tracking.data.manifest import FileReference, SplitManifest
from nana_tracking.data.schema import CaptureRecord

PipelineStage = Literal[
    "base-model-training",
    "expression-model-training",
    "teacher-labeling",
    "synthetic-rendering",
    "evaluation",
    "model-release",
]
SplitName = Literal["train", "validation", "test"]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class StrategyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CommercialPermissions(StrategyModel):
    commercial_training_allowed: bool
    model_distribution_allowed: bool
    redistribution_allowed: bool
    distillation_allowed: bool
    pseudo_labeling_allowed: bool
    derivative_labels_allowed: bool


class LicenseRecord(StrategyModel):
    record_id: str = Field(min_length=1)
    kind: Literal["dataset", "render-asset", "teacher-sdk", "first-party-capture"]
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    license_text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    review_status: Literal["approved", "pending", "rejected"]
    permissions: CommercialPermissions
    attribution_obligations: list[str] = Field(default_factory=list)
    share_alike_obligations: list[str] = Field(default_factory=list)
    personal_biometric_consent_basis: str = Field(min_length=1)
    allowed_pipeline_stages: set[PipelineStage] = Field(default_factory=set)
    prohibited_uses: list[str] = Field(min_length=1)
    evidence: str = Field(min_length=1)
    smoke_only: bool = False

    @model_validator(mode="after")
    def validate_approval_evidence(self) -> Self:
        if self.review_status == "approved" and self.license_text_sha256 is None:
            raise ValueError("approved license records require a pinned license-text digest")
        return self


class LicenseRegistry(StrategyModel):
    schema_version: Literal["nana-license-registry/1.0.0"] = "nana-license-registry/1.0.0"
    revision: str = Field(min_length=1)
    records: list[LicenseRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_records(self) -> Self:
        record_ids = [record.record_id for record in self.records]
        if record_ids != sorted(set(record_ids)):
            raise ValueError("license records must use unique increasing record IDs")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def admit(
        self,
        record_ids: Iterable[str],
        *,
        stage: PipelineStage,
        production: bool,
    ) -> list[LicenseRecord]:
        by_id = {record.record_id: record for record in self.records}
        requested = sorted(set(record_ids))
        if not requested:
            raise ValueError("license admission requires at least one record")
        admitted: list[LicenseRecord] = []
        for record_id in requested:
            record = by_id.get(record_id)
            if record is None:
                raise ValueError(f"license record is missing: {record_id}")
            if record.review_status != "approved":
                raise ValueError(f"license record is not approved: {record_id}")
            if stage not in record.allowed_pipeline_stages:
                raise ValueError(f"license record does not allow {stage}: {record_id}")
            if production and record.smoke_only:
                raise ValueError(f"smoke-only license record cannot enter production: {record_id}")
            if stage in {
                "base-model-training",
                "expression-model-training",
                "model-release",
            } and not (
                record.permissions.commercial_training_allowed
                and record.permissions.model_distribution_allowed
            ):
                raise ValueError(f"license record forbids commercial model use: {record_id}")
            if stage == "teacher-labeling" and not (
                record.permissions.distillation_allowed
                and record.permissions.pseudo_labeling_allowed
                and record.permissions.derivative_labels_allowed
            ):
                raise ValueError(f"license record forbids teacher-derived labels: {record_id}")
            admitted.append(record)
        return admitted

    def verify_local_license_texts(self, registry_path: Path) -> None:
        for record in self.records:
            if record.review_status != "approved":
                continue
            if record.license_text_sha256 is None:
                raise ValueError(f"approved license digest is missing: {record.record_id}")
            if "://" in record.license:
                continue
            license_path = (registry_path.parent / record.license).resolve()
            if not license_path.is_file():
                raise ValueError(f"approved license text is missing: {record.record_id}")
            actual = hashlib.sha256(license_path.read_bytes()).hexdigest()
            if actual != record.license_text_sha256:
                raise ValueError(f"approved license text digest drifted: {record.record_id}")


class ClipReference(StrategyModel):
    clip_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)


class ActorSplit(StrategyModel):
    actors: list[str] = Field(min_length=1)
    clips: list[str] = Field(min_length=1)


class ActorSplitManifest(StrategyModel):
    schema_version: Literal["nana-actor-splits/1.0.0"] = "nana-actor-splits/1.0.0"
    seed: int = Field(ge=0)
    splits: dict[SplitName, ActorSplit]

    @model_validator(mode="after")
    def validate_no_actor_or_clip_leakage(self) -> Self:
        required = {"train", "validation", "test"}
        if set(self.splits) != required:
            raise ValueError("actor split manifest requires train, validation, and test")
        actor_owner: dict[str, str] = {}
        clip_owner: dict[str, str] = {}
        for split_name, split in self.splits.items():
            for actor in split.actors:
                previous = actor_owner.setdefault(actor, split_name)
                if previous != split_name:
                    raise ValueError(f"actor {actor!r} leaks across splits")
            for clip in split.clips:
                previous = clip_owner.setdefault(clip, split_name)
                if previous != split_name:
                    raise ValueError(f"clip {clip!r} leaks across splits")
        return self


def split_clips_by_actor(
    clips: Iterable[ClipReference],
    *,
    seed: int,
    validation_actors: int,
    test_actors: int,
) -> ActorSplitManifest:
    clips_by_actor: dict[str, list[str]] = {}
    seen_clips: set[str] = set()
    for clip in clips:
        if clip.clip_id in seen_clips:
            raise ValueError(f"duplicate clip ID: {clip.clip_id}")
        seen_clips.add(clip.clip_id)
        clips_by_actor.setdefault(clip.actor_id, []).append(clip.clip_id)
    actors = sorted(
        clips_by_actor,
        key=lambda actor: hashlib.sha256(f"{seed}:{actor}".encode()).hexdigest(),
    )
    if validation_actors < 1 or test_actors < 1:
        raise ValueError("validation and test require at least one actor")
    if len(actors) <= validation_actors + test_actors:
        raise ValueError("actor split leaves no training identities")
    test = actors[:test_actors]
    validation = actors[test_actors : test_actors + validation_actors]
    train = actors[test_actors + validation_actors :]
    groups: dict[SplitName, list[str]] = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    return ActorSplitManifest(
        seed=seed,
        splits={
            split_name: ActorSplit(
                actors=sorted(split_actors),
                clips=sorted(clip for actor in split_actors for clip in clips_by_actor[actor]),
            )
            for split_name, split_actors in groups.items()
        },
    )


class CaptureSplitPlan(StrategyModel):
    schema_version: Literal["nana-capture-splits/1.0.0"] = "nana-capture-splits/1.0.0"
    seed: int = Field(ge=0)
    held_out_test_devices: list[str] = Field(min_length=1)
    splits: dict[SplitName, SplitManifest]


def split_captures(
    records: Iterable[CaptureRecord],
    *,
    seed: int,
    held_out_test_devices: set[str],
    validation_identities: int,
) -> CaptureSplitPlan:
    if not held_out_test_devices:
        raise ValueError("capture splitting requires a held-out test device")
    by_identity: dict[str, list[CaptureRecord]] = {}
    for record in records:
        by_identity.setdefault(record.identity_id, []).append(record)
    test = sorted(
        identity
        for identity, identity_records in by_identity.items()
        if any(record.device_id in held_out_test_devices for record in identity_records)
    )
    if not test:
        raise ValueError("no identity was recorded on a held-out test device")
    development = sorted(
        set(by_identity).difference(test),
        key=lambda identity: hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest(),
    )
    if validation_identities < 1 or len(development) <= validation_identities:
        raise ValueError("capture split leaves no train or validation identities")
    validation = development[:validation_identities]
    train = development[validation_identities:]
    assignments: dict[SplitName, list[str]] = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    splits: dict[SplitName, SplitManifest] = {}
    for split_name, identities in assignments.items():
        identity_records = [record for identity in identities for record in by_identity[identity]]
        splits[split_name] = SplitManifest(
            identities=sorted(identities),
            sessions=sorted({record.session_id for record in identity_records}),
            devices=sorted({record.device_id for record in identity_records}),
        )
    leaked = set(splits["test"].devices) & (
        set(splits["train"].devices) | set(splits["validation"].devices)
    )
    if leaked:
        raise ValueError(f"held-out devices also occur in development identities: {sorted(leaked)}")
    return CaptureSplitPlan(
        seed=seed,
        held_out_test_devices=sorted(held_out_test_devices),
        splits=splits,
    )


class FrozenFContract(StrategyModel):
    source_kind: Literal["frozen-f-predictions"] = "frozen-f-predictions"
    frozen: Literal[True] = True
    model_family: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    feature_revision: str = Field(min_length=1)


class ExpressionCacheManifest(StrategyModel):
    schema_version: Literal["nana-expression-cache/1.0.0"] = "nana-expression-cache/1.0.0"
    cache_revision: str = Field(min_length=1)
    cache_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dataset: str = Field(min_length=1)
    source_dataset_revision: str = Field(min_length=1)
    license_registry: FileReference
    license_record_ids: list[str] = Field(min_length=1)
    frozen_f: FrozenFContract
    parameter_signal_ids: list[int]
    shard_files: list[FileReference] = Field(min_length=1)
    actor_splits: ActorSplitManifest
    emotion_labels: list[str] = Field(min_length=2)
    label_source: str = Field(min_length=1)
    smoke_only: bool

    @model_validator(mode="after")
    def validate_parameter_contract(self) -> Self:
        if self.parameter_signal_ids != list(range(1, 37)):
            raise ValueError("expression caches require the complete ordered BasicSet 1..36")
        if len(self.emotion_labels) != len(set(self.emotion_labels)):
            raise ValueError("emotion labels must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def verify_files(self, manifest_path: Path) -> list[ExpressionCacheRecord]:
        root = manifest_path.parent
        for reference in [self.license_registry, *self.shard_files]:
            path = (root / reference.path).resolve()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != reference.sha256:
                raise ValueError(f"expression cache digest mismatch: {reference.path}")
        registry_path = (root / self.license_registry.path).resolve()
        registry = LicenseRegistry.load(registry_path)
        registry.verify_local_license_texts(registry_path)
        registry.admit(
            self.license_record_ids,
            stage="expression-model-training",
            production=not self.smoke_only,
        )
        if expression_cache_digest(self) != self.cache_digest:
            raise ValueError("expression cache manifest digest mismatch")
        records = [
            record
            for shard in self.shard_files
            for record in ExpressionCacheRecord.load_jsonl((root / shard.path).resolve())
        ]
        split_by_actor = {
            actor: split_name
            for split_name, split in self.actor_splits.splits.items()
            for actor in split.actors
        }
        split_by_clip = {
            clip: split_name
            for split_name, split in self.actor_splits.splits.items()
            for clip in split.clips
        }
        seen: set[str] = set()
        for record in records:
            if record.clip_id in seen:
                raise ValueError(f"duplicate expression cache clip: {record.clip_id}")
            seen.add(record.clip_id)
            if split_by_actor.get(record.actor_id) != split_by_clip.get(record.clip_id):
                raise ValueError(f"expression cache actor/clip split mismatch: {record.clip_id}")
            if len(record.emotion_distribution) != len(self.emotion_labels):
                raise ValueError(f"expression label width mismatch: {record.clip_id}")
            if record.label_source != self.label_source:
                raise ValueError(f"expression label source mismatch: {record.clip_id}")
        if seen != set(split_by_clip):
            raise ValueError("expression cache shards and actor split contain different clips")
        return records


class ExpressionCacheRecord(StrategyModel):
    actor_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    timestamps_ns: list[int] = Field(min_length=2)
    parameters: list[list[FiniteFloat]] = Field(min_length=2)
    confidence: list[list[FiniteFloat]] = Field(min_length=2)
    visibility: list[FiniteFloat] = Field(min_length=2)
    head_pose: list[list[FiniteFloat]] = Field(min_length=2)
    frame_quality: list[FiniteFloat] = Field(min_length=2)
    emotion_distribution: list[FiniteFloat] = Field(min_length=2)
    intensity: FiniteFloat = Field(ge=0.0, le=1.0)
    label_source: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sequence_shapes(self) -> Self:
        frame_count = len(self.timestamps_ns)
        if (
            any(
                len(values) != frame_count
                for values in (self.parameters, self.confidence, self.head_pose)
            )
            or len(self.visibility) != frame_count
            or len(self.frame_quality) != frame_count
        ):
            raise ValueError("all expression-cache frame fields must have equal length")
        if any(len(values) != 36 for values in self.parameters + self.confidence):
            raise ValueError("parameter and confidence frames require BasicSet width 36")
        if any(len(values) != 7 for values in self.head_pose):
            raise ValueError("head pose frames require xyz plus xyzw quaternion")
        if any(
            right <= left
            for left, right in zip(self.timestamps_ns, self.timestamps_ns[1:], strict=False)
        ):
            raise ValueError("expression-cache timestamps must strictly increase")
        if any(not 0.0 <= value <= 1.0 for frame in self.confidence for value in frame):
            raise ValueError("parameter confidence must stay within [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in self.visibility + self.frame_quality):
            raise ValueError("visibility and frame quality must stay within [0, 1]")
        total = sum(self.emotion_distribution)
        if any(value < 0.0 for value in self.emotion_distribution) or abs(total - 1.0) > 1e-5:
            raise ValueError("emotion distribution must be non-negative and sum to one")
        return self

    @classmethod
    def load_jsonl(cls, path: Path) -> list[Self]:
        records = [
            cls.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"expression cache shard is empty: {path}")
        return records


def load_clip_index(path: Path) -> list[ClipReference]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("clip index must be a JSON array")
    return [ClipReference.model_validate(item) for item in payload]


def expression_cache_digest(manifest: ExpressionCacheManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"cache_digest"})
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()
