"""Crash-safe capture chunks, reconciliation, ARKit derivation, and frozen datasets."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zlib
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated, BinaryIO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nana_tracking.data.labeling import LabelCatalog
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
from nana_tracking.data.schema import (
    CameraCalibration,
    CaptureConditions,
    CaptureRecord,
    DepthObservation,
    LabelObservation,
    RgbFrame,
    TeacherFrame,
)
from nana_tracking.data.strategy import CaptureSplitPlan, LicenseRegistry, split_captures

SHA256_PATTERN = r"^[0-9a-f]{64}$"
ChunkKind = Literal["rgb", "depth", "arkit", "geometry", "camera"]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
BlendshapeValue = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class CaptureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaptureChunk(CaptureModel):
    chunk_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    take_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    kind: ChunkKind
    relative_path: str = Field(min_length=1)
    sequence_start: int = Field(ge=0)
    sequence_end: int = Field(ge=0)
    capture_timestamp_start_ns: int = Field(ge=0)
    capture_timestamp_end_ns: int = Field(ge=0)
    byte_length: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("capture chunk path must be normalized and relative")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.sequence_end < self.sequence_start:
            raise ValueError("capture chunk sequence range is reversed")
        if self.capture_timestamp_end_ns < self.capture_timestamp_start_ns:
            raise ValueError("capture chunk timestamp range is reversed")
        expected_path = PurePosixPath(
            "chunks",
            self.take_id,
            self.kind,
            f"{self.sequence_start:020d}-{self.sequence_end:020d}-{self.chunk_id}.bin",
        ).as_posix()
        if self.relative_path != expected_path:
            raise ValueError("capture chunk path does not match its immutable descriptor")
        return self


class ChunkAcknowledgement(CaptureModel):
    chunk_id: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)


class CaptureSessionManifest(CaptureModel):
    schema_version: Literal["nana-capture-session/1.0.0"] = "nana-capture-session/1.0.0"
    capture_schema_version: Literal["ntp-capture/1.0.0"] = "ntp-capture/1.0.0"
    session_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    device_model: str = Field(min_length=1)
    os_version: str = Field(min_length=1)
    ntp_mapping_revision: str = Field(min_length=1)
    consent_record_id: str = Field(min_length=1)
    license_record_ids: list[str] = Field(min_length=1)
    started_at_ns: int = Field(ge=0)
    ended_at_ns: int = Field(ge=0)
    complete: Literal[True] = True
    chunks: list[CaptureChunk] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.ended_at_ns < self.started_at_ns:
            raise ValueError("capture session end precedes start")
        if self.license_record_ids != sorted(set(self.license_record_ids)):
            raise ValueError("license record IDs must be unique and increasing")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        paths = [chunk.relative_path for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)) or len(paths) != len(set(paths)):
            raise ValueError("capture chunk IDs and paths must be unique")
        for key, chunks in _group_chunks(self.chunks).items():
            previous_end: int | None = None
            for chunk in chunks:
                if previous_end is not None and chunk.sequence_start <= previous_end:
                    raise ValueError(f"overlapping capture chunks for {key}")
                previous_end = chunk.sequence_end
        if capture_session_digest(self) != self.manifest_sha256:
            raise ValueError("capture session manifest digest mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        subject_id: str,
        device_id: str,
        device_model: str,
        os_version: str,
        ntp_mapping_revision: str,
        consent_record_id: str,
        license_record_ids: list[str],
        chunks: list[CaptureChunk],
    ) -> Self:
        if not chunks:
            raise ValueError("cannot finalize a capture session without chunks")
        payload: dict[str, object] = {
            "schema_version": "nana-capture-session/1.0.0",
            "capture_schema_version": "ntp-capture/1.0.0",
            "session_id": session_id,
            "subject_id": subject_id,
            "device_id": device_id,
            "device_model": device_model,
            "os_version": os_version,
            "ntp_mapping_revision": ntp_mapping_revision,
            "consent_record_id": consent_record_id,
            "license_record_ids": sorted(set(license_record_ids)),
            "started_at_ns": min(chunk.capture_timestamp_start_ns for chunk in chunks),
            "ended_at_ns": max(chunk.capture_timestamp_end_ns for chunk in chunks),
            "complete": True,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        payload["manifest_sha256"] = _canonical_digest(payload)
        return cls.model_validate(payload)

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        _atomic_write(path, _canonical_json(self.model_dump(mode="json"), pretty=True))

    def verify_files(self, root: Path) -> None:
        root = root.resolve()
        for chunk in self.chunks:
            path = (root / chunk.relative_path).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"capture chunk is missing or outside session: {chunk.chunk_id}")
            if path.stat().st_size != chunk.byte_length:
                raise ValueError(f"capture chunk length mismatch: {chunk.chunk_id}")
            if _file_digest(path) != chunk.sha256:
                raise ValueError(f"capture chunk digest mismatch: {chunk.chunk_id}")


def capture_session_digest(manifest: CaptureSessionManifest) -> str:
    return _canonical_digest(manifest.model_dump(mode="json", exclude={"manifest_sha256"}))


class LocalChunkStore:
    """Append-only local-first chunk store with durable acknowledgement state."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._state = self.root / ".capture-state"
        self._state.mkdir(parents=True, exist_ok=True)
        self._chunks_journal = self._state / "chunks.jsonl"
        self._ack_journal = self._state / "acknowledged.jsonl"
        chunks = _load_jsonl(self._chunks_journal, CaptureChunk)
        acknowledgements = _load_jsonl(self._ack_journal, ChunkAcknowledgement)
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._chunk_paths = {chunk.relative_path for chunk in chunks}
        self._acknowledgements_by_id = {
            acknowledgement.chunk_id: acknowledgement for acknowledgement in acknowledgements
        }
        if len(self._chunks_by_id) != len(chunks):
            raise ValueError("local chunk journal contains duplicate IDs")
        if len(self._chunk_paths) != len(chunks):
            raise ValueError("local chunk journal contains duplicate paths")
        if len(self._acknowledgements_by_id) != len(acknowledgements):
            raise ValueError("acknowledgement journal contains duplicate chunk IDs")

    def write_chunk(
        self,
        *,
        chunk_id: str,
        take_id: str,
        kind: ChunkKind,
        sequence_start: int,
        sequence_end: int,
        capture_timestamp_start_ns: int,
        capture_timestamp_end_ns: int,
        payload: bytes,
    ) -> CaptureChunk:
        if not payload:
            raise ValueError("capture chunks cannot be empty")
        relative_path = PurePosixPath(
            "chunks",
            take_id,
            kind,
            f"{sequence_start:020d}-{sequence_end:020d}-{chunk_id}.bin",
        ).as_posix()
        chunk = CaptureChunk(
            chunk_id=chunk_id,
            take_id=take_id,
            kind=kind,
            relative_path=relative_path,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            capture_timestamp_start_ns=capture_timestamp_start_ns,
            capture_timestamp_end_ns=capture_timestamp_end_ns,
            byte_length=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        return self.receive_chunk(chunk, payload)

    def receive_chunk(self, chunk: CaptureChunk, payload: bytes) -> CaptureChunk:
        """Durably persist bytes and journal metadata before an acknowledgement may be returned."""

        if len(payload) != chunk.byte_length:
            raise ValueError(f"received chunk payload does not match descriptor: {chunk.chunk_id}")
        return self.receive_chunk_stream(chunk, BytesIO(payload))

    def receive_chunk_stream(self, chunk: CaptureChunk, stream: BinaryIO) -> CaptureChunk:
        """Verify and persist a bounded stream without retaining the whole chunk in memory."""

        existing = self._chunks_by_id.get(chunk.chunk_id)
        if existing is not None and existing != chunk:
            raise ValueError(f"chunk ID was reused with different metadata: {chunk.chunk_id}")
        if chunk.relative_path in self._chunk_paths and existing is None:
            raise ValueError(f"chunk path was reused: {chunk.relative_path}")
        path = self.root / chunk.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        remaining = chunk.byte_length
        try:
            with os.fdopen(descriptor, "wb") as output:
                while remaining:
                    block = stream.read(min(remaining, 1024 * 1024))
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    remaining -= len(block)
                output.flush()
                os.fsync(output.fileno())
            if remaining or digest.hexdigest() != chunk.sha256:
                raise ValueError(
                    f"received chunk payload does not match descriptor: {chunk.chunk_id}"
                )
            if existing is not None:
                return existing
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        _append_jsonl(self._chunks_journal, chunk.model_dump(mode="json"))
        self._chunks_by_id[chunk.chunk_id] = chunk
        self._chunk_paths.add(chunk.relative_path)
        return chunk

    def chunks(self) -> list[CaptureChunk]:
        return list(self._chunks_by_id.values())

    def acknowledge(self, acknowledgement: ChunkAcknowledgement) -> None:
        chunk = self._chunks_by_id.get(acknowledgement.chunk_id)
        if chunk is None:
            raise ValueError(f"cannot acknowledge unknown chunk: {acknowledgement.chunk_id}")
        if chunk.sha256 != acknowledgement.sha256:
            raise ValueError(f"remote acknowledgement digest mismatch: {acknowledgement.chunk_id}")
        previous = self._acknowledgements_by_id.get(acknowledgement.chunk_id)
        if previous is not None:
            if previous != acknowledgement:
                raise ValueError(f"conflicting acknowledgement: {acknowledgement.chunk_id}")
            return
        _append_jsonl(self._ack_journal, acknowledgement.model_dump(mode="json"))
        self._acknowledgements_by_id[acknowledgement.chunk_id] = acknowledgement

    def acknowledgements(self) -> list[ChunkAcknowledgement]:
        return list(self._acknowledgements_by_id.values())

    def pending_chunks(self) -> list[CaptureChunk]:
        acknowledged = {item.chunk_id for item in self.acknowledgements()}
        return [chunk for chunk in self.chunks() if chunk.chunk_id not in acknowledged]

    def finalize(
        self,
        *,
        session_id: str,
        subject_id: str,
        device_id: str,
        device_model: str,
        os_version: str,
        ntp_mapping_revision: str,
        consent_record_id: str,
        license_record_ids: list[str],
    ) -> CaptureSessionManifest:
        chunks = self.chunks()
        manifest = CaptureSessionManifest.create(
            session_id=session_id,
            subject_id=subject_id,
            device_id=device_id,
            device_model=device_model,
            os_version=os_version,
            ntp_mapping_revision=ntp_mapping_revision,
            consent_record_id=consent_record_id,
            license_record_ids=license_record_ids,
            chunks=chunks,
        )
        manifest.verify_files(self.root)
        manifest.save(self.root / "session.json")
        return manifest


class MissingChunkRange(CaptureModel):
    take_id: str
    kind: ChunkKind
    sequence_start: int
    sequence_end: int
    chunk_ids: list[str] = Field(min_length=1)


class ReconciliationResult(CaptureModel):
    missing_ranges: list[MissingChunkRange]
    mismatched_chunk_ids: list[str]
    unexpected_chunk_ids: list[str]

    @property
    def complete(self) -> bool:
        return not (self.missing_ranges or self.mismatched_chunk_ids or self.unexpected_chunk_ids)


def reconcile_chunks(
    expected: Iterable[CaptureChunk],
    received: Iterable[ChunkAcknowledgement],
) -> ReconciliationResult:
    expected_by_id = {chunk.chunk_id: chunk for chunk in expected}
    received_by_id: dict[str, ChunkAcknowledgement] = {}
    for item in received:
        if item.chunk_id in received_by_id:
            raise ValueError(f"duplicate remote chunk acknowledgement: {item.chunk_id}")
        received_by_id[item.chunk_id] = item
    mismatched = sorted(
        chunk_id
        for chunk_id in expected_by_id.keys() & received_by_id.keys()
        if expected_by_id[chunk_id].sha256 != received_by_id[chunk_id].sha256
    )
    missing = [
        chunk
        for chunk_id, chunk in expected_by_id.items()
        if chunk_id not in received_by_id or chunk_id in mismatched
    ]
    unexpected = sorted(received_by_id.keys() - expected_by_id.keys())
    return ReconciliationResult(
        missing_ranges=_merge_missing_ranges(missing),
        mismatched_chunk_ids=mismatched,
        unexpected_chunk_ids=unexpected,
    )


class PreviewFrame:
    __slots__ = ("capture_timestamp_ns", "jpeg", "sequence")

    def __init__(self, *, sequence: int, capture_timestamp_ns: int, jpeg: bytes) -> None:
        if sequence < 0 or capture_timestamp_ns < 0 or not jpeg:
            raise ValueError("preview frames require non-negative timing and non-empty JPEG bytes")
        self.sequence = sequence
        self.capture_timestamp_ns = capture_timestamp_ns
        self.jpeg = jpeg


class LatestPreview:
    """Single-slot preview handoff, intentionally unrelated to durable capture chunks."""

    def __init__(self) -> None:
        self._pending: PreviewFrame | None = None
        self.dropped = 0

    def publish(self, frame: PreviewFrame) -> None:
        if self._pending is not None:
            self.dropped += 1
        self._pending = frame

    def take(self) -> PreviewFrame | None:
        pending = self._pending
        self._pending = None
        return pending


class RawArkitFrame(CaptureModel):
    schema_version: Literal["nana-raw-arkit-frame/1.0.0"] = "nana-raw-arkit-frame/1.0.0"
    record_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    take_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    action_script_id: str = Field(min_length=1)
    consent_record_id: str = Field(min_length=1)
    capture_timestamp_ns: int = Field(ge=0)
    sequence: int = Field(ge=0)
    rgb: RgbFrame
    camera: CameraCalibration
    blendshapes: dict[str, BlendshapeValue]
    head_transform_column_major: list[FiniteFloat] = Field(min_length=16, max_length=16)
    left_eye_transform_column_major: list[FiniteFloat] = Field(min_length=16, max_length=16)
    right_eye_transform_column_major: list[FiniteFloat] = Field(min_length=16, max_length=16)
    face_geometry_uri: str | None = None
    depth_uri: str | None = None
    depth_confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    tracking_state: Literal["normal", "limited", "not_available"]
    conditions: CaptureConditions

    @classmethod
    def load_jsonl(cls, path: Path) -> list[Self]:
        values = _load_jsonl(path, cls)
        if not values:
            raise ValueError(f"raw ARKit frame file is empty: {path}")
        return values


class ArkitMappingRule(CaptureModel):
    target_signal: str = Field(min_length=1)
    source_blendshape: str = Field(min_length=1)
    scale: FiniteFloat = 1.0
    offset: FiniteFloat = 0.0
    minimum: FiniteFloat
    maximum: FiniteFloat
    confidence: FiniteFloat = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.maximum <= self.minimum:
            raise ValueError("ARKit mapping range must increase")
        return self


class ArkitMapping(CaptureModel):
    schema_version: Literal["arkit-to-ntp/1.0.0"] = "arkit-to-ntp/1.0.0"
    mapping_revision: str = Field(min_length=1)
    signal_registry_revision: Literal["ntp-signals/1.0.0"] = "ntp-signals/1.0.0"
    teacher_source_id: str = Field(min_length=1)
    license_record_id: str = Field(min_length=1)
    rules: list[ArkitMappingRule] = Field(min_length=1)
    smoke_only: bool

    @model_validator(mode="after")
    def validate_rules(self) -> Self:
        targets = [rule.target_signal for rule in self.rules]
        if len(targets) != len(set(targets)):
            raise ValueError("ARKit mapping target signals must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        _atomic_write(path, _canonical_json(self.model_dump(mode="json"), pretty=True))


def convert_arkit_frames(
    frames: Iterable[RawArkitFrame], mapping: ArkitMapping
) -> list[CaptureRecord]:
    records: list[CaptureRecord] = []
    last_by_session: dict[str, tuple[int, int]] = {}
    for frame in frames:
        previous = last_by_session.get(frame.session_id)
        if previous is not None and (
            frame.sequence <= previous[0] or frame.capture_timestamp_ns <= previous[1]
        ):
            raise ValueError(f"raw ARKit frame order regressed in session {frame.session_id}")
        last_by_session[frame.session_id] = (frame.sequence, frame.capture_timestamp_ns)
        labels = {
            rule.target_signal: _derive_arkit_label(frame, mapping.mapping_revision, rule)
            for rule in mapping.rules
        }
        depth = [
            DepthObservation(
                source_id=mapping.teacher_source_id,
                capture_timestamp_ns=frame.capture_timestamp_ns,
                state="observed",
                confidence=frame.depth_confidence,
                depth_uri=frame.depth_uri,
            )
            if frame.depth_uri is not None and frame.tracking_state == "normal"
            else DepthObservation(
                source_id=mapping.teacher_source_id,
                capture_timestamp_ns=frame.capture_timestamp_ns,
                state="unavailable",
                confidence=0.0,
                unavailable_reason="depth-or-tracking-unavailable",
            )
        ]
        records.append(
            CaptureRecord(
                record_id=frame.record_id,
                identity_id=frame.subject_id,
                session_id=frame.session_id,
                take_id=frame.take_id,
                device_id=frame.device_id,
                action_script_id=frame.action_script_id,
                consent_record_id=frame.consent_record_id,
                capture_timestamp_ns=frame.capture_timestamp_ns,
                sequence=frame.sequence,
                rgb=frame.rgb,
                camera=frame.camera,
                teachers=[
                    TeacherFrame(
                        source_id=mapping.teacher_source_id,
                        capture_timestamp_ns=frame.capture_timestamp_ns,
                        labels=labels,
                    )
                ],
                depth=depth,
                conditions=frame.conditions,
            )
        )
    if not records:
        raise ValueError("ARKit conversion requires at least one raw frame")
    return records


def write_capture_records(records: Iterable[CaptureRecord], path: Path) -> None:
    lines = [record.model_dump_json() for record in records]
    if not lines:
        raise ValueError("capture record output cannot be empty")
    _atomic_write(path, ("\n".join(lines) + "\n").encode())


class FrozenCaptureDataset(CaptureModel):
    schema_version: Literal["nana-frozen-capture-dataset/1.0.0"] = (
        "nana-frozen-capture-dataset/1.0.0"
    )
    frozen: Literal[True] = True
    data_revision: str = Field(min_length=1)
    ntp_mapping_revisions: list[str] = Field(min_length=1)
    arkit_mappings: list[FileReference] = Field(min_length=1)
    capture_sessions: list[FileReference] = Field(min_length=1)
    capture_records: RecordFile
    license_registry: FileReference
    license_record_ids: list[str] = Field(min_length=1)
    split_plan: CaptureSplitPlan
    smoke_only: bool
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        if self.ntp_mapping_revisions != sorted(set(self.ntp_mapping_revisions)):
            raise ValueError("mapping revisions must be unique and increasing")
        if self.license_record_ids != sorted(set(self.license_record_ids)):
            raise ValueError("license record IDs must be unique and increasing")
        if frozen_capture_dataset_digest(self) != self.dataset_sha256:
            raise ValueError("frozen capture dataset digest mismatch")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        _atomic_write(path, _canonical_json(self.model_dump(mode="json"), pretty=True))

    def verify(self, manifest_path: Path) -> None:
        root = manifest_path.parent.resolve()
        references: list[FileReference] = [
            *self.capture_sessions,
            *self.arkit_mappings,
            self.capture_records,
            self.license_registry,
        ]
        for reference in references:
            path = (root / reference.path).resolve()
            if not path.is_file() or _file_digest(path) != reference.sha256:
                raise ValueError(f"frozen capture reference mismatch: {reference.path}")
        sessions = [
            CaptureSessionManifest.load((root / reference.path).resolve())
            for reference in self.capture_sessions
        ]
        session_paths = [(root / reference.path).resolve() for reference in self.capture_sessions]
        for path, session in zip(session_paths, sessions, strict=True):
            session.verify_files(path.parent)
        records_path = (root / self.capture_records.path).resolve()
        records = CaptureRecord.load_jsonl(records_path)
        if len(records) != self.capture_records.record_count:
            raise ValueError("frozen capture record count mismatch")
        _validate_session_records(sessions, records, self.license_record_ids)
        mappings = [
            ArkitMapping.load((root / reference.path).resolve())
            for reference in self.arkit_mappings
        ]
        _validate_mapping_records(
            mappings,
            sessions,
            records,
            self.license_record_ids,
            smoke_only=self.smoke_only,
        )
        _validate_frozen_record_paths(session_paths, sessions, records, records_path.parent)
        rebuilt = split_captures(
            records,
            seed=self.split_plan.seed,
            held_out_test_devices=set(self.split_plan.held_out_test_devices),
            validation_identities=len(self.split_plan.splits["validation"].identities),
        )
        if rebuilt != self.split_plan:
            raise ValueError("frozen capture split plan is not reproducible")
        registry_path = (root / self.license_registry.path).resolve()
        registry = LicenseRegistry.load(registry_path)
        registry.verify_local_license_texts(registry_path)
        registry.admit(
            self.license_record_ids,
            stage="base-model-training",
            production=not self.smoke_only,
        )
        registry.admit(
            self.license_record_ids,
            stage="teacher-labeling",
            production=not self.smoke_only,
        )


def freeze_capture_dataset(
    *,
    data_revision: str,
    session_manifests: list[Path],
    capture_records: Path,
    arkit_mappings: list[Path],
    license_registry: Path,
    license_record_ids: list[str],
    held_out_test_devices: set[str],
    validation_identities: int,
    seed: int,
    smoke_only: bool,
    output: Path,
) -> FrozenCaptureDataset:
    sessions = [CaptureSessionManifest.load(path) for path in session_manifests]
    for path, session in zip(session_manifests, sessions, strict=True):
        session.verify_files(path.parent)
    records = CaptureRecord.load_jsonl(capture_records)
    _validate_session_records(sessions, records, license_record_ids)
    mappings = [ArkitMapping.load(path) for path in arkit_mappings]
    _validate_mapping_records(
        mappings,
        sessions,
        records,
        license_record_ids,
        smoke_only=smoke_only,
    )
    normalized_records = _normalize_record_paths(
        session_manifests,
        sessions,
        records,
        output.parent.resolve(),
    )
    normalized_records_path = output.with_name(f"{output.stem}-records.jsonl")
    write_capture_records(normalized_records, normalized_records_path)
    split_plan = split_captures(
        records,
        seed=seed,
        held_out_test_devices=held_out_test_devices,
        validation_identities=validation_identities,
    )
    registry = LicenseRegistry.load(license_registry)
    registry.verify_local_license_texts(license_registry)
    registry.admit(
        license_record_ids,
        stage="base-model-training",
        production=not smoke_only,
    )
    registry.admit(
        license_record_ids,
        stage="teacher-labeling",
        production=not smoke_only,
    )
    root = output.parent.resolve()
    payload: dict[str, object] = {
        "schema_version": "nana-frozen-capture-dataset/1.0.0",
        "frozen": True,
        "data_revision": data_revision,
        "ntp_mapping_revisions": sorted(mapping.mapping_revision for mapping in mappings),
        "arkit_mappings": [
            _file_reference(path, root).model_dump(mode="json") for path in sorted(arkit_mappings)
        ],
        "capture_sessions": [
            _file_reference(path, root).model_dump(mode="json")
            for path in sorted(session_manifests)
        ],
        "capture_records": {
            **_file_reference(normalized_records_path, root).model_dump(mode="json"),
            "record_count": len(normalized_records),
        },
        "license_registry": _file_reference(license_registry, root).model_dump(mode="json"),
        "license_record_ids": sorted(set(license_record_ids)),
        "split_plan": split_plan.model_dump(mode="json"),
        "smoke_only": smoke_only,
    }
    payload["dataset_sha256"] = _canonical_digest(payload)
    frozen = FrozenCaptureDataset.model_validate(payload)
    frozen.save(output)
    frozen.verify(output)
    return frozen


def frozen_capture_dataset_digest(index: FrozenCaptureDataset) -> str:
    return _canonical_digest(index.model_dump(mode="json", exclude={"dataset_sha256"}))


def build_training_manifest_from_frozen(
    frozen_path: Path,
    *,
    label_catalog_path: Path,
    output: Path,
) -> DatasetManifest:
    """Build the only DatasetManifest permitted to expose a frozen capture revision to training."""

    frozen = FrozenCaptureDataset.load(frozen_path)
    frozen.verify(frozen_path)
    frozen_root = frozen_path.parent.resolve()
    records_path = (frozen_root / frozen.capture_records.path).resolve()
    registry_path = (frozen_root / frozen.license_registry.path).resolve()
    mappings = [
        ArkitMapping.load((frozen_root / reference.path).resolve())
        for reference in frozen.arkit_mappings
    ]
    catalog = LabelCatalog.load(label_catalog_path)
    registry = LicenseRegistry.load(registry_path)
    admitted = registry.admit(
        frozen.license_record_ids,
        stage="base-model-training",
        production=not frozen.smoke_only,
    )
    registry.admit(
        frozen.license_record_ids,
        stage="teacher-labeling",
        production=not frozen.smoke_only,
    )
    admitted_by_id = {record.record_id: record for record in admitted}
    mapping_license_ids = sorted({mapping.license_record_id for mapping in mappings})
    license_reviews: list[LicenseReview] = []
    for record_id in mapping_license_ids:
        record = admitted_by_id[record_id]
        license_reviews.append(
            LicenseReview(
                license_id=record.record_id,
                scope=(
                    "synthetic"
                    if record.smoke_only
                    else "first_party"
                    if record.kind == "first-party-capture"
                    else "third_party"
                ),
                status="approved",
                evidence=record.evidence,
                permissions=LicensePermissions(
                    collection=True,
                    distillation=record.permissions.distillation_allowed,
                    pseudo_labeling=record.permissions.pseudo_labeling_allowed,
                    commercial_training=record.permissions.commercial_training_allowed,
                ),
            )
        )
    root = output.parent.resolve()
    payload: dict[str, object] = {
        "schema_version": "ntp-dataset/2.0.0",
        "capture_schema_version": "ntp-capture/1.0.0",
        "data_revision": frozen.data_revision,
        "digest": "0" * 64,
        "ntp_schema_revision": catalog.ntp_schema_revision,
        "signal_registry_revision": catalog.signal_registry_revision,
        "normalization_revision": catalog.normalization_revision,
        "calibration_revision": catalog.calibration_revision,
        "feature_revision": catalog.feature_revision,
        "pipeline_stage": "base-model-training",
        "label_catalog": _file_reference(label_catalog_path, root).model_dump(mode="json"),
        "license_registry": _file_reference(registry_path, root).model_dump(mode="json"),
        "license_record_ids": frozen.license_record_ids,
        "record_files": [
            {
                **_file_reference(records_path, root).model_dump(mode="json"),
                "record_count": frozen.capture_records.record_count,
            }
        ],
        "teacher_sources": [
            TeacherSource(
                source_id=mapping.teacher_source_id,
                source_type="truedepth",
                version=mapping.mapping_revision,
                license_id=mapping.license_record_id,
            ).model_dump(mode="json")
            for mapping in mappings
        ],
        "license_reviews": [review.model_dump(mode="json") for review in license_reviews],
        "synchronization": SynchronizationPolicy(
            max_teacher_skew_ns=5_000_000,
            max_depth_skew_ns=2_000_000,
        ).model_dump(mode="json"),
        "splits": {
            name: SplitManifest.model_validate(split.model_dump(mode="json")).model_dump(
                mode="json"
            )
            for name, split in frozen.split_plan.splits.items()
        },
        "smoke_only": frozen.smoke_only,
    }
    candidate = DatasetManifest.model_validate(payload)
    payload["digest"] = dataset_digest(candidate)
    manifest = DatasetManifest.model_validate(payload)
    manifest.save(output)
    manifest.verify_files(output)
    return manifest


def run_capture_pipeline_smoke(
    work_dir: Path,
    *,
    mapping_path: Path,
    license_registry: Path,
    label_catalog_path: Path,
) -> dict[str, object]:
    """Run a deterministic synthetic local-record, sync, derive, split, and freeze closure."""

    mapping = ArkitMapping.load(mapping_path)
    if not mapping.smoke_only:
        raise ValueError("capture smoke requires an explicitly smoke-only mapping")
    identities = [
        ("smoke-train", "smoke-train-session", "smoke-main-device"),
        ("smoke-validation", "smoke-validation-session", "smoke-main-device"),
        ("smoke-test", "smoke-test-session", "smoke-heldout-device"),
    ]
    sessions: list[Path] = []
    raw_frames: list[RawArkitFrame] = []
    receiver_ack_count = 0
    identity_matrix = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    for index, (subject_id, session_id, device_id) in enumerate(identities):
        sender = LocalChunkStore(work_dir / "ios" / session_id)
        timestamp_ns = 1_000_000_000 + index * 1_000_000
        rgb = sender.write_chunk(
            chunk_id=f"{session_id}-rgb-0",
            take_id="take-basic-1",
            kind="rgb",
            sequence_start=0,
            sequence_end=0,
            capture_timestamp_start_ns=timestamp_ns,
            capture_timestamp_end_ns=timestamp_ns,
            payload=_synthetic_png(64, 64, index),
        )
        depth = sender.write_chunk(
            chunk_id=f"{session_id}-depth-0",
            take_id="take-basic-1",
            kind="depth",
            sequence_start=0,
            sequence_end=0,
            capture_timestamp_start_ns=timestamp_ns,
            capture_timestamp_end_ns=timestamp_ns,
            payload=f"synthetic-depth-{subject_id}".encode(),
        )
        geometry = sender.write_chunk(
            chunk_id=f"{session_id}-geometry-0",
            take_id="take-basic-1",
            kind="geometry",
            sequence_start=0,
            sequence_end=0,
            capture_timestamp_start_ns=timestamp_ns,
            capture_timestamp_end_ns=timestamp_ns,
            payload=b'{"synthetic":true,"vertices":[]}',
        )
        frame = RawArkitFrame(
            record_id=f"{session_id}-record-0",
            subject_id=subject_id,
            session_id=session_id,
            take_id="take-basic-1",
            device_id=device_id,
            action_script_id="basic-smoke-v1",
            consent_record_id=f"synthetic-consent-{subject_id}",
            capture_timestamp_ns=timestamp_ns,
            sequence=0,
            rgb=RgbFrame(
                uri=rgb.relative_path,
                width=64,
                height=64,
                exposure_duration_ns=10_000_000,
                iso=100.0,
                frame_duration_ns=16_666_667,
            ),
            camera=CameraCalibration(
                intrinsics=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                distortion_model="none",
                distortion_coefficients=[],
            ),
            blendshapes={rule.source_blendshape: 0.25 for rule in mapping.rules},
            head_transform_column_major=identity_matrix,
            left_eye_transform_column_major=identity_matrix,
            right_eye_transform_column_major=identity_matrix,
            face_geometry_uri=geometry.relative_path,
            depth_uri=depth.relative_path,
            depth_confidence=0.9,
            tracking_state="normal",
            conditions=CaptureConditions(lighting="normal"),
        )
        raw_frames.append(frame)
        sender.write_chunk(
            chunk_id=f"{session_id}-arkit-0",
            take_id="take-basic-1",
            kind="arkit",
            sequence_start=0,
            sequence_end=0,
            capture_timestamp_start_ns=timestamp_ns,
            capture_timestamp_end_ns=timestamp_ns,
            payload=(frame.model_dump_json() + "\n").encode(),
        )
        manifest = sender.finalize(
            session_id=session_id,
            subject_id=subject_id,
            device_id=device_id,
            device_model="synthetic-ios-device",
            os_version="synthetic-ios",
            ntp_mapping_revision=mapping.mapping_revision,
            consent_record_id=f"synthetic-consent-{subject_id}",
            license_record_ids=["nana-synthetic-smoke"],
        )
        manifest_path = sender.root / "session.json"
        sessions.append(manifest_path)
        receiver = LocalChunkStore(work_dir / "studio" / session_id)
        acknowledgements: list[ChunkAcknowledgement] = []
        for chunk in manifest.chunks:
            payload = (sender.root / chunk.relative_path).read_bytes()
            receiver.receive_chunk(chunk, payload)
            acknowledgements.append(
                ChunkAcknowledgement(chunk_id=chunk.chunk_id, sha256=chunk.sha256)
            )
        if not reconcile_chunks(manifest.chunks, acknowledgements).complete:
            raise ValueError(f"synthetic receiver reconciliation failed: {session_id}")
        receiver_ack_count += len(acknowledgements)

    records_path = work_dir / "derived" / "capture-records.jsonl"
    records = convert_arkit_frames(raw_frames, mapping)
    write_capture_records(records, records_path)
    dataset_path = work_dir / "dataset" / "frozen-capture.json"
    frozen = freeze_capture_dataset(
        data_revision="capture-pipeline-smoke-v1",
        session_manifests=sessions,
        capture_records=records_path,
        arkit_mappings=[mapping_path],
        license_registry=license_registry,
        license_record_ids=["nana-synthetic-smoke"],
        held_out_test_devices={"smoke-heldout-device"},
        validation_identities=1,
        seed=17,
        smoke_only=True,
        output=dataset_path,
    )
    training_manifest_path = work_dir / "dataset" / "training-manifest.json"
    training_manifest = build_training_manifest_from_frozen(
        dataset_path,
        label_catalog_path=label_catalog_path,
        output=training_manifest_path,
    )
    return {
        "schema": "nana-capture-pipeline-smoke/1.0.0",
        "smoke_only": True,
        "session_count": len(sessions),
        "chunk_acknowledgement_count": receiver_ack_count,
        "derived_record_count": len(records),
        "mapping_revision": mapping.mapping_revision,
        "dataset": str(dataset_path),
        "dataset_sha256": frozen.dataset_sha256,
        "training_manifest": str(training_manifest_path),
        "training_manifest_digest": training_manifest.digest,
        "splits": frozen.split_plan.model_dump(mode="json"),
        "warning": (
            "Synthetic capture smoke does not prove TrueDepth, Windows, or tracking quality."
        ),
    }


def _derive_arkit_label(
    frame: RawArkitFrame, mapping_revision: str, rule: ArkitMappingRule
) -> LabelObservation:
    source = frame.blendshapes.get(rule.source_blendshape)
    if frame.tracking_state != "normal" or source is None:
        return LabelObservation(
            confidence=0.0,
            state="unavailable",
            evidence="teacher_label",
            method=f"{mapping_revision}:{rule.source_blendshape}",
            unavailable_reason="tracking-or-source-unavailable",
        )
    value = source * rule.scale + rule.offset
    if not rule.minimum <= value <= rule.maximum:
        raise ValueError(f"ARKit mapping produced out-of-range {rule.target_signal}: {value}")
    return LabelObservation(
        value=value,
        confidence=rule.confidence,
        state="observed",
        evidence="teacher_label",
        method=f"{mapping_revision}:{rule.source_blendshape}",
    )


def _validate_session_records(
    sessions: Iterable[CaptureSessionManifest],
    records: Iterable[CaptureRecord],
    license_record_ids: Iterable[str],
) -> None:
    sessions_by_id: dict[str, CaptureSessionManifest] = {}
    admitted_licenses = set(license_record_ids)
    for session in sessions:
        if session.session_id in sessions_by_id:
            raise ValueError(f"duplicate frozen capture session: {session.session_id}")
        if not set(session.license_record_ids).issubset(admitted_licenses):
            raise ValueError(f"capture session licenses are not admitted: {session.session_id}")
        sessions_by_id[session.session_id] = session
    seen_sessions: set[str] = set()
    for record in records:
        session = sessions_by_id.get(record.session_id)
        if session is None:
            raise ValueError(f"capture record references an unfrozen session: {record.session_id}")
        if record.identity_id != session.subject_id or record.device_id != session.device_id:
            raise ValueError(f"capture record identity/device mismatch: {record.record_id}")
        if record.consent_record_id != session.consent_record_id:
            raise ValueError(f"capture record consent mismatch: {record.record_id}")
        if not record.take_id:
            raise ValueError(f"frozen capture record has no take ID: {record.record_id}")
        seen_sessions.add(record.session_id)
    if seen_sessions != set(sessions_by_id):
        raise ValueError("capture records do not cover every frozen session")


def _validate_mapping_records(
    mappings: list[ArkitMapping],
    sessions: list[CaptureSessionManifest],
    records: list[CaptureRecord],
    license_record_ids: Iterable[str],
    *,
    smoke_only: bool,
) -> None:
    if not mappings:
        raise ValueError("frozen capture requires at least one versioned ARKit mapping")
    revisions = [mapping.mapping_revision for mapping in mappings]
    if revisions != sorted(set(revisions)):
        raise ValueError("frozen ARKit mapping revisions must be unique and increasing")
    expected_revisions = sorted({session.ntp_mapping_revision for session in sessions})
    if revisions != expected_revisions:
        raise ValueError("frozen ARKit mappings do not cover every session mapping revision")
    by_source = {mapping.teacher_source_id: mapping for mapping in mappings}
    if len(by_source) != len(mappings):
        raise ValueError("each frozen ARKit mapping requires a unique teacher source ID")
    admitted_licenses = set(license_record_ids)
    for mapping in mappings:
        if mapping.smoke_only != smoke_only:
            raise ValueError("mapping smoke status does not match the frozen dataset")
        if mapping.license_record_id not in admitted_licenses:
            raise ValueError("ARKit mapping license is not admitted by the frozen dataset")
    for session in sessions:
        mapping = next(
            item for item in mappings if item.mapping_revision == session.ntp_mapping_revision
        )
        if mapping.license_record_id not in session.license_record_ids:
            raise ValueError(f"capture session does not admit its mapping: {session.session_id}")
    for record in records:
        for teacher in record.teachers:
            mapping = by_source.get(teacher.source_id)
            if mapping is None:
                raise ValueError(f"capture teacher has no frozen mapping: {teacher.source_id}")
            expected_prefix = f"{mapping.mapping_revision}:"
            if any(
                not observation.method.startswith(expected_prefix)
                for observation in teacher.labels.values()
            ):
                raise ValueError(
                    f"capture teacher labels do not match mapping {mapping.mapping_revision}"
                )


def _normalize_record_paths(
    session_paths: list[Path],
    sessions: list[CaptureSessionManifest],
    records: list[CaptureRecord],
    output_root: Path,
) -> list[CaptureRecord]:
    session_index = {
        session.session_id: (path.resolve(), session)
        for path, session in zip(session_paths, sessions, strict=True)
    }
    normalized: list[CaptureRecord] = []
    for record in records:
        session_path, session = session_index[record.session_id]
        rgb_path = _resolve_record_chunk(
            session_path,
            session,
            record.take_id,
            record.sequence,
            "rgb",
            record.rgb.uri,
        )
        depth = []
        for observation in record.depth:
            if observation.depth_uri is None:
                depth.append(observation)
                continue
            depth_path = _resolve_record_chunk(
                session_path,
                session,
                record.take_id,
                record.sequence,
                "depth",
                observation.depth_uri,
            )
            depth.append(
                observation.model_copy(update={"depth_uri": _relative_uri(depth_path, output_root)})
            )
        normalized.append(
            record.model_copy(
                update={
                    "rgb": record.rgb.model_copy(
                        update={"uri": _relative_uri(rgb_path, output_root)}
                    ),
                    "depth": depth,
                }
            )
        )
    return normalized


def _validate_frozen_record_paths(
    session_paths: list[Path],
    sessions: list[CaptureSessionManifest],
    records: list[CaptureRecord],
    records_root: Path,
) -> None:
    session_index = {
        session.session_id: (path.resolve(), session)
        for path, session in zip(session_paths, sessions, strict=True)
    }
    for record in records:
        session_path, session = session_index[record.session_id]
        assert record.take_id is not None
        expected_rgb = _find_chunk_path(
            session_path, session, record.take_id, record.sequence, "rgb"
        )
        actual_rgb = (records_root / record.rgb.uri).resolve()
        if actual_rgb != expected_rgb:
            raise ValueError(f"frozen RGB reference does not match its session: {record.record_id}")
        for observation in record.depth:
            if observation.depth_uri is None:
                continue
            expected_depth = _find_chunk_path(
                session_path, session, record.take_id, record.sequence, "depth"
            )
            actual_depth = (records_root / observation.depth_uri).resolve()
            if actual_depth != expected_depth:
                raise ValueError(
                    f"frozen depth reference does not match its session: {record.record_id}"
                )


def _resolve_record_chunk(
    session_path: Path,
    session: CaptureSessionManifest,
    take_id: str | None,
    sequence: int,
    kind: ChunkKind,
    uri: str,
) -> Path:
    if take_id is None:
        raise ValueError("capture record has no take ID")
    expected = _find_chunk_path(session_path, session, take_id, sequence, kind)
    actual = (session_path.parent / uri).resolve()
    if actual != expected:
        raise ValueError(f"capture {kind} reference does not match its session chunk: {uri}")
    return expected


def _find_chunk_path(
    session_path: Path,
    session: CaptureSessionManifest,
    take_id: str,
    sequence: int,
    kind: ChunkKind,
) -> Path:
    matches = [
        chunk
        for chunk in session.chunks
        if chunk.take_id == take_id
        and chunk.kind == kind
        and chunk.sequence_start <= sequence <= chunk.sequence_end
    ]
    if len(matches) != 1:
        raise ValueError(
            f"capture session must contain one {kind} chunk for {take_id} sequence {sequence}"
        )
    return (session_path.parent / matches[0].relative_path).resolve()


def _relative_uri(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _group_chunks(
    chunks: Iterable[CaptureChunk],
) -> dict[tuple[str, ChunkKind], list[CaptureChunk]]:
    grouped: dict[tuple[str, ChunkKind], list[CaptureChunk]] = {}
    for chunk in chunks:
        grouped.setdefault((chunk.take_id, chunk.kind), []).append(chunk)
    for values in grouped.values():
        values.sort(key=lambda chunk: chunk.sequence_start)
    return grouped


def _merge_missing_ranges(chunks: Iterable[CaptureChunk]) -> list[MissingChunkRange]:
    ranges: list[MissingChunkRange] = []
    for (take_id, kind), values in sorted(_group_chunks(chunks).items()):
        for chunk in values:
            if (
                ranges
                and ranges[-1].take_id == take_id
                and ranges[-1].kind == kind
                and chunk.sequence_start == ranges[-1].sequence_end + 1
            ):
                ranges[-1].sequence_end = chunk.sequence_end
                ranges[-1].chunk_ids.append(chunk.chunk_id)
            else:
                ranges.append(
                    MissingChunkRange(
                        take_id=take_id,
                        kind=kind,
                        sequence_start=chunk.sequence_start,
                        sequence_end=chunk.sequence_end,
                        chunk_ids=[chunk.chunk_id],
                    )
                )
    return ranges


def _file_reference(path: Path, root: Path) -> FileReference:
    path = path.resolve()
    relative = os.path.relpath(path, root).replace(os.sep, "/")
    return FileReference(path=relative, sha256=_file_digest(path))


def _load_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.is_file():
        return []
    values: list[T] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(model.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid journal entry at {path}:{line_number}") from error
    return values


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with path.open("ab") as stream:
        stream.write(_canonical_json(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    if created:
        _sync_directory(path.parent)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(payload: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _canonical_digest(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    canonical.pop("dataset_sha256", None)
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_png(width: int, height: int, seed: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    pixel = bytes(((seed * 53) % 256, (seed * 97) % 256, (seed * 193) % 256))
    scanlines = b"".join(b"\x00" + pixel * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
