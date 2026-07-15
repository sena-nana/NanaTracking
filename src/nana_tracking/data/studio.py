"""Durable Capture Studio control, quality, preview, and receiver state."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nana_tracking.data.capture import (
    CaptureChunk,
    CaptureSessionManifest,
    ChunkAcknowledgement,
    LocalChunkStore,
)
from nana_tracking.data.journal import append_jsonl as _append_jsonl
from nana_tracking.data.journal import load_jsonl as _load_jsonl

ControlAction = Literal["start", "pause", "stop", "retake", "end"]
SessionStatus = Literal["ready", "recording", "paused", "stopped", "complete"]
TakeStatus = Literal["recording", "paused", "stopped", "replaced"]
QualityFlag = Literal[
    "low_light",
    "overexposed",
    "occluded",
    "tracking_limited",
    "sequence_gap",
    "timestamp_gap",
]
FiniteParameter = Annotated[float, Field(allow_inf_nan=False)]


class StudioModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StudioSessionDefinition(StudioModel):
    schema_version: Literal["nana-capture-studio-session/1.0.0"] = (
        "nana-capture-studio-session/1.0.0"
    )
    session_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    device_model: str = Field(min_length=1)
    os_version: str = Field(min_length=1)
    ntp_mapping_revision: str = Field(min_length=1)
    consent_record_id: str = Field(min_length=1)
    license_record_ids: list[str] = Field(min_length=1)
    created_at_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_licenses(self) -> Self:
        if self.license_record_ids != sorted(set(self.license_record_ids)):
            raise ValueError("studio license record IDs must be unique and increasing")
        return self


class StudioControlCommand(StudioModel):
    schema_version: Literal["nana-capture-control/1.0.0"] = "nana-capture-control/1.0.0"
    session_id: str = Field(min_length=1)
    revision: int = Field(gt=0)
    command_id: str = Field(min_length=1)
    action: ControlAction
    take_id: str | None = None
    action_script_id: str | None = None
    retake_of: str | None = None
    issued_at_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_action_fields(self) -> Self:
        if self.action in {"start", "retake"} and not (self.take_id and self.action_script_id):
            raise ValueError("start and retake commands require take and action-script IDs")
        if self.action == "retake" and not self.retake_of:
            raise ValueError("retake commands require the replaced take ID")
        if self.action in {"pause", "stop"} and not self.take_id:
            raise ValueError("pause and stop commands require the current take ID")
        return self


class StudioCommandAcknowledgement(StudioModel):
    session_id: str = Field(min_length=1)
    revision: int = Field(gt=0)
    command_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    applied_at_ns: int = Field(ge=0)


class StudioTakeState(StudioModel):
    take_id: str
    action_script_id: str
    status: TakeStatus
    retake_of: str | None = None


class NormalizedPreviewPoint(StudioModel):
    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class CaptureQualitySample(StudioModel):
    session_id: str = Field(min_length=1)
    take_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    capture_timestamp_ns: int = Field(ge=0)
    luminance: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    clipped_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    occluded_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    tracking_state: Literal["normal", "limited", "not_available"]
    face_mesh: list[NormalizedPreviewPoint] = Field(default_factory=list, max_length=2048)
    parameters: dict[str, FiniteParameter] = Field(default_factory=dict, max_length=128)

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        if any(not name for name in self.parameters):
            raise ValueError("preview parameter names cannot be empty")
        return self


class CaptureQualityResult(StudioModel):
    sample: CaptureQualitySample
    flags: list[QualityFlag]
    acceptable: bool


class StudioState(StudioModel):
    definition: StudioSessionDefinition
    status: SessionStatus
    current_take_id: str | None
    takes: list[StudioTakeState]
    last_command_revision: int
    acknowledged_revision: int
    received_chunk_count: int
    latest_quality: CaptureQualityResult | None


class PreviewMetadata(StudioModel):
    session_id: str
    take_id: str
    sequence: int = Field(ge=0)
    capture_timestamp_ns: int = Field(ge=0)
    byte_length: int = Field(gt=0)


class CaptureStudio:
    """Single-session Studio backend with append-only control and durable chunk state."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._state_root = self.root / ".studio-state"
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._definition_path = self._state_root / "session.json"
        self._commands_path = self._state_root / "commands.jsonl"
        self._acks_path = self._state_root / "command-acks.jsonl"
        self._quality_path = self._state_root / "latest-quality.json"
        self._preview_path = self._state_root / "latest-preview.jpg"
        self._preview_metadata_path = self._state_root / "latest-preview.json"
        self._receiver = LocalChunkStore(self.root / "receiver")
        self._lock = threading.RLock()
        self._definition = self._load_definition()
        self._commands = _load_jsonl(self._commands_path, StudioControlCommand)
        self._acknowledgements = _load_jsonl(self._acks_path, StudioCommandAcknowledgement)
        self._validate_history()

    @classmethod
    def create(cls, root: Path, definition: StudioSessionDefinition) -> Self:
        root = root.resolve()
        definition_path = root / ".studio-state" / "session.json"
        if definition_path.exists():
            existing = StudioSessionDefinition.model_validate_json(
                definition_path.read_text(encoding="utf-8")
            )
            if existing != definition:
                raise ValueError("studio root already belongs to a different session")
        else:
            _atomic_write_json(definition_path, definition.model_dump(mode="json"))
        return cls(root)

    @property
    def definition(self) -> StudioSessionDefinition:
        if self._definition is None:
            raise ValueError("studio session has not been created")
        return self._definition

    def issue_control(
        self,
        action: ControlAction,
        *,
        take_id: str | None = None,
        action_script_id: str | None = None,
        retake_of: str | None = None,
    ) -> StudioControlCommand:
        with self._lock:
            state = self.state()
            self._validate_transition(
                state,
                action,
                take_id=take_id,
                action_script_id=action_script_id,
                retake_of=retake_of,
            )
            revision = len(self._commands) + 1
            command = StudioControlCommand(
                session_id=self.definition.session_id,
                revision=revision,
                command_id=f"{self.definition.session_id}-{revision:020d}",
                action=action,
                take_id=take_id,
                action_script_id=action_script_id,
                retake_of=retake_of,
                issued_at_ns=time.time_ns(),
            )
            _append_jsonl(self._commands_path, command.model_dump(mode="json"))
            self._commands.append(command)
            return command

    def commands_after(self, revision: int) -> list[StudioControlCommand]:
        if revision < 0:
            raise ValueError("command revision cannot be negative")
        with self._lock:
            return [command for command in self._commands if command.revision > revision]

    def acknowledge_command(
        self,
        *,
        revision: int,
        command_id: str,
        device_id: str,
        applied_at_ns: int,
    ) -> StudioCommandAcknowledgement:
        with self._lock:
            command = next(
                (candidate for candidate in self._commands if candidate.revision == revision),
                None,
            )
            if command is None or command.command_id != command_id:
                raise ValueError("cannot acknowledge an unknown or mismatched studio command")
            if device_id != self.definition.device_id:
                raise ValueError("command acknowledgement device does not match the session")
            acknowledgement = StudioCommandAcknowledgement(
                session_id=self.definition.session_id,
                revision=revision,
                command_id=command_id,
                device_id=device_id,
                applied_at_ns=applied_at_ns,
            )
            previous = next(
                (item for item in self._acknowledgements if item.revision == revision),
                None,
            )
            if previous is not None:
                if previous != acknowledgement:
                    raise ValueError("conflicting studio command acknowledgement")
                return previous
            _append_jsonl(self._acks_path, acknowledgement.model_dump(mode="json"))
            self._acknowledgements.append(acknowledgement)
            return acknowledgement

    def publish_quality(self, sample: CaptureQualitySample) -> CaptureQualityResult:
        with self._lock:
            self._validate_active_sample(sample)
            previous = self.latest_quality()
            if previous is not None and previous.sample.take_id == sample.take_id:
                prior = previous.sample
                if sample.sequence <= prior.sequence:
                    raise ValueError("quality sample sequence regressed")
                if sample.capture_timestamp_ns <= prior.capture_timestamp_ns:
                    raise ValueError("quality sample capture timestamp regressed")
            flags: list[QualityFlag] = []
            if sample.luminance < 0.12:
                flags.append("low_light")
            if sample.luminance > 0.95 or sample.clipped_fraction > 0.05:
                flags.append("overexposed")
            if sample.occluded_fraction > 0.25:
                flags.append("occluded")
            if sample.tracking_state != "normal":
                flags.append("tracking_limited")
            if previous is not None and previous.sample.take_id == sample.take_id:
                prior = previous.sample
                if sample.sequence != prior.sequence + 1:
                    flags.append("sequence_gap")
                if sample.capture_timestamp_ns - prior.capture_timestamp_ns > 100_000_000:
                    flags.append("timestamp_gap")
            result = CaptureQualityResult(
                sample=sample,
                flags=flags,
                acceptable=not flags,
            )
            _atomic_write_json(self._quality_path, result.model_dump(mode="json"))
            return result

    def latest_quality(self) -> CaptureQualityResult | None:
        if not self._quality_path.is_file():
            return None
        return CaptureQualityResult.model_validate_json(
            self._quality_path.read_text(encoding="utf-8")
        )

    def publish_preview(self, metadata: PreviewMetadata, jpeg: bytes) -> None:
        with self._lock:
            self._validate_active_sample(metadata)
            if len(jpeg) != metadata.byte_length or not jpeg.startswith(b"\xff\xd8"):
                raise ValueError("preview must be a complete JPEG matching its metadata")
            current = self.preview_metadata()
            if (
                current is not None
                and current.take_id == metadata.take_id
                and metadata.sequence <= current.sequence
            ):
                raise ValueError("preview sequence regressed")
            _atomic_write_bytes(self._preview_path, jpeg)
            _atomic_write_json(self._preview_metadata_path, metadata.model_dump(mode="json"))

    def preview_metadata(self) -> PreviewMetadata | None:
        if not self._preview_metadata_path.is_file():
            return None
        return PreviewMetadata.model_validate_json(
            self._preview_metadata_path.read_text(encoding="utf-8")
        )

    def preview_bytes(self) -> bytes | None:
        if not self._preview_path.is_file():
            return None
        return self._preview_path.read_bytes()

    def receive_chunk(self, chunk: CaptureChunk, stream: BinaryIO) -> ChunkAcknowledgement:
        with self._lock:
            persisted = self._receiver.receive_chunk_stream(chunk, stream)
            return ChunkAcknowledgement(chunk_id=persisted.chunk_id, sha256=persisted.sha256)

    def receiver_index(self) -> list[ChunkAcknowledgement]:
        with self._lock:
            return [
                ChunkAcknowledgement(chunk_id=chunk.chunk_id, sha256=chunk.sha256)
                for chunk in self._receiver.chunks()
            ]

    def finalize_receiver_session(self) -> CaptureSessionManifest:
        with self._lock:
            state = self.state()
            if state.status != "complete":
                raise ValueError("studio session must be ended before its archive is finalized")
            definition = self.definition
            return self._receiver.finalize(
                session_id=definition.session_id,
                subject_id=definition.subject_id,
                device_id=definition.device_id,
                device_model=definition.device_model,
                os_version=definition.os_version,
                ntp_mapping_revision=definition.ntp_mapping_revision,
                consent_record_id=definition.consent_record_id,
                license_record_ids=definition.license_record_ids,
            )

    def state(self) -> StudioState:
        with self._lock:
            definition = self.definition
            status: SessionStatus = "ready"
            current_take_id: str | None = None
            takes: dict[str, StudioTakeState] = {}
            for command in self._commands:
                if command.action in {"start", "retake"}:
                    if command.action == "retake" and command.retake_of is not None:
                        replaced = takes[command.retake_of]
                        takes[command.retake_of] = replaced.model_copy(
                            update={"status": "replaced"}
                        )
                    assert command.take_id is not None
                    assert command.action_script_id is not None
                    take = takes.get(command.take_id)
                    if take is None:
                        take = StudioTakeState(
                            take_id=command.take_id,
                            action_script_id=command.action_script_id,
                            status="recording",
                            retake_of=command.retake_of,
                        )
                    else:
                        take = take.model_copy(update={"status": "recording"})
                    takes[command.take_id] = take
                    current_take_id = command.take_id
                    status = "recording"
                elif command.action == "pause":
                    assert current_take_id is not None
                    takes[current_take_id] = takes[current_take_id].model_copy(
                        update={"status": "paused"}
                    )
                    status = "paused"
                elif command.action == "stop":
                    assert current_take_id is not None
                    takes[current_take_id] = takes[current_take_id].model_copy(
                        update={"status": "stopped"}
                    )
                    status = "stopped"
                else:
                    current_take_id = None
                    status = "complete"
            acknowledged_revision = max(
                (item.revision for item in self._acknowledgements), default=0
            )
            return StudioState(
                definition=definition,
                status=status,
                current_take_id=current_take_id,
                takes=list(takes.values()),
                last_command_revision=len(self._commands),
                acknowledged_revision=acknowledged_revision,
                received_chunk_count=len(self._receiver.chunks()),
                latest_quality=self.latest_quality(),
            )

    def _validate_transition(
        self,
        state: StudioState,
        action: ControlAction,
        *,
        take_id: str | None,
        action_script_id: str | None,
        retake_of: str | None,
    ) -> None:
        if state.status == "complete":
            raise ValueError("completed studio sessions cannot accept new controls")
        takes = {take.take_id: take for take in state.takes}
        if action == "start":
            if state.status == "paused":
                current = takes[state.current_take_id or ""]
                if take_id != current.take_id or action_script_id != current.action_script_id:
                    raise ValueError("resume must target the paused take and action script")
            elif state.status not in {"ready", "stopped"} or not take_id or not action_script_id:
                raise ValueError("start requires a new take while the session is ready or stopped")
            elif take_id in takes:
                raise ValueError("take IDs cannot be reused")
        elif action == "pause":
            if state.status != "recording" or take_id != state.current_take_id:
                raise ValueError("pause must target the recording take")
        elif action == "stop":
            if state.status not in {"recording", "paused"} or take_id != state.current_take_id:
                raise ValueError("stop must target the active take")
        elif action == "retake":
            if state.status not in {"ready", "stopped"}:
                raise ValueError("retake requires an idle session")
            if not take_id or take_id in takes or not action_script_id:
                raise ValueError("retake requires a fresh take and action-script ID")
            replaced = takes.get(retake_of or "")
            if replaced is None or replaced.status != "stopped":
                raise ValueError("retake must replace a stopped take")
        elif state.status not in {"ready", "stopped"}:
            raise ValueError("end requires an idle session")

    def _validate_active_sample(self, sample: CaptureQualitySample | PreviewMetadata) -> None:
        state = self.state()
        if sample.session_id != self.definition.session_id:
            raise ValueError("capture sample session does not match the studio")
        if state.status not in {"recording", "paused"} or sample.take_id != state.current_take_id:
            raise ValueError("capture sample does not belong to the active take")

    def _load_definition(self) -> StudioSessionDefinition | None:
        if not self._definition_path.is_file():
            return None
        return StudioSessionDefinition.model_validate_json(
            self._definition_path.read_text(encoding="utf-8")
        )

    def _validate_history(self) -> None:
        if self._definition is None and (self._commands or self._acknowledgements):
            raise ValueError("studio journals exist without a session definition")
        for index, command in enumerate(self._commands, 1):
            if command.revision != index or command.session_id != self.definition.session_id:
                raise ValueError("studio command journal is not contiguous for this session")
        seen_revisions: set[int] = set()
        commands = {command.revision: command for command in self._commands}
        for acknowledgement in self._acknowledgements:
            command = commands.get(acknowledgement.revision)
            if (
                acknowledgement.revision in seen_revisions
                or command is None
                or command.command_id != acknowledgement.command_id
                or acknowledgement.session_id != self.definition.session_id
                or acknowledgement.device_id != self.definition.device_id
            ):
                raise ValueError("studio acknowledgement journal is inconsistent")
            seen_revisions.add(acknowledgement.revision)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_bytes(path, _canonical_json(payload, pretty=True))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(payload: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
