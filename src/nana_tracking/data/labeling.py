"""Deterministic NTP label materialization and dataset quality gates."""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nana_tracking.data.manifest import DatasetManifest, TeacherSource, TeacherSourceType
from nana_tracking.data.schema import CaptureRecord, DepthObservation, LabelObservation

Profile = Literal["Basic", "Spatial", "Full", "Optional"]
ScalarType = Literal["NS", "NU", "GY", "GP", "TT", "HT", "AR"]
ObservationState = Literal["observed", "fused", "predicted"]

_SCALAR_LIMITS: dict[str, tuple[float, float]] = {
    "NS": (-1.0, 1.0),
    "NU": (0.0, 1.0),
    "GY": (-1.2, 1.2),
    "GP": (-0.8, 0.8),
    "TT": (-1.0, 1.0),
    "HT": (-1.0, 1.0),
    "AR": (-math.pi, math.pi),
}
_DISAGREEMENT_LIMITS: dict[str, float] = {
    "NS": 0.15,
    "NU": 0.15,
    "GY": 0.08,
    "GP": 0.08,
    "TT": 0.10,
    "HT": 0.10,
    "AR": 0.12,
}


class LabelModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelStrategy(LabelModel):
    source_types: set[TeacherSourceType] = Field(min_length=1)
    evidence: Literal["teacher_label", "geometry", "derived"]
    method: str = Field(min_length=1)
    accepted_states: set[ObservationState] = Field(min_length=1)
    strict_observation: bool = False

    @model_validator(mode="after")
    def validate_strict_observation(self) -> Self:
        if self.strict_observation and self.accepted_states != {"observed"}:
            raise ValueError("strict observation strategies may accept only observed truth")
        if "predicted" in self.accepted_states:
            raise ValueError("predicted observations cannot be admitted as training truth")
        return self


class SignalMapping(LabelModel):
    signal_id: int = Field(ge=1, le=88)
    stable_name: str = Field(min_length=1)
    scalar_type: ScalarType
    profile: Profile
    strategy: str = Field(min_length=1)


class LabelCatalog(LabelModel):
    schema_version: Literal["ntp-label-catalog/1.0.0"]
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    calibration_revision: str = Field(min_length=1)
    feature_revision: str = Field(min_length=1)
    strategies: dict[str, LabelStrategy]
    signals: list[SignalMapping]

    @model_validator(mode="after")
    def validate_registry_coverage(self) -> Self:
        ids = [signal.signal_id for signal in self.signals]
        names = [signal.stable_name for signal in self.signals]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise ValueError("signal IDs and stable names must be unique")
        if set(ids) != set(range(1, 89)):
            missing = sorted(set(range(1, 89)).difference(ids))
            extra = sorted(set(ids).difference(range(1, 89)))
            raise ValueError(
                f"catalog must cover stable IDs 1..88; missing={missing}, extra={extra}"
            )
        expected_profiles = {
            "Basic": set(range(1, 37)),
            "Spatial": set(range(37, 42)),
            "Full": set(range(42, 77)),
            "Optional": set(range(77, 89)),
        }
        for profile, expected in expected_profiles.items():
            actual = {signal.signal_id for signal in self.signals if signal.profile == profile}
            if actual != expected:
                raise ValueError(f"{profile} signal membership does not match NTP v1")
        unknown = {signal.strategy for signal in self.signals}.difference(self.strategies)
        if unknown:
            raise ValueError(f"signals reference unknown strategies: {sorted(unknown)}")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class QualityIssue(LabelModel):
    severity: Literal["warning", "error"]
    code: str
    record_id: str
    signal_name: str | None = None
    details: str


class ResolvedLabel(LabelModel):
    signal_id: int
    stable_name: str
    profile: Profile
    state: Literal["available", "unavailable"]
    value: float | None
    confidence: float = Field(ge=0.0, le=1.0)
    source_ids: list[str]
    unavailable_reason: str | None


class ResolvedDepth(LabelModel):
    state: Literal["available", "unavailable"]
    confidence: float = Field(ge=0.0, le=1.0)
    source_id: str | None
    depth_uri: str | None
    unavailable_reason: str | None


class MaterializedRecord(LabelModel):
    record_id: str
    identity_id: str
    session_id: str
    device_id: str
    capture_timestamp_ns: int
    sequence: int
    labels: list[ResolvedLabel]
    depth: ResolvedDepth


class QualitySummary(LabelModel):
    schema_version: Literal["ntp-data-quality/1.0.0"] = "ntp-data-quality/1.0.0"
    data_revision: str
    dataset_digest: str
    smoke_only: bool
    record_count: int
    available_label_count: int
    unavailable_label_count: int
    unavailable_reasons: dict[str, int]
    warning_count: int
    error_count: int
    issues: list[QualityIssue]


class MaterializationResult(LabelModel):
    records: list[MaterializedRecord]
    quality: QualitySummary


class _Candidate(LabelModel):
    source_id: str
    observation: LabelObservation


def materialize_dataset(manifest_path: Path) -> MaterializationResult:
    manifest = DatasetManifest.load(manifest_path)
    manifest.verify_files(manifest_path)
    catalog = LabelCatalog.load(manifest.resolve(manifest_path, manifest.label_catalog))
    _validate_revisions(manifest, catalog)
    records = _load_records(manifest_path, manifest)
    issues = _validate_record_contract(manifest, records)
    source_by_id = {source.source_id: source for source in manifest.teacher_sources}
    materialized: list[MaterializedRecord] = []
    for record in records:
        labels = [
            _resolve_signal(record, signal, catalog, source_by_id, manifest, issues)
            for signal in sorted(catalog.signals, key=lambda item: item.signal_id)
        ]
        materialized.append(
            MaterializedRecord(
                record_id=record.record_id,
                identity_id=record.identity_id,
                session_id=record.session_id,
                device_id=record.device_id,
                capture_timestamp_ns=record.capture_timestamp_ns,
                sequence=record.sequence,
                labels=labels,
                depth=_resolve_depth(record, source_by_id, manifest, issues),
            )
        )

    reasons = Counter(
        label.unavailable_reason
        for record in materialized
        for label in record.labels
        if label.unavailable_reason is not None
    )
    available = sum(
        label.state == "available" for record in materialized for label in record.labels
    )
    unavailable = sum(
        label.state == "unavailable" for record in materialized for label in record.labels
    )
    quality = QualitySummary(
        data_revision=manifest.data_revision,
        dataset_digest=manifest.digest,
        smoke_only=manifest.smoke_only,
        record_count=len(records),
        available_label_count=available,
        unavailable_label_count=unavailable,
        unavailable_reasons=dict(sorted(reasons.items())),
        warning_count=sum(issue.severity == "warning" for issue in issues),
        error_count=sum(issue.severity == "error" for issue in issues),
        issues=issues,
    )
    return MaterializationResult(records=materialized, quality=quality)


def write_materialized_labels(result: MaterializationResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
        for record in result.records
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_revisions(manifest: DatasetManifest, catalog: LabelCatalog) -> None:
    pairs = {
        "ntp schema": (manifest.ntp_schema_revision, catalog.ntp_schema_revision),
        "Signal Registry": (
            manifest.signal_registry_revision,
            catalog.signal_registry_revision,
        ),
        "normalization": (manifest.normalization_revision, catalog.normalization_revision),
        "calibration": (manifest.calibration_revision, catalog.calibration_revision),
        "features": (manifest.feature_revision, catalog.feature_revision),
    }
    mismatches = [name for name, (left, right) in pairs.items() if left != right]
    if mismatches:
        raise ValueError(f"manifest and label catalog revision mismatch: {mismatches}")


def _load_records(manifest_path: Path, manifest: DatasetManifest) -> list[CaptureRecord]:
    records: list[CaptureRecord] = []
    for reference in manifest.record_files:
        loaded = CaptureRecord.load_jsonl(manifest.resolve(manifest_path, reference))
        if len(loaded) != reference.record_count:
            raise ValueError(
                f"record count mismatch for {reference.path}: "
                f"expected {reference.record_count}, got {len(loaded)}"
            )
        records.extend(loaded)
    return records


def _validate_record_contract(
    manifest: DatasetManifest, records: list[CaptureRecord]
) -> list[QualityIssue]:
    record_ids: set[str] = set()
    source_ids = {source.source_id for source in manifest.teacher_sources}
    identity_split = {
        identity: split_name
        for split_name, split in manifest.splits.items()
        for identity in split.identities
    }
    last_by_session: dict[str, tuple[int, int]] = {}
    identity_by_session: dict[str, str] = {}
    issues: list[QualityIssue] = []
    for record in records:
        if record.record_id in record_ids:
            raise ValueError(f"duplicate record ID: {record.record_id}")
        record_ids.add(record.record_id)
        split_name = identity_split.get(record.identity_id)
        if split_name is None:
            raise ValueError(f"record identity is not assigned to a split: {record.identity_id}")
        split = manifest.splits[split_name]
        if not manifest.smoke_only and (
            not record.environment_id
            or not record.action_script_id
            or not record.consent_record_id
            or record.human_review_status != "approved"
        ):
            raise ValueError(
                f"production record {record.record_id!r} lacks capture authorization metadata"
            )
        if split.sessions and record.session_id not in split.sessions:
            raise ValueError(f"session {record.session_id!r} is not declared in {split_name!r}")
        if split.devices and record.device_id not in split.devices:
            raise ValueError(f"device {record.device_id!r} is not declared in {split_name!r}")
        previous_identity = identity_by_session.setdefault(record.session_id, record.identity_id)
        if previous_identity != record.identity_id:
            raise ValueError(f"session {record.session_id!r} contains more than one identity")
        unknown_sources = {
            teacher.source_id for teacher in record.teachers if teacher.source_id not in source_ids
        }
        unknown_sources.update(
            depth.source_id for depth in record.depth if depth.source_id not in source_ids
        )
        if unknown_sources:
            raise ValueError(
                f"record {record.record_id!r} uses unknown teacher sources: "
                f"{sorted(unknown_sources)}"
            )
        previous = last_by_session.get(record.session_id)
        if previous is not None:
            previous_sequence, previous_timestamp = previous
            if (
                manifest.synchronization.require_increasing_sequence
                and record.sequence <= previous_sequence
            ):
                raise ValueError(f"sequence did not increase in session {record.session_id!r}")
            if (
                manifest.synchronization.require_monotonic_timestamps
                and record.capture_timestamp_ns <= previous_timestamp
            ):
                raise ValueError(f"timestamp did not increase in session {record.session_id!r}")
        last_by_session[record.session_id] = (record.sequence, record.capture_timestamp_ns)
    return issues


def _resolve_signal(
    record: CaptureRecord,
    signal: SignalMapping,
    catalog: LabelCatalog,
    source_by_id: dict[str, TeacherSource],
    manifest: DatasetManifest,
    issues: list[QualityIssue],
) -> ResolvedLabel:
    strategy = catalog.strategies[signal.strategy]
    candidates: list[_Candidate] = []
    rejected_reason = "not_observed"
    for teacher in record.teachers:
        observation = teacher.labels.get(signal.stable_name)
        if observation is None:
            continue
        source = source_by_id[teacher.source_id]
        if abs(teacher.capture_timestamp_ns - record.capture_timestamp_ns) > (
            manifest.synchronization.max_teacher_skew_ns
        ):
            rejected_reason = "unsynchronized"
            issues.append(
                QualityIssue(
                    severity="warning",
                    code="teacher_timestamp_skew",
                    record_id=record.record_id,
                    signal_name=signal.stable_name,
                    details=f"teacher {teacher.source_id} exceeded the RGB skew limit",
                )
            )
            continue
        if not _matches_strategy(source.source_type, observation, strategy):
            if observation.state not in strategy.accepted_states:
                rejected_reason = "unreliable_observation"
            else:
                rejected_reason = "unapproved_provenance"
            continue
        if observation.value is None or not math.isfinite(observation.value):
            rejected_reason = "invalid_value"
            issues.append(
                QualityIssue(
                    severity="error",
                    code="non_finite_label",
                    record_id=record.record_id,
                    signal_name=signal.stable_name,
                    details=f"teacher {teacher.source_id} emitted a non-finite label",
                )
            )
            continue
        low, high = _SCALAR_LIMITS[signal.scalar_type]
        upper_valid = (
            observation.value < high if signal.scalar_type == "AR" else observation.value <= high
        )
        if observation.value < low or not upper_valid:
            rejected_reason = "invalid_value"
            issues.append(
                QualityIssue(
                    severity="error",
                    code="label_out_of_range",
                    record_id=record.record_id,
                    signal_name=signal.stable_name,
                    details=f"teacher {teacher.source_id} value is outside {signal.scalar_type}",
                )
            )
            continue
        candidates.append(_Candidate(source_id=teacher.source_id, observation=observation))

    if not candidates:
        return _unavailable(signal, rejected_reason)
    candidates.sort(key=lambda item: item.source_id)
    values = [candidate.observation.value for candidate in candidates]
    numeric_values = [value for value in values if value is not None]
    spread = max(numeric_values) - min(numeric_values)
    disagreement_limit = _DISAGREEMENT_LIMITS[signal.scalar_type]
    if spread > disagreement_limit:
        issues.append(
            QualityIssue(
                severity="warning",
                code="teacher_disagreement",
                record_id=record.record_id,
                signal_name=signal.stable_name,
                details=f"approved teachers differ by {spread:.6f}, limit {disagreement_limit:.6f}",
            )
        )
        return _unavailable(signal, "teacher_disagreement")

    weights = [candidate.observation.confidence for candidate in candidates]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return _unavailable(signal, "zero_confidence")
    value = sum(value * weight for value, weight in zip(numeric_values, weights, strict=True))
    value /= total_weight
    confidence = min(weights) * max(0.0, 1.0 - spread / disagreement_limit)
    return ResolvedLabel(
        signal_id=signal.signal_id,
        stable_name=signal.stable_name,
        profile=signal.profile,
        state="available",
        value=value,
        confidence=confidence,
        source_ids=[candidate.source_id for candidate in candidates],
        unavailable_reason=None,
    )


def _matches_strategy(
    source_type: TeacherSourceType,
    observation: LabelObservation,
    strategy: LabelStrategy,
) -> bool:
    return (
        source_type in strategy.source_types
        and observation.evidence == strategy.evidence
        and observation.method == strategy.method
        and observation.state in strategy.accepted_states
    )


def _unavailable(signal: SignalMapping, reason: str) -> ResolvedLabel:
    return ResolvedLabel(
        signal_id=signal.signal_id,
        stable_name=signal.stable_name,
        profile=signal.profile,
        state="unavailable",
        value=None,
        confidence=0.0,
        source_ids=[],
        unavailable_reason=reason,
    )


def _resolve_depth(
    record: CaptureRecord,
    source_by_id: dict[str, TeacherSource],
    manifest: DatasetManifest,
    issues: list[QualityIssue],
) -> ResolvedDepth:
    reliable: list[DepthObservation] = []
    for observation in record.depth:
        source = source_by_id[observation.source_id]
        synchronized = abs(observation.capture_timestamp_ns - record.capture_timestamp_ns) <= (
            manifest.synchronization.max_depth_skew_ns
        )
        if (
            observation.state == "observed"
            and source.source_type in {"truedepth", "multiview_pose"}
            and synchronized
        ):
            reliable.append(observation)
        else:
            issues.append(
                QualityIssue(
                    severity="warning",
                    code="depth_not_observed_truth",
                    record_id=record.record_id,
                    details=(
                        f"depth from {observation.source_id} remains unavailable because it is "
                        "not a synchronized direct observation"
                    ),
                )
            )
    if not reliable:
        return ResolvedDepth(
            state="unavailable",
            confidence=0.0,
            source_id=None,
            depth_uri=None,
            unavailable_reason="not_reliably_observed",
        )
    selected = sorted(reliable, key=lambda item: (-item.confidence, item.source_id))[0]
    return ResolvedDepth(
        state="available",
        confidence=selected.confidence,
        source_id=selected.source_id,
        depth_uri=selected.depth_uri,
        unavailable_reason=None,
    )
