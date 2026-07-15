"""Versioned, profile-independent tracking evaluation standard."""

import hashlib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nana_tracking.data.manifest import FileReference

Profile = Literal["Basic", "Spatial", "Full"]

REQUIRED_METRICS: set[str] = {
    "per_signal_error",
    "per_signal_correlation",
    "neutral_jitter",
    "dynamic_response_delay",
    "peak_attenuation",
    "left_right_asymmetry",
    "identity_neutral_bias",
    "geometry_consistency",
    "state_classification",
    "occlusion_recovery_time",
    "confidence_calibration",
    "capture_to_result_latency",
    "result_age_at_consume",
    "runtime_resources",
    "character_drive_ab_video",
}


class StandardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FiniteMetric = Annotated[float, Field(allow_inf_nan=False)]


class MetricDefinition(StandardModel):
    metric_id: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    aggregations: list[str] = Field(min_length=1)
    required_metadata: list[str] = Field(default_factory=list)


class EvaluationSuite(StandardModel):
    profile: Profile
    required_signal_ids: list[int] = Field(min_length=1)
    metric_ids: list[str] = Field(min_length=1)
    fixed_sequence_ids: list[str] = Field(min_length=1)


class FixedSequence(StandardModel):
    sequence_id: str = Field(min_length=1)
    profiles: set[Profile] = Field(min_length=1)
    category: str = Field(min_length=1)
    action: str = Field(min_length=1)
    conditions: list[str] = Field(min_length=1)
    duration_seconds: float = Field(gt=0)
    repetitions: int = Field(gt=0)


class FixedSequenceCatalog(StandardModel):
    schema_version: Literal["ntp-fixed-sequences/1.0.0"]
    standard_revision: str = Field(min_length=1)
    sequences: list[FixedSequence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sequence_ids(self) -> Self:
        ids = [sequence.sequence_id for sequence in self.sequences]
        if len(ids) != len(set(ids)):
            raise ValueError("fixed sequence IDs must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class MetricResult(StandardModel):
    status: Literal["pending", "measured", "unavailable"]
    values: dict[str, FiniteMetric] | None = None
    sample_count: int | None = Field(default=None, gt=0)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "measured" and (not self.values or self.sample_count is None):
            raise ValueError("measured metrics require named values and a positive sample count")
        if self.status == "unavailable" and not self.reason:
            raise ValueError("unavailable metrics require a reason")
        if self.status != "measured" and (self.values is not None or self.sample_count is not None):
            raise ValueError("only measured metrics may contain values or a sample count")
        if self.status == "pending" and self.reason is not None:
            raise ValueError("pending metrics cannot contain a reason")
        return self


class BenchmarkReport(StandardModel):
    schema_version: Literal["ntp-benchmark-report/1.0.0"]
    standard_revision: str = Field(min_length=1)
    profile: Profile
    checkpoint_digest: str = Field(min_length=1)
    data_revision: str = Field(min_length=1)
    data_digest: str = Field(min_length=1)
    config_digest: str = Field(min_length=1)
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    calibration_revision: str = Field(min_length=1)
    feature_revision: str = Field(min_length=1)
    git_commit: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    smoke_only: bool
    output_families: dict[str, dict[str, MetricResult]] = Field(min_length=1)
    failure_sample_ids: list[str]
    ab_video_uris: list[str]


class EvaluationStandard(StandardModel):
    schema_version: Literal["ntp-evaluation-standard/1.0.0"]
    standard_revision: str = Field(min_length=1)
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    calibration_revision: str = Field(min_length=1)
    feature_revision: str = Field(min_length=1)
    fixed_sequences: FileReference
    report_template: FileReference
    metrics: list[MetricDefinition]
    suites: list[EvaluationSuite]

    @model_validator(mode="after")
    def validate_shared_standard(self) -> Self:
        metric_ids = [metric.metric_id for metric in self.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("metric IDs must be unique")
        if set(metric_ids) != REQUIRED_METRICS:
            raise ValueError("evaluation standard does not define the complete required metric set")
        suites = {suite.profile: suite for suite in self.suites}
        if set(suites) != {"Basic", "Spatial", "Full"} or len(suites) != len(self.suites):
            raise ValueError("the standard requires exactly one Basic, Spatial, and Full suite")
        expected_signals = {
            "Basic": list(range(1, 37)),
            "Spatial": list(range(1, 42)),
            "Full": list(range(1, 77)),
        }
        for profile, suite in suites.items():
            if suite.required_signal_ids != expected_signals[profile]:
                raise ValueError(f"{profile} required signals do not match NTP v1")
            if set(suite.metric_ids) != REQUIRED_METRICS:
                raise ValueError(f"{profile} does not use the shared required metric set")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def validate_assets(self, standard_path: Path) -> tuple[FixedSequenceCatalog, BenchmarkReport]:
        sequence_path = _verify_reference(standard_path, self.fixed_sequences)
        template_path = _verify_reference(standard_path, self.report_template)
        sequences = FixedSequenceCatalog.load(sequence_path)
        report = BenchmarkReport.model_validate_json(template_path.read_text(encoding="utf-8"))
        if sequences.standard_revision != self.standard_revision:
            raise ValueError("fixed sequences use a different evaluation standard revision")
        if report.standard_revision != self.standard_revision:
            raise ValueError("report template uses a different evaluation standard revision")
        revision_pairs = {
            "NTP schema": (report.ntp_schema_revision, self.ntp_schema_revision),
            "Signal Registry": (
                report.signal_registry_revision,
                self.signal_registry_revision,
            ),
            "normalization": (report.normalization_revision, self.normalization_revision),
            "calibration": (report.calibration_revision, self.calibration_revision),
            "features": (report.feature_revision, self.feature_revision),
        }
        mismatches = [name for name, (left, right) in revision_pairs.items() if left != right]
        if mismatches:
            raise ValueError(f"report template revision mismatch: {mismatches}")
        sequence_by_id = {sequence.sequence_id: sequence for sequence in sequences.sequences}
        for suite in self.suites:
            for sequence_id in suite.fixed_sequence_ids:
                sequence = sequence_by_id.get(sequence_id)
                if sequence is None:
                    raise ValueError(f"suite references unknown fixed sequence {sequence_id!r}")
                if suite.profile not in sequence.profiles:
                    raise ValueError(
                        f"fixed sequence {sequence_id!r} does not apply to {suite.profile}"
                    )
        template_metric_ids = {
            metric_id for family in report.output_families.values() for metric_id in family
        }
        if template_metric_ids != REQUIRED_METRICS:
            raise ValueError("report template does not expose the complete required metric set")
        return sequences, report


def _verify_reference(owner_path: Path, reference: FileReference) -> Path:
    path = (owner_path.parent / reference.path).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != reference.sha256:
        raise ValueError(
            f"digest mismatch for {reference.path}: expected {reference.sha256}, got {actual}"
        )
    return path
