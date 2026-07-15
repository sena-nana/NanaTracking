"""Issue #7 acceptance evidence gate.

This gate deliberately rejects smoke, partial, cross-checkpoint, and non-target-hardware evidence.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.evaluation.standard import REQUIRED_METRICS, BenchmarkReport


class AcceptanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceFile(AcceptanceModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaselineEvidence(AcceptanceModel):
    status: Literal["measured", "unavailable"]
    report: EvidenceFile | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "measured" and self.report is None:
            raise ValueError("measured baselines require a report")
        if self.status == "unavailable" and not self.reason:
            raise ValueError("unavailable baselines require a reason")
        return self


class FaceBasicAcceptanceBundle(AcceptanceModel):
    schema_version: Literal["face-basic-acceptance/1.0.0"]
    runtime_metadata: EvidenceFile
    conformance_report: EvidenceFile
    quality_report: EvidenceFile
    runtime_report: EvidenceFile
    baselines: dict[
        Literal["mediapipe", "maxine", "openseeface_vbridge", "single_frame_postprocess"],
        BaselineEvidence,
    ]

    @model_validator(mode="after")
    def validate_baselines(self) -> Self:
        required = {
            "mediapipe",
            "maxine",
            "openseeface_vbridge",
            "single_frame_postprocess",
        }
        if set(self.baselines) != required:
            raise ValueError("acceptance bundle must account for every required baseline")
        for name, evidence in self.baselines.items():
            if name != "maxine" and evidence.status != "measured":
                raise ValueError(f"baseline {name} must be measured")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class AcceptanceFinding(AcceptanceModel):
    code: str
    details: str


class AcceptanceResult(AcceptanceModel):
    passed: bool
    findings: list[AcceptanceFinding]


def _load_evidence(owner: Path, evidence: EvidenceFile) -> tuple[Path, str]:
    path = (owner.parent / evidence.path).resolve()
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != evidence.sha256:
        raise ValueError(f"evidence digest mismatch for {evidence.path}")
    return path, payload.decode("utf-8")


def _finding(findings: list[AcceptanceFinding], code: str, details: str) -> None:
    findings.append(AcceptanceFinding(code=code, details=details))


def _complete_quality_metrics(report: BenchmarkReport) -> set[str]:
    return {
        metric_id
        for family in report.output_families.values()
        for metric_id, result in family.items()
        if result.status == "measured"
    }


def _has_finite_percentiles(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    for name in ("p50", "p95", "p99"):
        item = value.get(name)
        if not isinstance(item, int | float) or not math.isfinite(float(item)):
            return False
    return True


def validate_face_basic_acceptance(bundle_path: Path) -> AcceptanceResult:
    bundle = FaceBasicAcceptanceBundle.load(bundle_path)
    metadata_path, metadata_json = _load_evidence(bundle_path, bundle.runtime_metadata)
    del metadata_path
    metadata = ModelPackageMetadata.model_validate_json(metadata_json)
    _, conformance_json = _load_evidence(bundle_path, bundle.conformance_report)
    conformance = json.loads(conformance_json)
    _, quality_json = _load_evidence(bundle_path, bundle.quality_report)
    quality = BenchmarkReport.model_validate_json(quality_json)
    _, runtime_json = _load_evidence(bundle_path, bundle.runtime_report)
    runtime = json.loads(runtime_json)
    findings: list[AcceptanceFinding] = []

    if metadata.smoke_only:
        _finding(findings, "smoke_model", "model package is smoke-only")
    if metadata.supported_signals != list(range(1, 37)):
        _finding(findings, "incomplete_basic", "package does not declare Basic IDs 1..36")
    if metadata.supported_structures != ["head_geometry"]:
        _finding(findings, "missing_head_pose", "package does not declare head geometry")
    if conformance.get("passed") is not True or conformance.get("certified_profile") != "Basic":
        _finding(findings, "conformance_failed", "NTP report does not certify Basic")

    if quality.smoke_only:
        _finding(findings, "smoke_quality", "quality report is smoke-only")
    if quality.profile != "Basic":
        _finding(findings, "quality_profile_mismatch", "quality report is not a Basic suite")
    if quality.checkpoint_digest != metadata.source_checkpoint_digest:
        _finding(findings, "quality_checkpoint_mismatch", "quality and package checkpoints differ")
    missing_metrics = REQUIRED_METRICS.difference(_complete_quality_metrics(quality))
    if missing_metrics:
        _finding(
            findings,
            "quality_metrics_missing",
            f"unmeasured required metrics: {sorted(missing_metrics)}",
        )
    if not quality.ab_video_uris:
        _finding(findings, "ab_video_missing", "NanaLive A/B video evidence is absent")

    if runtime.get("smoke_only") is not False:
        _finding(findings, "smoke_runtime", "runtime report is absent or smoke-only")
    if runtime.get("source_checkpoint_digest") != metadata.source_checkpoint_digest:
        _finding(findings, "runtime_checkpoint_mismatch", "runtime and package checkpoints differ")
    runtime_contract = runtime.get("runtime", {})
    resources = runtime.get("resources", {})
    gpu = resources.get("nvidia_smi_snapshot") if isinstance(resources, dict) else None
    gpu_name = gpu.get("name", "") if isinstance(gpu, dict) else ""
    if "RTX 4060" not in gpu_name:
        _finding(findings, "wrong_gpu", "runtime evidence was not measured on an RTX 4060")
    if not isinstance(runtime_contract, dict) or (
        "TensorrtExecutionProvider" not in runtime_contract.get("active_providers", [])
        or runtime_contract.get("tensorrt_fp16") is not True
    ):
        _finding(findings, "tensorrt_fp16_missing", "TensorRT FP16 was not active")
    for family in ("capture_to_result_ms", "result_age_at_consume_ms"):
        values = runtime.get(family)
        if not _has_finite_percentiles(values):
            _finding(findings, "runtime_metric_missing", f"{family} percentiles are incomplete")

    for name, evidence in bundle.baselines.items():
        if evidence.status != "measured":
            continue
        assert evidence.report is not None
        _, baseline_json = _load_evidence(bundle_path, evidence.report)
        baseline = BenchmarkReport.model_validate_json(baseline_json)
        if baseline.smoke_only:
            _finding(findings, "smoke_baseline", f"baseline {name} is smoke-only")
        if baseline.profile != "Basic":
            _finding(findings, "baseline_profile_mismatch", f"baseline {name} is not Basic")
        if (baseline.data_revision, baseline.data_digest) != (
            quality.data_revision,
            quality.data_digest,
        ):
            _finding(findings, "baseline_data_mismatch", f"baseline {name} used different data")

    return AcceptanceResult(passed=not findings, findings=findings)
