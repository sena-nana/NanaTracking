import hashlib
import json
from pathlib import Path

from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.evaluation.acceptance import validate_face_basic_acceptance


def _write(path: Path, payload: object) -> dict[str, str]:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_acceptance_gate_rejects_smoke_partial_and_wrong_hardware_evidence(
    tmp_path: Path,
) -> None:
    metadata = ModelPackageMetadata(
        model_family="nana-face-basic",
        model_version="0.1.0-smoke",
        source_checkpoint_digest="checkpoint-digest",
        ntp_schema_revision="ntp/1.0",
        signal_registry_revision="ntp-signals/1.0.0",
        normalization_revision="ntp-normalization/1.0.0",
        calibration_revision="ntp-calibration/1.0.0",
        feature_revision="ntp-features/1.0.0",
        onnx_opset=18,
        input_shape=[1, 3, 64, 64],
        output_names=["rig", "pose", "landmarks", "visibility", "confidence"],
        model_digest="model-digest",
        smoke_only=True,
        supported_signals=list(range(1, 37)),
        supported_structures=["head_geometry"],
    )
    runtime_metadata = _write(tmp_path / "runtime-metadata.json", metadata.model_dump(mode="json"))
    conformance = _write(
        tmp_path / "conformance.json", {"passed": True, "certified_profile": "Basic"}
    )
    quality_payload = json.loads(
        Path("examples/evaluation/benchmark-report-template-v1.json").read_text(encoding="utf-8")
    )
    quality_payload["checkpoint_digest"] = "checkpoint-digest"
    quality_payload["profile"] = "Basic"
    quality_payload["smoke_only"] = True
    quality = _write(tmp_path / "quality.json", quality_payload)
    runtime = _write(
        tmp_path / "runtime.json",
        {
            "smoke_only": True,
            "source_checkpoint_digest": "checkpoint-digest",
            "runtime": {
                "active_providers": ["CPUExecutionProvider"],
                "tensorrt_fp16": False,
            },
            "resources": {"nvidia_smi_snapshot": None},
        },
    )
    bundle = {
        "schema_version": "face-basic-acceptance/1.0.0",
        "runtime_metadata": runtime_metadata,
        "conformance_report": conformance,
        "quality_report": quality,
        "runtime_report": runtime,
        "baselines": {
            "mediapipe": {"status": "measured", "report": quality},
            "maxine": {"status": "unavailable", "reason": "license not approved"},
            "openseeface_vbridge": {"status": "measured", "report": quality},
            "single_frame_postprocess": {"status": "measured", "report": quality},
        },
    }
    bundle_path = tmp_path / "acceptance.json"
    _write(bundle_path, bundle)
    result = validate_face_basic_acceptance(bundle_path)
    codes = {finding.code for finding in result.findings}
    assert result.passed is False
    assert {
        "smoke_model",
        "smoke_quality",
        "quality_metrics_missing",
        "ab_video_missing",
        "wrong_gpu",
        "tensorrt_fp16_missing",
        "runtime_metric_missing",
        "smoke_baseline",
    }.issubset(codes)
