import hashlib
import importlib.metadata
import json
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torchvision.io import write_png

from nana_tracking.config import load_config
from nana_tracking.data.loaders import create_loader
from nana_tracking.data.manifest import SplitManifest
from nana_tracking.data.schema import CaptureConditions, RgbFrame
from nana_tracking.data.stage_a import (
    CameraRigCalibrationBundle,
    MaterializedStageAGroup,
    OverlayReviewDecision,
    OverlayReviewIndex,
    ReviewApproval,
    RigCameraCalibration,
    StageACaptureGroup,
    StageAInputView,
    StageAQualityProfile,
    StageARecordReview,
    StageASplitPlan,
    SubjectHeadScale,
    build_stage_a_manifest,
    materialize_stage_a_labels,
    validate_stage_a_manifest,
)
from nana_tracking.data.teachers import SEMANTIC_ANCHORS, TeacherModelDescriptor

_HEAD_POINTS = np.asarray(
    [
        (0.03, -0.02, 0.00),
        (0.08, -0.02, 0.01),
        (-0.03, -0.02, 0.00),
        (-0.08, -0.02, 0.01),
        (0.02, -0.06, 0.01),
        (0.09, -0.05, 0.02),
        (-0.02, -0.06, 0.01),
        (-0.09, -0.05, 0.02),
        (0.00, 0.01, -0.18),
        (0.03, 0.02, -0.08),
        (-0.03, 0.02, -0.08),
        (0.06, 0.06, 0.02),
        (-0.06, 0.06, 0.02),
        (0.00, 0.055, -0.02),
        (0.00, 0.075, -0.01),
        (0.00, 0.13, 0.03),
    ],
    dtype=np.float64,
)
_INTRINSICS = ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))


class _QueueDetector:
    def __init__(self, values: list[list[tuple[float, float]]]) -> None:
        self.values = deque(values)

    def detect(self, rgb: np.ndarray[Any, np.dtype[np.uint8]]) -> list[tuple[float, float]]:
        assert rgb.shape == (480, 640, 3)
        return self.values.popleft()

    def close(self) -> None:
        pass


def _pnp_failure(
    *_args: object,
    **_kwargs: object,
) -> tuple[bool, None, None, None]:
    return False, None, None, None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_model(model: Any, path: Path) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record_review(
    record_id: str,
    *,
    action: Literal["accepted", "rejected"] = "accepted",
) -> StageARecordReview:
    return StageARecordReview(
        record_id=record_id,
        action=action,
        reviewed_by="test-reviewer",
        reviewed_at=datetime.fromisoformat("2026-07-27T00:00:00+08:00"),
        evidence_sha256=hashlib.sha256(f"review:{record_id}".encode()).hexdigest(),
        reviewed_anchor_names=list(SEMANTIC_ANCHORS) if action != "rejected" else [],
    )


def _camera(camera_id: str, center_x: float) -> RigCameraCalibration:
    return RigCameraCalibration(
        camera_id=camera_id,
        intrinsics=_INTRINSICS,
        distortion_model="none",
        distortion_coefficients=[],
        camera_to_capture=(
            (1.0, 0.0, 0.0, center_x),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _project(center_x: float, origin_z: float = 1.0) -> list[tuple[float, float]]:
    capture = _HEAD_POINTS * 0.2 + np.asarray((0.0, 0.0, origin_z))
    camera = capture - np.asarray((center_x, 0.0, 0.0))
    pixels = np.column_stack(
        (
            500.0 * camera[:, 0] / camera[:, 2] + 320.0,
            500.0 * camera[:, 1] / camera[:, 2] + 240.0,
        )
    )
    return [(float(x / 640.0), float(y / 480.0)) for x, y in pixels]


def _approved_registry(tmp_path: Path) -> Path:
    payload = json.loads(Path("configs/data/license-registry.json").read_text(encoding="utf-8"))
    capture = next(
        record for record in payload["records"] if record["record_id"] == "self-captured-consented"
    )
    capture.update(
        {
            "version": "reviewed-test-batch",
            "license": "https://example.invalid/reviewed-consent",
            "license_text_sha256": "9" * 64,
            "review_status": "approved",
            "allowed_pipeline_stages": ["teacher-labeling", "base-model-training"],
            "evidence": "Synthetic contract test standing in for a reviewed first-party batch.",
            "reviewed_by": "test-reviewer",
            "reviewed_at": "2026-07-27T00:00:00+08:00",
        }
    )
    for permission in capture["permissions"]:
        capture["permissions"][permission] = True
    for record in payload["records"]:
        if record["review_status"] == "approved" and "://" not in record["license"]:
            record["license"] = f"https://example.invalid/{record['record_id']}"
    path = tmp_path / "license-registry.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _detector_factory(
    values: list[list[tuple[float, float]]],
) -> Callable[[Path, TeacherModelDescriptor, StageAQualityProfile], _QueueDetector]:
    def create(
        _model_asset: Path,
        _descriptor: TeacherModelDescriptor,
        _profile: StageAQualityProfile,
    ) -> _QueueDetector:
        return _QueueDetector(values)

    return create


def _inputs(
    tmp_path: Path,
    *,
    group_specs: list[tuple[str, str, str, float]],
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    list[list[tuple[float, float]]],
    list[StageACaptureGroup],
]:
    model_asset = tmp_path / "face-landmarker.task"
    model_asset.write_bytes(b"synthetic-pinned-bundle")
    descriptor_payload = json.loads(
        Path("configs/data/mediapipe-face-landmarker-v1.json").read_text(encoding="utf-8")
    )
    descriptor_payload["model"]["sha256"] = _sha256(model_asset)
    descriptor_path = tmp_path / "teacher.json"
    descriptor_path.write_text(
        json.dumps(descriptor_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cameras: list[RigCameraCalibration] = []
    scales: list[SubjectHeadScale] = []
    groups: list[StageACaptureGroup] = []
    detector_values: list[list[tuple[float, float]]] = []
    positions = (-0.15, 0.0, 0.15)
    roles = ("front", "left", "right")
    for group_index, (record_id, identity_id, split_name, effective_fps) in enumerate(group_specs):
        views: list[StageAInputView] = []
        for role, center_x in zip(roles, positions, strict=True):
            camera_id = f"{split_name}-{role}-camera"
            device_id = f"{split_name}-{role}-device"
            image_path = tmp_path / f"{camera_id}.png"
            write_png(torch.zeros((3, 480, 640), dtype=torch.uint8), str(image_path))
            cameras.append(_camera(camera_id, center_x))
            detector_values.append(_project(center_x))
            views.append(
                StageAInputView(
                    role=role,
                    camera_id=camera_id,
                    device_id=device_id,
                    capture_timestamp_ns=1_000_000_000 + group_index * 10_000_000,
                    rgb=RgbFrame(
                        uri=str(image_path),
                        width=640,
                        height=480,
                        exposure_duration_ns=1_000_000,
                        iso=100.0,
                        frame_duration_ns=33_333_333,
                    ),
                )
            )
        scales.append(SubjectHeadScale(identity_id=identity_id, head_width_m=0.2))
        groups.append(
            StageACaptureGroup(
                record_id=record_id,
                identity_id=identity_id,
                session_id=f"{split_name}-session",
                take_id=f"{split_name}-take",
                environment_id=f"{split_name}-environment",
                action_script_id="face-basic-stage-a",
                consent_record_id=f"{identity_id}-consent",
                expression="neutral",
                capture_timestamp_ns=1_000_000_000 + group_index * 10_000_000,
                sequence=group_index,
                source_fps=30.0,
                effective_fps=effective_fps,
                conditions=CaptureConditions(lighting="normal"),
                views=views,
            )
        )
    capture_index = tmp_path / "capture-index.jsonl"
    capture_index.write_text(
        "\n".join(group.model_dump_json() for group in groups) + "\n",
        encoding="utf-8",
    )
    approval = ReviewApproval(
        status="approved",
        reviewed_by="test-reviewer",
        reviewed_at=datetime.fromisoformat("2026-07-27T00:00:00+08:00"),
        evidence_sha256="1" * 64,
    )
    calibration = CameraRigCalibrationBundle(
        calibration_revision="synthetic-reviewed-rig/1",
        cameras=cameras,
        subject_scales=scales,
        review=approval,
    )
    calibration_path = tmp_path / "calibration.json"
    _write_model(calibration, calibration_path)
    profile = StageAQualityProfile(
        profile_revision="synthetic-reviewed-quality/1",
        max_view_skew_ns=5_000_000,
        pseudo_label_confidence=0.5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_triangulation_angle_degrees=1.0,
        min_depth_m=0.2,
        max_depth_m=2.0,
        max_reprojection_rms_px=1.0,
        max_reprojection_p95_px=1.5,
        max_reprojection_px=2.0,
        pnp_reprojection_error_px=2.0,
        pnp_iterations=1000,
        min_pnp_inliers=12,
        max_capture_pose_translation_head_widths=0.05,
        max_capture_pose_rotation_degrees=1.0,
        overlay_review_sample_count=2,
        review=approval,
    )
    profile_path = tmp_path / "quality-profile.json"
    _write_model(profile, profile_path)
    return (
        capture_index,
        descriptor_path,
        model_asset,
        calibration_path,
        profile_path,
        detector_values,
        groups,
    )


def _materialize(
    tmp_path: Path,
    *,
    groups: list[tuple[str, str, str, float]] | None = None,
    registry: Path | None = None,
) -> tuple[Path, Path, list[MaterializedStageAGroup], tuple[Path, Path, Path, Path]]:
    specs = groups or [("train-record", "identity-0", "train", 5.0)]
    (
        capture_index,
        descriptor,
        model_asset,
        calibration,
        profile,
        detector_values,
        _,
    ) = _inputs(tmp_path, group_specs=specs)
    output = tmp_path / "materialized"
    materialize_stage_a_labels(
        capture_index,
        teacher_descriptor=descriptor,
        model_asset=model_asset,
        calibration_path=calibration,
        quality_profile_path=profile,
        license_registry_path=registry or _approved_registry(tmp_path),
        capture_license_record_id="self-captured-consented",
        output_directory=output,
        detector_factory=_detector_factory(detector_values),
    )
    candidates = output / "stage-a-candidates.jsonl"
    return (
        candidates,
        output / "overlay-review-index.json",
        MaterializedStageAGroup.load_jsonl(candidates),
        (descriptor, calibration, profile, capture_index),
    )


def test_runtime_bundle_mapping_and_single_cv2_provider_are_pinned() -> None:
    descriptor = TeacherModelDescriptor.load(Path("configs/data/mediapipe-face-landmarker-v1.json"))
    assert importlib.metadata.version("mediapipe") == "0.10.35"
    assert importlib.metadata.version("opencv-contrib-python") == "5.0.0.93"
    assert cv2.__version__ == "5.0.0"
    assert (
        tuple(binding.semantic_name for binding in descriptor.anchor_bindings) == SEMANTIC_ANCHORS
    )
    assert descriptor.model.sha256 == (
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
    )
    installed = {
        distribution.metadata["Name"].lower()
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
        and distribution.metadata["Name"].lower().startswith("opencv")
    }
    assert installed == {"opencv-contrib-python"}


def test_synthetic_triangulation_pose_confidence_and_overlays_are_deterministic(
    tmp_path: Path,
) -> None:
    first, first_index, groups, _ = _materialize(tmp_path)
    first_payload = first.read_bytes()
    first_overlay_digest = OverlayReviewIndex.load(first_index).overlay_sha256
    second, second_index, repeated, _ = _materialize(tmp_path)
    accepted = groups[0]
    assert accepted.status == "accepted"
    assert accepted.canonical_geometry.state == "available"
    assert accepted.canonical_geometry.confidence == 0.5
    assert all(view.pose.state == "available" for view in accepted.views)
    assert all(view.pose.confidence == 0.5 for view in accepted.views)
    assert all(label.confidence == 0.5 for view in accepted.views for label in view.landmarks_2d)
    geometry = np.asarray(accepted.canonical_geometry.points_head_relative)
    left_center = geometry[[0, 1]].mean(axis=0)
    right_center = geometry[[2, 3]].mean(axis=0)
    np.testing.assert_allclose((left_center + right_center) * 0.5, 0.0, atol=1e-5)
    assert left_center[0] > right_center[0]
    assert geometry[15, 1] > 0.0
    assert [group.model_dump() for group in groups] == [group.model_dump() for group in repeated]
    assert first_payload == second.read_bytes()
    assert first_overlay_digest == OverlayReviewIndex.load(second_index).overlay_sha256


def test_pending_capture_and_unreviewed_calibration_fail_closed(tmp_path: Path) -> None:
    (
        capture_index,
        descriptor,
        model_asset,
        calibration,
        profile,
        detector_values,
        _,
    ) = _inputs(
        tmp_path,
        group_specs=[("train-record", "identity-0", "train", 5.0)],
    )
    with pytest.raises(ValueError, match="license record is not approved"):
        materialize_stage_a_labels(
            capture_index,
            teacher_descriptor=descriptor,
            model_asset=model_asset,
            calibration_path=calibration,
            quality_profile_path=profile,
            license_registry_path=Path("configs/data/license-registry.json"),
            capture_license_record_id="self-captured-consented",
            output_directory=tmp_path / "rejected",
            detector_factory=_detector_factory(detector_values),
        )
    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    approved_calibration_review = calibration_payload["review"]
    calibration_payload["review"] = {"status": "pending"}
    calibration.write_text(json.dumps(calibration_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed camera"):
        materialize_stage_a_labels(
            capture_index,
            teacher_descriptor=descriptor,
            model_asset=model_asset,
            calibration_path=calibration,
            quality_profile_path=profile,
            license_registry_path=_approved_registry(tmp_path),
            capture_license_record_id="self-captured-consented",
            output_directory=tmp_path / "unreviewed",
            detector_factory=_detector_factory(detector_values),
        )
    calibration_payload["review"] = approved_calibration_review
    calibration.write_text(json.dumps(calibration_payload), encoding="utf-8")
    profile_payload = json.loads(profile.read_text(encoding="utf-8"))
    profile_payload["review"] = {"status": "pending"}
    profile.write_text(json.dumps(profile_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed quality profile"):
        materialize_stage_a_labels(
            capture_index,
            teacher_descriptor=descriptor,
            model_asset=model_asset,
            calibration_path=calibration,
            quality_profile_path=profile,
            license_registry_path=_approved_registry(tmp_path),
            capture_license_record_id="self-captured-consented",
            output_directory=tmp_path / "unreviewed-quality",
            detector_factory=_detector_factory(detector_values),
        )


@pytest.mark.parametrize(
    ("mutation", "failure_code"),
    [
        ("missing", "missing_view"),
        ("skew", "view_timestamp_skew"),
        ("duplicate-camera", "duplicate_camera"),
        ("out-of-frame", "mediapipe_landmarks_out_of_frame"),
        ("detection-failed", "mediapipe_detection_failed"),
        ("low-angle", "triangulation_angle_low"),
        ("negative-depth", "triangulation_depth_invalid"),
        ("high-residual", "reprojection_threshold"),
        ("pnp-failed", "pnp_failed"),
    ],
)
def test_materialization_failures_emit_null_zero_labels(
    tmp_path: Path,
    mutation: str,
    failure_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        capture_index,
        descriptor,
        model_asset,
        calibration,
        profile,
        detector_values,
        groups,
    ) = _inputs(
        tmp_path,
        group_specs=[("record", "identity-0", "train", 5.0)],
    )
    group = groups[0]
    if mutation == "missing":
        group = group.model_copy(update={"views": group.views[:2]})
    elif mutation == "skew":
        views = list(group.views)
        views[2] = views[2].model_copy(
            update={"capture_timestamp_ns": views[2].capture_timestamp_ns + 5_000_001}
        )
        group = group.model_copy(update={"views": views})
    elif mutation == "duplicate-camera":
        views = list(group.views)
        views[2] = views[2].model_copy(update={"camera_id": views[0].camera_id})
        group = group.model_copy(update={"views": views})
    elif mutation == "out-of-frame":
        detector_values[0][0] = (1.01, 0.5)
    elif mutation == "detection-failed":
        detector_values.clear()
    elif mutation == "low-angle":
        centers = (-0.0001, 0.0, 0.0001)
        detector_values[:] = [_project(center) for center in centers]
        payload = json.loads(calibration.read_text(encoding="utf-8"))
        for camera, center in zip(payload["cameras"], centers, strict=True):
            camera["camera_to_capture"][0][3] = center
        calibration.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "negative-depth":
        detector_values[:] = [_project(center, -1.0) for center in (-0.15, 0.0, 0.15)]
    elif mutation == "high-residual":
        detector_values[0][0] = (
            detector_values[0][0][0] + 1.0 / 640.0,
            detector_values[0][0][1],
        )
        payload = json.loads(profile.read_text(encoding="utf-8"))
        payload.update(
            {
                "max_reprojection_rms_px": 0.1,
                "max_reprojection_p95_px": 0.15,
                "max_reprojection_px": 0.2,
            }
        )
        profile.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "pnp-failed":
        monkeypatch.setattr(cv2, "solvePnPRansac", _pnp_failure)
    capture_index.write_text(group.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "output"
    materialize_stage_a_labels(
        capture_index,
        teacher_descriptor=descriptor,
        model_asset=model_asset,
        calibration_path=calibration,
        quality_profile_path=profile,
        license_registry_path=_approved_registry(tmp_path),
        capture_license_record_id="self-captured-consented",
        output_directory=output,
        detector_factory=_detector_factory(detector_values),
    )
    rejected = MaterializedStageAGroup.load_jsonl(output / "stage-a-candidates.jsonl")[0]
    assert rejected.status == "rejected"
    assert rejected.failure_codes == [failure_code]
    assert rejected.canonical_geometry.points_head_relative is None
    assert rejected.canonical_geometry.confidence == 0.0
    assert all(view.pose.translation_head_relative is None for view in rejected.views)
    assert all(view.pose.orientation_xyzw is None for view in rejected.views)
    assert all(view.pose.confidence == 0.0 for view in rejected.views)
    if mutation == "high-residual":
        assert all(
            label.state == "available" and label.confidence == 0.5
            for view in rejected.views
            for label in view.landmarks_2d
        )
    assert len(OverlayReviewIndex.load(output / "overlay-review-index.json").artifacts) == 3


@pytest.mark.parametrize("dimension", ["identities", "sessions", "devices", "camera_ids"])
def test_split_plan_rejects_every_cross_split_leakage(dimension: str) -> None:
    plan = {
        "train": {
            "identities": [f"identity-{index}" for index in range(5)],
            "sessions": ["train-session"],
            "devices": ["train-front", "train-left", "train-right"],
            "camera_ids": ["train-front", "train-left", "train-right"],
        },
        "validation": {
            "identities": ["identity-5"],
            "sessions": ["validation-session"],
            "devices": ["validation-front", "validation-left", "validation-right"],
            "camera_ids": ["validation-front", "validation-left", "validation-right"],
        },
        "test": {
            "identities": ["identity-6", "identity-7"],
            "sessions": ["test-session"],
            "devices": ["test-front", "test-left", "test-right"],
            "camera_ids": ["test-front", "test-left", "test-right"],
        },
    }
    plan["validation"][dimension][0] = plan["train"][dimension][0]
    with pytest.raises(ValidationError, match="appears in both"):
        StageASplitPlan.model_validate({"splits": plan})


def test_reviewed_manifest_gates_record_evidence_digests_and_canonical_boundary(
    tmp_path: Path,
) -> None:
    specs = [
        ("train-record", "identity-0", "train", 5.0),
        ("validation-record", "identity-5", "validation", 15.0),
        ("test-record", "identity-6", "test", 30.0),
    ]
    candidates, overlay_index_path, _, paths = _materialize(tmp_path, groups=specs)
    descriptor, calibration, profile, _ = paths
    registry = tmp_path / "license-registry.json"
    index = OverlayReviewIndex.load(overlay_index_path)
    review = OverlayReviewDecision(
        materialization_sha256=index.materialization_sha256,
        overlay_sha256=index.overlay_sha256,
        record_reviews=[_record_review(record_id) for record_id, *_ in specs],
        review=ReviewApproval(
            status="approved",
            reviewed_by="test-reviewer",
            reviewed_at=datetime.fromisoformat("2026-07-27T00:00:00+08:00"),
            evidence_sha256="2" * 64,
        ),
    )
    review_path = tmp_path / "overlay-review.json"
    _write_model(review, review_path)
    plan = StageASplitPlan(
        splits={
            "train": SplitManifest(
                identities=[f"identity-{index}" for index in range(5)],
                sessions=["train-session"],
                devices=[f"train-{role}-device" for role in ("front", "left", "right")],
                camera_ids=[f"train-{role}-camera" for role in ("front", "left", "right")],
            ),
            "validation": SplitManifest(
                identities=["identity-5"],
                sessions=["validation-session"],
                devices=[f"validation-{role}-device" for role in ("front", "left", "right")],
                camera_ids=[f"validation-{role}-camera" for role in ("front", "left", "right")],
            ),
            "test": SplitManifest(
                identities=["identity-6", "identity-7"],
                sessions=["test-session"],
                devices=[f"test-{role}-device" for role in ("front", "left", "right")],
                camera_ids=[f"test-{role}-camera" for role in ("front", "left", "right")],
            ),
        }
    )
    plan_path = tmp_path / "split-plan.json"
    _write_model(plan, plan_path)
    manifest_path = tmp_path / "stage-a-manifest.json"
    build_stage_a_manifest(
        candidates,
        overlay_index_path=overlay_index_path,
        overlay_review_path=review_path,
        split_plan_path=plan_path,
        teacher_descriptor_path=descriptor,
        calibration_path=calibration,
        quality_profile_path=profile,
        license_registry_path=registry,
        label_catalog_path=Path("configs/data/ntp-v1-label-catalog.json").resolve(),
        training_recipe_path=Path("configs/training/nana-training-recipe-1.0.0.json").resolve(),
        capture_license_record_id="self-captured-consented",
        data_revision="synthetic-contract-test-only",
        output_path=manifest_path,
    )
    validation = validate_stage_a_manifest(manifest_path)
    assert validation["split_record_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert validation["canonical_core16_candidate_ready"] is True
    assert validation["production_model_ready"] is False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = manifest_path.parent / manifest["record_files"][0]["path"]
    approved = MaterializedStageAGroup.load_jsonl(shard_path)
    assert all(group.human_review is not None for group in approved)
    assert all(group.review_status == "approved" for group in approved)
    assert all(
        label.evidence == "human_corrected_pseudo_label"
        for group in approved
        for view in group.views
        for label in view.landmarks_2d
    )
    assert all(group.canonical_geometry.evidence == "deterministic_geometry" for group in approved)
    base = load_config(Path("configs/face-basic-stage-a-smoke.yaml"))
    legacy_config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "dataset": "manifest",
                    "usage_tier": "commercial",
                    "manifest": manifest_path,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="future HR-Canonical loader"):
        create_loader(legacy_config, split="train", shuffle=False)

    calibration_payload = calibration.read_bytes()
    calibration.write_bytes(calibration_payload + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_stage_a_manifest(manifest_path)
    calibration.write_bytes(calibration_payload)

    overlay = overlay_index_path.parent / index.artifacts[0].path
    overlay.write_bytes(overlay.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="overlay artifact digest mismatch"):
        index.verify_files(overlay_index_path)


def test_aggregate_review_cannot_bulk_approve_unsampled_records(tmp_path: Path) -> None:
    specs = [
        ("train-record", "identity-0", "train", 5.0),
        ("validation-record", "identity-1", "validation", 15.0),
        ("test-record", "identity-2", "test", 30.0),
    ]
    candidates, overlay_index_path, _, paths = _materialize(tmp_path, groups=specs)
    descriptor, calibration, profile, _ = paths
    index = OverlayReviewIndex.load(overlay_index_path)
    review = OverlayReviewDecision(
        materialization_sha256=index.materialization_sha256,
        overlay_sha256=index.overlay_sha256,
        record_reviews=[_record_review(record_id) for record_id in index.sampled_record_ids],
        review=ReviewApproval(
            status="approved",
            reviewed_by="test-reviewer",
            reviewed_at=datetime.fromisoformat("2026-07-27T00:00:00+08:00"),
            evidence_sha256="3" * 64,
        ),
    )
    review_path = tmp_path / "sampled-review.json"
    _write_model(review, review_path)
    plan = StageASplitPlan(
        splits=cast(
            Any,
            {
                split: SplitManifest(
                    identities=[identity],
                    sessions=[f"{split}-session"],
                    devices=[f"{split}-{role}-device" for role in ("front", "left", "right")],
                    camera_ids=[f"{split}-{role}-camera" for role in ("front", "left", "right")],
                )
                for _, identity, split, _ in specs
            },
        )
    )
    plan_path = tmp_path / "split-plan.json"
    _write_model(plan, plan_path)
    manifest_path = tmp_path / "sampled-only-manifest.json"
    corrected_id = index.sampled_record_ids[0]
    corrected = StageARecordReview(
        record_id=corrected_id,
        action="corrected",
        reviewed_by="test-reviewer",
        reviewed_at=datetime.fromisoformat("2026-07-27T00:00:00+08:00"),
        evidence_sha256=hashlib.sha256(f"corrected:{corrected_id}".encode()).hexdigest(),
        reviewed_anchor_names=list(SEMANTIC_ANCHORS),
        correction_sha256="4" * 64,
        correction_magnitude_px=0.25,
    )
    corrected_review = review.model_copy(
        update={
            "record_reviews": [
                corrected if item.record_id == corrected_id else item
                for item in review.record_reviews
            ]
        }
    )
    corrected_review_path = tmp_path / "corrected-review.json"
    _write_model(corrected_review, corrected_review_path)
    with pytest.raises(ValueError, match="must be re-materialized"):
        build_stage_a_manifest(
            candidates,
            overlay_index_path=overlay_index_path,
            overlay_review_path=corrected_review_path,
            split_plan_path=plan_path,
            teacher_descriptor_path=descriptor,
            calibration_path=calibration,
            quality_profile_path=profile,
            license_registry_path=tmp_path / "license-registry.json",
            label_catalog_path=Path("configs/data/ntp-v1-label-catalog.json").resolve(),
            training_recipe_path=Path("configs/training/nana-training-recipe-1.0.0.json").resolve(),
            capture_license_record_id="self-captured-consented",
            data_revision="synthetic-corrected-review",
            output_path=tmp_path / "corrected-manifest.json",
        )
    build_stage_a_manifest(
        candidates,
        overlay_index_path=overlay_index_path,
        overlay_review_path=review_path,
        split_plan_path=plan_path,
        teacher_descriptor_path=descriptor,
        calibration_path=calibration,
        quality_profile_path=profile,
        license_registry_path=tmp_path / "license-registry.json",
        label_catalog_path=Path("configs/data/ntp-v1-label-catalog.json").resolve(),
        training_recipe_path=Path("configs/training/nana-training-recipe-1.0.0.json").resolve(),
        capture_license_record_id="self-captured-consented",
        data_revision="synthetic-sampled-review-only",
        output_path=manifest_path,
    )
    shard_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = manifest_path.parent / shard_payload["record_files"][0]["path"]
    approved_ids = {group.record_id for group in MaterializedStageAGroup.load_jsonl(shard_path)}
    assert approved_ids == set(index.sampled_record_ids)
    with pytest.raises(ValueError, match="every Stage A split requires"):
        validate_stage_a_manifest(manifest_path)
