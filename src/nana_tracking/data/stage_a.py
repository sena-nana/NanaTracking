"""First-party synchronized MediaPipe/OpenCV Stage A label materialization.

The public artifacts in this module are framework-neutral JSON contracts. MediaPipe and OpenCV
objects stay behind the materializer boundary and never enter NTP, FFI, checkpoints, or consumers.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2.functional import crop, resize

from nana_tracking.config import ExperimentConfig
from nana_tracking.contracts import MultiViewTrackingBatch
from nana_tracking.data.face_basic import resolve_image_uri
from nana_tracking.data.manifest import (
    DatasetManifest,
    FileReference,
    LicensePermissions,
    LicenseReview,
    SplitManifest,
    SynchronizationPolicy,
    TeacherSource,
    dataset_digest,
)
from nana_tracking.data.schema import CaptureConditions, RgbFrame
from nana_tracking.data.strategy import LicenseRecord, LicenseRegistry
from nana_tracking.data.teachers import TeacherModelDescriptor

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ViewRole = Literal["front", "left", "right"]
LabelState = Literal["available", "unavailable"]
ReviewStatus = Literal["pending", "approved", "rejected"]
Point2 = tuple[FiniteFloat, FiniteFloat]
Point3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Matrix3 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
]
Matrix4 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
]

MEDIAPIPE_VERSION = "0.10.35"
OPENCV_DISTRIBUTION = "opencv-contrib-python"
OPENCV_VERSION = "5.0.0.93"
MEDIAPIPE_SOURCE_ID = "mediapipe-face-landmarker-v1"
OPENCV_SOURCE_ID = "opencv-calibrated-geometry-v1"
HEAD_FRAME_REVISION = "nana-head-frame/1.0.0"
GEOMETRY_DERIVATION_REVISION = "opencv-multiview-geometry/1.0.0"


class StageAContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewApproval(StageAContract):
    status: ReviewStatus
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    evidence_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_approval(self) -> Self:
        fields = (self.reviewed_by, self.reviewed_at, self.evidence_sha256)
        if self.status == "approved" and any(value is None for value in fields):
            raise ValueError("approved review requires reviewer, timestamp, and evidence digest")
        if self.status != "approved" and any(value is not None for value in fields):
            raise ValueError("non-approved review cannot carry approval evidence")
        return self


class StageAInputView(StageAContract):
    role: ViewRole
    camera_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    capture_timestamp_ns: int = Field(ge=0)
    rgb: RgbFrame


class StageACaptureGroup(StageAContract):
    schema_version: Literal["nana-first-party-multiview-capture/1.0.0"] = (
        "nana-first-party-multiview-capture/1.0.0"
    )
    record_id: str = Field(min_length=1)
    identity_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    take_id: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    action_script_id: str = Field(min_length=1)
    consent_record_id: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    capture_timestamp_ns: int = Field(ge=0)
    sequence: int = Field(ge=0)
    source_fps: FiniteFloat = Field(ge=15.0, le=30.0)
    effective_fps: FiniteFloat = Field(ge=4.5, le=30.0)
    conditions: CaptureConditions
    views: list[StageAInputView] = Field(min_length=1, max_length=3)

    @classmethod
    def load_jsonl(cls, path: Path) -> list[Self]:
        values = [
            cls.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not values:
            raise ValueError(f"Stage A capture index is empty: {path}")
        return values


class RigCameraCalibration(StageAContract):
    camera_id: str = Field(min_length=1)
    intrinsics: Matrix3
    distortion_model: Literal["none", "brown-conrady", "fisheye"]
    distortion_coefficients: list[FiniteFloat]
    camera_to_capture: Matrix4

    @model_validator(mode="after")
    def validate_transform(self) -> Self:
        transform = np.asarray(self.camera_to_capture, dtype=np.float64)
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError("camera_to_capture must be an affine homogeneous transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("camera_to_capture rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("camera_to_capture rotation must be right-handed")
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if (
            intrinsics[0, 0] <= 0.0
            or intrinsics[1, 1] <= 0.0
            or not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=1e-8)
        ):
            raise ValueError("camera intrinsics are invalid")
        return self


class SubjectHeadScale(StageAContract):
    identity_id: str = Field(min_length=1)
    head_width_m: FiniteFloat = Field(gt=0.05, lt=0.40)


class CameraRigCalibrationBundle(StageAContract):
    schema_version: Literal["nana-camera-rig-calibration/1.0.0"] = (
        "nana-camera-rig-calibration/1.0.0"
    )
    calibration_revision: str = Field(min_length=1)
    coordinate_convention: Literal["opencv-camera-x-right-y-down-z-forward"] = (
        "opencv-camera-x-right-y-down-z-forward"
    )
    cameras: list[RigCameraCalibration] = Field(min_length=3)
    subject_scales: list[SubjectHeadScale] = Field(min_length=1)
    review: ReviewApproval

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        camera_ids = [camera.camera_id for camera in self.cameras]
        identity_ids = [scale.identity_id for scale in self.subject_scales]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("calibration camera IDs must be unique")
        if len(identity_ids) != len(set(identity_ids)):
            raise ValueError("calibration subject identities must be unique")
        return self

    @classmethod
    def load(cls, path: Path, *, require_approved: bool = True) -> Self:
        value = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if require_approved and value.review.status != "approved":
            raise ValueError("Stage A requires reviewed camera and subject calibration")
        return value


class StageAQualityProfile(StageAContract):
    schema_version: Literal["nana-stage-a-quality-profile/1.0.0"] = (
        "nana-stage-a-quality-profile/1.0.0"
    )
    profile_revision: str = Field(min_length=1)
    opencv_distribution: Literal["opencv-contrib-python"] = OPENCV_DISTRIBUTION
    opencv_version: Literal["5.0.0.93"] = OPENCV_VERSION
    geometry_derivation_revision: Literal["opencv-multiview-geometry/1.0.0"] = (
        GEOMETRY_DERIVATION_REVISION
    )
    head_frame_revision: Literal["nana-head-frame/1.0.0"] = HEAD_FRAME_REVISION
    max_view_skew_ns: int = Field(gt=0, le=5_000_000)
    pseudo_label_confidence: FiniteFloat = Field(gt=0.0, le=0.5)
    min_face_detection_confidence: FiniteFloat = Field(gt=0.0, le=1.0)
    min_face_presence_confidence: FiniteFloat = Field(gt=0.0, le=1.0)
    min_triangulation_angle_degrees: FiniteFloat = Field(gt=0.0, lt=90.0)
    min_depth_m: FiniteFloat = Field(gt=0.0)
    max_depth_m: FiniteFloat = Field(gt=0.0)
    max_reprojection_rms_px: FiniteFloat = Field(gt=0.0)
    max_reprojection_p95_px: FiniteFloat = Field(gt=0.0)
    max_reprojection_px: FiniteFloat = Field(gt=0.0)
    pnp_reprojection_error_px: FiniteFloat = Field(gt=0.0)
    pnp_iterations: int = Field(ge=1, le=10_000)
    min_pnp_inliers: int = Field(ge=4, le=16)
    max_capture_pose_translation_head_widths: FiniteFloat = Field(gt=0.0)
    max_capture_pose_rotation_degrees: FiniteFloat = Field(gt=0.0, le=180.0)
    overlay_review_sample_count: int = Field(ge=1)
    review_seed: Literal[17] = 17
    review: ReviewApproval

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("maximum triangulation depth must exceed minimum depth")
        if not (
            self.max_reprojection_rms_px <= self.max_reprojection_p95_px <= self.max_reprojection_px
        ):
            raise ValueError("reprojection RMS, P95, and max thresholds must be increasing")
        return self

    @classmethod
    def load(cls, path: Path, *, require_approved: bool = True) -> Self:
        value = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if require_approved and value.review.status != "approved":
            raise ValueError("Stage A requires a reviewed quality profile")
        return value


class Landmark2DLabel(StageAContract):
    semantic_name: str
    teacher_index: int = Field(ge=0)
    state: LabelState
    point_pixels: Point2 | None
    confidence: FiniteFloat = Field(ge=0.0, le=0.5)
    evidence: Literal["pseudo_label"] = "pseudo_label"
    source_id: Literal["mediapipe-face-landmarker-v1"] = MEDIAPIPE_SOURCE_ID
    model_sha256: Sha256
    mapping_revision: str = Field(min_length=1)
    capture_timestamp_ns: int = Field(ge=0)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.state == "available":
            if self.point_pixels is None or self.confidence <= 0.0 or self.unavailable_reason:
                raise ValueError("available 2D pseudo-label requires a point and confidence")
        elif self.point_pixels is not None or self.confidence != 0.0 or not self.unavailable_reason:
            raise ValueError("unavailable 2D pseudo-label requires no point and zero confidence")
        return self


class GeometryLabel(StageAContract):
    state: LabelState
    points_head_relative: list[Point3] | None
    confidence: FiniteFloat = Field(ge=0.0, le=0.5)
    evidence: Literal["geometry"] = "geometry"
    source_id: Literal["opencv-calibrated-geometry-v1"] = OPENCV_SOURCE_ID
    calibration_revision: str
    derivation_revision: str
    head_frame_revision: str
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.state == "available":
            if (
                self.points_head_relative is None
                or len(self.points_head_relative) != 16
                or self.confidence <= 0.0
                or self.unavailable_reason
            ):
                raise ValueError("available geometry requires 16 points and confidence")
        elif (
            self.points_head_relative is not None
            or self.confidence != 0.0
            or not self.unavailable_reason
        ):
            raise ValueError("unavailable geometry requires no points and zero confidence")
        return self


class PoseLabel(StageAContract):
    state: LabelState
    translation_head_relative: Point3 | None
    orientation_xyzw: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None
    confidence: FiniteFloat = Field(ge=0.0, le=0.5)
    evidence: Literal["geometry"] = "geometry"
    source_id: Literal["opencv-calibrated-geometry-v1"] = OPENCV_SOURCE_ID
    calibration_revision: str
    derivation_revision: str
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.state == "available":
            if (
                self.translation_head_relative is None
                or self.orientation_xyzw is None
                or self.confidence <= 0.0
                or self.unavailable_reason
            ):
                raise ValueError("available pose requires translation, orientation, and confidence")
            norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
            if not math.isclose(norm, 1.0, abs_tol=1e-5):
                raise ValueError("pose quaternion must be normalized")
        elif (
            self.translation_head_relative is not None
            or self.orientation_xyzw is not None
            or self.confidence != 0.0
            or not self.unavailable_reason
        ):
            raise ValueError("unavailable pose requires no value and zero confidence")
        return self


class ViewQuality(StageAContract):
    passed: bool
    residuals_px: list[FiniteFloat] | None = None
    rms_px: FiniteFloat | None = None
    p95_px: FiniteFloat | None = None
    max_px: FiniteFloat | None = None
    pnp_inlier_count: int = Field(default=0, ge=0, le=16)
    failure_codes: list[str] = Field(default_factory=list)


class MaterializedStageAView(StageAContract):
    role: ViewRole
    camera_id: str
    device_id: str
    capture_timestamp_ns: int
    image_uri: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    face_bbox_xyxy: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None
    intrinsics: Matrix3
    camera_to_capture: Matrix4
    landmarks_2d: list[Landmark2DLabel] = Field(min_length=16, max_length=16)
    reprojected_points_pixels: list[Point2] | None = None
    pose: PoseLabel
    visibility: int = Field(ge=0, le=2)
    quality: ViewQuality


class MaterializedStageAGroup(StageAContract):
    schema_version: Literal["nana-stage-a-materialization/1.0.0"] = (
        "nana-stage-a-materialization/1.0.0"
    )
    record_id: str
    identity_id: str
    session_id: str
    take_id: str
    environment_id: str
    action_script_id: str
    consent_record_id: str
    expression: str
    capture_timestamp_ns: int
    sequence: int
    source_fps: FiniteFloat
    effective_fps: FiniteFloat
    conditions: CaptureConditions
    status: Literal["accepted", "rejected"]
    review_status: ReviewStatus = "pending"
    failure_codes: list[str]
    views: list[MaterializedStageAView] = Field(min_length=3, max_length=3)
    canonical_geometry: GeometryLabel
    teacher_descriptor_sha256: Sha256
    calibration_sha256: Sha256
    quality_profile_sha256: Sha256

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        available = self.canonical_geometry.state == "available" and all(
            view.pose.state == "available" and view.quality.passed for view in self.views
        )
        if self.status == "accepted" and (not available or self.failure_codes):
            raise ValueError("accepted Stage A group requires complete passing geometry and pose")
        if self.status == "rejected" and (available or not self.failure_codes):
            raise ValueError("rejected Stage A group requires unavailable output and failure codes")
        if self.review_status == "approved" and self.status != "accepted":
            raise ValueError("rejected Stage A groups cannot be approved for training")
        return self

    @classmethod
    def load_jsonl(cls, path: Path) -> list[Self]:
        values = [
            cls.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not values:
            raise ValueError(f"Stage A materialization shard is empty: {path}")
        return values


class OverlayArtifact(StageAContract):
    record_id: str
    camera_id: str
    path: str
    sha256: Sha256


class OverlayReviewIndex(StageAContract):
    schema_version: Literal["nana-stage-a-overlay-index/1.0.0"] = "nana-stage-a-overlay-index/1.0.0"
    materialization_sha256: Sha256
    overlay_sha256: Sha256
    sampled_record_ids: list[str] = Field(min_length=1)
    artifacts: list[OverlayArtifact] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def verify_files(self, path: Path) -> None:
        aggregate = hashlib.sha256()
        for artifact in sorted(self.artifacts, key=lambda item: item.path):
            artifact_path = (path.parent / artifact.path).resolve()
            payload = artifact_path.read_bytes()
            if _sha256_bytes(payload) != artifact.sha256:
                raise ValueError(f"overlay artifact digest mismatch: {artifact.path}")
            aggregate.update(artifact.path.encode())
            aggregate.update(payload)
        if aggregate.hexdigest() != self.overlay_sha256:
            raise ValueError("overlay aggregate digest mismatch")


class OverlayReviewDecision(StageAContract):
    schema_version: Literal["nana-stage-a-overlay-review/1.0.0"] = (
        "nana-stage-a-overlay-review/1.0.0"
    )
    materialization_sha256: Sha256
    overlay_sha256: Sha256
    reviewed_record_ids: list[str] = Field(min_length=1)
    review: ReviewApproval

    @classmethod
    def load(cls, path: Path, *, require_approved: bool = True) -> Self:
        value = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if require_approved and value.review.status != "approved":
            raise ValueError("Stage A overlay review is not approved")
        return value


class StageASplitPlan(StageAContract):
    schema_version: Literal["nana-stage-a-splits/1.0.0"] = "nana-stage-a-splits/1.0.0"
    seed: Literal[17] = 17
    splits: dict[Literal["train", "validation", "test"], SplitManifest]

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        if set(self.splits) != {"train", "validation", "test"}:
            raise ValueError("Stage A split plan requires train, validation, and test")
        counts = {name: len(split.identities) for name, split in self.splits.items()}
        if counts != {"train": 5, "validation": 1, "test": 2}:
            raise ValueError("commercial Stage A requires the reviewed 5/1/2 identity split")
        for name, split in self.splits.items():
            if len(split.camera_ids) != 3:
                raise ValueError(f"Stage A {name} split requires exactly three camera IDs")
            if len(split.devices) != 3:
                raise ValueError(f"Stage A {name} split requires exactly three device IDs")
        _reject_split_overlap(self.splits)
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class StageAMaterializationSummary(StageAContract):
    schema_version: Literal["nana-stage-a-quality-summary/1.0.0"] = (
        "nana-stage-a-quality-summary/1.0.0"
    )
    smoke_only: Literal[False] = False
    record_count: int
    accepted_count: int
    rejected_count: int
    failure_counts: dict[str, int]
    materialization_sha256: Sha256
    overlay_sha256: Sha256
    candidates: str
    overlay_index: str


class LandmarkDetector(Protocol):
    def detect(self, rgb: NDArray[np.uint8]) -> list[tuple[float, float]]: ...

    def close(self) -> None: ...


class MediaPipeLandmarkDetector:
    """Pinned IMAGE-mode detector exposing normalized 16-point coordinates only."""

    def __init__(
        self,
        model_asset: Path,
        descriptor: TeacherModelDescriptor,
        profile: StageAQualityProfile,
    ) -> None:
        if importlib.metadata.version("mediapipe") != descriptor.framework_version:
            raise ValueError("installed MediaPipe version does not match the teacher descriptor")
        mp: Any = importlib.import_module("mediapipe")
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_asset))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=profile.min_face_detection_confidence,
            min_face_presence_confidence=profile.min_face_presence_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mp = mp
        self._bindings = tuple(descriptor.anchor_bindings)
        self._landmarker: Any = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb: NDArray[np.uint8]) -> list[tuple[float, float]]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result: Any = self._landmarker.detect(image)
        if len(result.face_landmarks) != 1:
            raise MaterializationFailure("mediapipe_face_count")
        landmarks: Any = result.face_landmarks[0]
        return [
            (float(landmarks[binding.teacher_index].x), float(landmarks[binding.teacher_index].y))
            for binding in self._bindings
        ]

    def close(self) -> None:
        self._landmarker.close()


class MaterializationFailure(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class _ViewWork:
    source: StageAInputView
    calibration: RigCameraCalibration
    points_pixels: NDArray[np.float64]
    points_undistorted: NDArray[np.float64]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_cv2(profile: StageAQualityProfile) -> Any:
    if importlib.metadata.version(profile.opencv_distribution) != profile.opencv_version:
        raise ValueError("installed OpenCV distribution does not match the quality profile")
    conflicting_distributions = (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    for conflicting in conflicting_distributions:
        try:
            importlib.metadata.version(conflicting)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise ValueError(f"multiple cv2 providers are installed: {conflicting}")
    cv2: Any = importlib.import_module("cv2")
    if not str(cv2.__version__).startswith("5.0.0"):
        raise ValueError("imported cv2 runtime does not match the reviewed OpenCV version")
    return cv2


def _read_bgr(path: Path, cv2: Any) -> NDArray[np.uint8]:
    payload = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cast(NDArray[np.uint8] | None, cv2.imdecode(payload, cv2.IMREAD_COLOR))
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise MaterializationFailure("rgb_decode_failed")
    return image


def _distortion(camera: RigCameraCalibration) -> NDArray[np.float64]:
    return np.asarray(camera.distortion_coefficients, dtype=np.float64)


def _projection(camera: RigCameraCalibration) -> NDArray[np.float64]:
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
    capture_to_camera = np.linalg.inv(np.asarray(camera.camera_to_capture, dtype=np.float64))
    return intrinsics @ capture_to_camera[:3]


def _project_capture_points(
    points: NDArray[np.float64],
    camera: RigCameraCalibration,
    cv2: Any,
) -> NDArray[np.float64]:
    capture_to_camera = np.linalg.inv(np.asarray(camera.camera_to_capture, dtype=np.float64))
    rotation = capture_to_camera[:3, :3]
    translation = capture_to_camera[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(
        points,
        rvec,
        translation,
        np.asarray(camera.intrinsics, dtype=np.float64),
        _distortion(camera),
    )
    return projected.reshape(-1, 2)


def _depths(points: NDArray[np.float64], camera: RigCameraCalibration) -> NDArray[np.float64]:
    capture_to_camera = np.linalg.inv(np.asarray(camera.camera_to_capture, dtype=np.float64))
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return (capture_to_camera @ homogeneous.T).T[:, 2]


def _triangulation_angle_degrees(
    point: NDArray[np.float64],
    left: RigCameraCalibration,
    right: RigCameraCalibration,
) -> float:
    left_center = np.asarray(left.camera_to_capture, dtype=np.float64)[:3, 3]
    right_center = np.asarray(right.camera_to_capture, dtype=np.float64)[:3, 3]
    left_ray = point - left_center
    right_ray = point - right_center
    cosine = float(
        np.dot(left_ray, right_ray)
        / max(np.linalg.norm(left_ray) * np.linalg.norm(right_ray), 1e-12)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _triangulate(
    views: list[_ViewWork],
    profile: StageAQualityProfile,
    cv2: Any,
) -> NDArray[np.float64]:
    selected: list[NDArray[np.float64]] = []
    pairs = [(left, right) for index, left in enumerate(views) for right in views[index + 1 :]]
    for point_index in range(16):
        candidates: list[tuple[float, str, NDArray[np.float64]]] = []
        depth_rejected = False
        angle_rejected = False
        for left, right in pairs:
            homogeneous = cast(
                NDArray[np.float64],
                cv2.triangulatePoints(
                    _projection(left.calibration),
                    _projection(right.calibration),
                    left.points_undistorted[point_index].reshape(2, 1),
                    right.points_undistorted[point_index].reshape(2, 1),
                ),
            )
            divisor = float(homogeneous[3, 0])
            if abs(divisor) < 1e-12:
                continue
            point = np.asarray(homogeneous[:3, 0] / divisor, dtype=np.float64)
            if not np.isfinite(point).all():
                continue
            depths = np.concatenate(
                [_depths(point.reshape(1, 3), view.calibration) for view in views]
            )
            if np.any(depths < profile.min_depth_m) or np.any(depths > profile.max_depth_m):
                depth_rejected = True
                continue
            angle = _triangulation_angle_degrees(point, left.calibration, right.calibration)
            if angle < profile.min_triangulation_angle_degrees:
                angle_rejected = True
                continue
            residuals: list[float] = []
            for view in views:
                projected = _project_capture_points(point.reshape(1, 3), view.calibration, cv2)[0]
                residual = cast(
                    np.float64,
                    np.linalg.norm(projected - view.points_pixels[point_index]),
                )
                residuals.append(float(residual))
            key = f"{left.source.camera_id}:{right.source.camera_id}"
            residual_rms = math.sqrt(sum(value * value for value in residuals) / 3)
            candidates.append((residual_rms, key, point))
        if not candidates:
            if depth_rejected:
                raise MaterializationFailure("triangulation_depth_invalid")
            if angle_rejected:
                raise MaterializationFailure("triangulation_angle_low")
            raise MaterializationFailure("triangulation_invalid")
        selected.append(min(candidates, key=lambda item: (item[0], item[1]))[2])
    return np.stack(selected)


def _head_relative_geometry(
    points_capture: NDArray[np.float64],
    head_width_m: float,
) -> NDArray[np.float64]:
    left_eye = (points_capture[0] + points_capture[1]) * 0.5
    right_eye = (points_capture[2] + points_capture[3]) * 0.5
    origin = (left_eye + right_eye) * 0.5
    x_axis = left_eye - right_eye
    x_norm = float(cast(np.float64, np.linalg.norm(x_axis)))
    x_axis /= max(x_norm, 1e-12)
    y_axis = points_capture[15] - origin
    y_axis -= x_axis * np.dot(y_axis, x_axis)
    y_norm = float(cast(np.float64, np.linalg.norm(y_axis)))
    y_axis /= max(y_norm, 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(np.linalg.norm(z_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    basis = np.stack((x_axis, y_axis, z_axis), axis=1)
    geometry = (points_capture - origin) @ basis / head_width_m
    if not np.isfinite(geometry).all():
        raise MaterializationFailure("head_frame_invalid")
    return cast(NDArray[np.float64], geometry)


def _quaternion_xyzw(rotation: NDArray[np.float64], cv2: Any) -> tuple[float, float, float, float]:
    rotation_vector = np.asarray(cv2.Rodrigues(rotation)[0], dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    xyz = rotation_vector * (math.sin(angle * 0.5) / angle)
    quaternion = (*map(float, xyz), math.cos(angle * 0.5))
    norm = math.sqrt(sum(value * value for value in quaternion))
    return cast(tuple[float, float, float, float], tuple(value / norm for value in quaternion))


def _rotation_delta_degrees(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    cosine = (float(np.trace(left.T @ right)) - 1.0) * 0.5
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _solve_poses(
    geometry: NDArray[np.float64],
    views: list[_ViewWork],
    head_width_m: float,
    profile: StageAQualityProfile,
    cv2: Any,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]]:
    solved: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]] = {}
    capture_poses: dict[str, NDArray[np.float64]] = {}
    for view in views:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            geometry,
            view.points_pixels,
            np.asarray(view.calibration.intrinsics, dtype=np.float64),
            _distortion(view.calibration),
            iterationsCount=profile.pnp_iterations,
            reprojectionError=profile.pnp_reprojection_error_px,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inlier_values = (
            np.empty((0,), dtype=np.int32)
            if inliers is None
            else np.asarray(inliers, dtype=np.int32).reshape(-1)
        )
        if not success or len(inlier_values) < profile.min_pnp_inliers:
            raise MaterializationFailure("pnp_failed")
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        head_to_camera = np.eye(4, dtype=np.float64)
        head_to_camera[:3, :3] = rotation
        head_to_camera[:3, 3] = translation * head_width_m
        capture_pose = (
            np.asarray(view.calibration.camera_to_capture, dtype=np.float64) @ head_to_camera
        )
        solved[view.source.camera_id] = (rotation, translation, inlier_values)
        capture_poses[view.source.camera_id] = capture_pose
    values = list(capture_poses.values())
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            translation_delta = np.linalg.norm(left[:3, 3] - right[:3, 3]) / head_width_m
            rotation_delta = _rotation_delta_degrees(left[:3, :3], right[:3, :3])
            if translation_delta > profile.max_capture_pose_translation_head_widths:
                raise MaterializationFailure("pose_translation_disagreement")
            if rotation_delta > profile.max_capture_pose_rotation_degrees:
                raise MaterializationFailure("pose_rotation_disagreement")
    return solved


def _unavailable_landmarks(
    descriptor: TeacherModelDescriptor,
    timestamp_ns: int,
    reason: str,
) -> list[Landmark2DLabel]:
    return [
        Landmark2DLabel(
            semantic_name=binding.semantic_name,
            teacher_index=binding.teacher_index,
            state="unavailable",
            point_pixels=None,
            confidence=0.0,
            model_sha256=descriptor.model.sha256,
            mapping_revision=descriptor.output_contract_revision,
            capture_timestamp_ns=timestamp_ns,
            unavailable_reason=reason,
        )
        for binding in descriptor.anchor_bindings
    ]


def _unavailable_geometry(
    calibration: CameraRigCalibrationBundle,
    profile: StageAQualityProfile,
    reason: str,
) -> GeometryLabel:
    return GeometryLabel(
        state="unavailable",
        points_head_relative=None,
        confidence=0.0,
        calibration_revision=calibration.calibration_revision,
        derivation_revision=profile.geometry_derivation_revision,
        head_frame_revision=profile.head_frame_revision,
        unavailable_reason=reason,
    )


def _unavailable_pose(
    calibration: CameraRigCalibrationBundle,
    profile: StageAQualityProfile,
    reason: str,
) -> PoseLabel:
    return PoseLabel(
        state="unavailable",
        translation_head_relative=None,
        orientation_xyzw=None,
        confidence=0.0,
        calibration_revision=calibration.calibration_revision,
        derivation_revision=profile.geometry_derivation_revision,
        unavailable_reason=reason,
    )


def _visibility(conditions: CaptureConditions) -> int:
    if "out_of_frame" in conditions.occlusions:
        return 2
    return 1 if conditions.occlusions else 0


def _bbox(
    points: NDArray[np.float64], width: int, height: int
) -> tuple[float, float, float, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    side = max(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1])) * 1.5
    side = max(side, 8.0)
    left = max(0.0, float(center[0] - side * 0.5))
    top = max(0.0, float(center[1] - side * 0.5))
    right = min(float(width), float(center[0] + side * 0.5))
    bottom = min(float(height), float(center[1] + side * 0.5))
    if right <= left or bottom <= top:
        raise MaterializationFailure("face_bbox_invalid")
    return left, top, right, bottom


def _base_rejected_group(
    group: StageACaptureGroup,
    views: list[MaterializedStageAView],
    reason: str,
    descriptor_digest: str,
    calibration_digest: str,
    profile_digest: str,
    calibration: CameraRigCalibrationBundle,
    profile: StageAQualityProfile,
) -> MaterializedStageAGroup:
    rejected_views = [
        view.model_copy(
            update={
                "pose": _unavailable_pose(calibration, profile, reason),
                "quality": view.quality.model_copy(
                    update={
                        "passed": False,
                        "failure_codes": sorted(set([*view.quality.failure_codes, reason])),
                    }
                ),
            }
        )
        for view in views
    ]
    return MaterializedStageAGroup(
        record_id=group.record_id,
        identity_id=group.identity_id,
        session_id=group.session_id,
        take_id=group.take_id,
        environment_id=group.environment_id,
        action_script_id=group.action_script_id,
        consent_record_id=group.consent_record_id,
        expression=group.expression,
        capture_timestamp_ns=group.capture_timestamp_ns,
        sequence=group.sequence,
        source_fps=group.source_fps,
        effective_fps=group.effective_fps,
        conditions=group.conditions,
        status="rejected",
        failure_codes=[reason],
        views=rejected_views,
        canonical_geometry=_unavailable_geometry(calibration, profile, reason),
        teacher_descriptor_sha256=descriptor_digest,
        calibration_sha256=calibration_digest,
        quality_profile_sha256=profile_digest,
    )


def _materialize_group(
    group: StageACaptureGroup,
    *,
    index_path: Path,
    detector: LandmarkDetector,
    descriptor: TeacherModelDescriptor,
    calibration: CameraRigCalibrationBundle,
    profile: StageAQualityProfile,
    descriptor_digest: str,
    calibration_digest: str,
    profile_digest: str,
    cv2: Any,
) -> MaterializedStageAGroup:
    cameras = {camera.camera_id: camera for camera in calibration.cameras}
    scales = {scale.identity_id: scale.head_width_m for scale in calibration.subject_scales}
    ordered_sources = sorted(
        group.views, key=lambda view: ("front", "left", "right").index(view.role)
    )
    initial_views: list[MaterializedStageAView] = []
    work: list[_ViewWork] = []
    try:
        timestamps = [view.capture_timestamp_ns for view in ordered_sources]
        roles = [view.role for view in ordered_sources]
        camera_ids = [view.camera_id for view in ordered_sources]
        device_ids = [view.device_id for view in ordered_sources]
        if len(roles) != len(set(roles)):
            raise MaterializationFailure("duplicate_view_role")
        if set(roles) != {"front", "left", "right"}:
            raise MaterializationFailure("missing_view")
        if len(camera_ids) != len(set(camera_ids)):
            raise MaterializationFailure("duplicate_camera")
        if len(device_ids) != len(set(device_ids)):
            raise MaterializationFailure("duplicate_device")
        if max(timestamps) - min(timestamps) > profile.max_view_skew_ns:
            raise MaterializationFailure("view_timestamp_skew")
        if group.identity_id not in scales:
            raise MaterializationFailure("subject_scale_missing")
        for source in ordered_sources:
            camera = cameras.get(source.camera_id)
            if camera is None:
                raise MaterializationFailure("camera_calibration_missing")
            image_path = resolve_image_uri(index_path, source.rgb.uri)
            bgr = _read_bgr(image_path, cv2)
            if bgr.shape[:2] != (source.rgb.height, source.rgb.width):
                raise MaterializationFailure("rgb_dimensions_mismatch")
            try:
                detected = detector.detect(
                    cast(NDArray[np.uint8], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                )
            except MaterializationFailure:
                raise
            except Exception as error:
                raise MaterializationFailure("mediapipe_detection_failed") from error
            normalized = np.asarray(detected, dtype=np.float64)
            if normalized.shape != (16, 2) or not np.isfinite(normalized).all():
                raise MaterializationFailure("mediapipe_landmarks_invalid")
            if np.any(normalized < 0.0) or np.any(normalized > 1.0):
                raise MaterializationFailure("mediapipe_landmarks_out_of_frame")
            pixels = normalized * np.asarray((source.rgb.width, source.rgb.height))
            intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
            undistorted = cv2.undistortPoints(
                pixels.reshape(-1, 1, 2),
                intrinsics,
                _distortion(camera),
                P=intrinsics,
            ).reshape(-1, 2)
            labels = [
                Landmark2DLabel(
                    semantic_name=binding.semantic_name,
                    teacher_index=binding.teacher_index,
                    state="available",
                    point_pixels=cast(Point2, tuple(float(value) for value in pixels[index])),
                    confidence=profile.pseudo_label_confidence,
                    model_sha256=descriptor.model.sha256,
                    mapping_revision=descriptor.output_contract_revision,
                    capture_timestamp_ns=source.capture_timestamp_ns,
                )
                for index, binding in enumerate(descriptor.anchor_bindings)
            ]
            initial_views.append(
                MaterializedStageAView(
                    role=source.role,
                    camera_id=source.camera_id,
                    device_id=source.device_id,
                    capture_timestamp_ns=source.capture_timestamp_ns,
                    image_uri=str(image_path),
                    width=source.rgb.width,
                    height=source.rgb.height,
                    face_bbox_xyxy=_bbox(pixels, source.rgb.width, source.rgb.height),
                    intrinsics=camera.intrinsics,
                    camera_to_capture=camera.camera_to_capture,
                    landmarks_2d=labels,
                    pose=_unavailable_pose(calibration, profile, "geometry_pending"),
                    visibility=_visibility(group.conditions),
                    quality=ViewQuality(passed=False, failure_codes=["geometry_pending"]),
                )
            )
            work.append(_ViewWork(source, camera, pixels, undistorted))
        geometry_capture = _triangulate(work, profile, cv2)
        geometry = _head_relative_geometry(geometry_capture, scales[group.identity_id])
        solved = _solve_poses(
            geometry,
            work,
            scales[group.identity_id],
            profile,
            cv2,
        )
        accepted_views: list[MaterializedStageAView] = []
        for view, candidate in zip(work, initial_views, strict=True):
            rotation, translation, inliers = solved[view.source.camera_id]
            projected, _ = cv2.projectPoints(
                geometry,
                cv2.Rodrigues(rotation)[0],
                translation,
                np.asarray(view.calibration.intrinsics, dtype=np.float64),
                _distortion(view.calibration),
            )
            reprojected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
            residuals = np.linalg.norm(reprojected - view.points_pixels, axis=1)
            rms = float(math.sqrt(float(np.mean(residuals * residuals))))
            p95 = float(np.quantile(residuals, 0.95))
            maximum = float(np.max(residuals))
            if (
                rms > profile.max_reprojection_rms_px
                or p95 > profile.max_reprojection_p95_px
                or maximum > profile.max_reprojection_px
            ):
                raise MaterializationFailure("reprojection_threshold")
            accepted_views.append(
                candidate.model_copy(
                    update={
                        "reprojected_points_pixels": [
                            cast(Point2, tuple(float(value) for value in point))
                            for point in reprojected
                        ],
                        "pose": PoseLabel(
                            state="available",
                            translation_head_relative=cast(
                                Point3, tuple(float(value) for value in translation)
                            ),
                            orientation_xyzw=_quaternion_xyzw(rotation, cv2),
                            confidence=profile.pseudo_label_confidence,
                            calibration_revision=calibration.calibration_revision,
                            derivation_revision=profile.geometry_derivation_revision,
                        ),
                        "quality": ViewQuality(
                            passed=True,
                            residuals_px=[float(value) for value in residuals],
                            rms_px=rms,
                            p95_px=p95,
                            max_px=maximum,
                            pnp_inlier_count=len(inliers),
                        ),
                    }
                )
            )
        return MaterializedStageAGroup(
            record_id=group.record_id,
            identity_id=group.identity_id,
            session_id=group.session_id,
            take_id=group.take_id,
            environment_id=group.environment_id,
            action_script_id=group.action_script_id,
            consent_record_id=group.consent_record_id,
            expression=group.expression,
            capture_timestamp_ns=group.capture_timestamp_ns,
            sequence=group.sequence,
            source_fps=group.source_fps,
            effective_fps=group.effective_fps,
            conditions=group.conditions,
            status="accepted",
            failure_codes=[],
            views=accepted_views,
            canonical_geometry=GeometryLabel(
                state="available",
                points_head_relative=[
                    cast(Point3, tuple(float(value) for value in point)) for point in geometry
                ],
                confidence=profile.pseudo_label_confidence,
                calibration_revision=calibration.calibration_revision,
                derivation_revision=profile.geometry_derivation_revision,
                head_frame_revision=profile.head_frame_revision,
            ),
            teacher_descriptor_sha256=descriptor_digest,
            calibration_sha256=calibration_digest,
            quality_profile_sha256=profile_digest,
        )
    except MaterializationFailure as error:
        materialized_camera_ids = {view.camera_id for view in initial_views}
        for source in ordered_sources:
            if source.camera_id not in materialized_camera_ids:
                camera = cameras.get(source.camera_id)
                if camera is None:
                    fallback_intrinsics: Matrix3 = (
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    )
                    fallback_transform: Matrix4 = (
                        (1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    )
                else:
                    fallback_intrinsics = camera.intrinsics
                    fallback_transform = camera.camera_to_capture
                initial_views.append(
                    MaterializedStageAView(
                        role=source.role,
                        camera_id=source.camera_id,
                        device_id=source.device_id,
                        capture_timestamp_ns=source.capture_timestamp_ns,
                        image_uri=str(resolve_image_uri(index_path, source.rgb.uri)),
                        width=source.rgb.width,
                        height=source.rgb.height,
                        face_bbox_xyxy=None,
                        intrinsics=fallback_intrinsics,
                        camera_to_capture=fallback_transform,
                        landmarks_2d=_unavailable_landmarks(
                            descriptor, source.capture_timestamp_ns, error.code
                        ),
                        pose=_unavailable_pose(calibration, profile, error.code),
                        visibility=_visibility(group.conditions),
                        quality=ViewQuality(passed=False, failure_codes=[error.code]),
                    )
                )
        materialized_roles = {view.role for view in initial_views}
        for missing_role in ("front", "left", "right"):
            if missing_role not in materialized_roles:
                initial_views.append(
                    MaterializedStageAView(
                        role=missing_role,
                        camera_id=f"missing:{missing_role}",
                        device_id=f"missing:{missing_role}",
                        capture_timestamp_ns=group.capture_timestamp_ns,
                        image_uri="",
                        width=64,
                        height=64,
                        face_bbox_xyxy=None,
                        intrinsics=(
                            (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0),
                            (0.0, 0.0, 1.0),
                        ),
                        camera_to_capture=(
                            (1.0, 0.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0, 0.0),
                            (0.0, 0.0, 1.0, 0.0),
                            (0.0, 0.0, 0.0, 1.0),
                        ),
                        landmarks_2d=_unavailable_landmarks(
                            descriptor, group.capture_timestamp_ns, error.code
                        ),
                        pose=_unavailable_pose(calibration, profile, error.code),
                        visibility=_visibility(group.conditions),
                        quality=ViewQuality(passed=False, failure_codes=[error.code]),
                    )
                )
        initial_views.sort(key=lambda view: ("front", "left", "right").index(view.role))
        return _base_rejected_group(
            group,
            initial_views,
            error.code,
            descriptor_digest,
            calibration_digest,
            profile_digest,
            calibration,
            profile,
        )


def _write_json(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_groups(groups: Iterable[MaterializedStageAGroup], path: Path) -> str:
    lines = [
        json.dumps(group.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
        for group in groups
    ]
    if not lines:
        raise ValueError("Stage A materialization output cannot be empty")
    payload = ("\n".join(lines) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def _overlay_selection(
    groups: list[MaterializedStageAGroup], profile: StageAQualityProfile
) -> list[MaterializedStageAGroup]:
    rejected = [group for group in groups if group.status == "rejected"]
    accepted = [group for group in groups if group.status == "accepted"]
    ranked = sorted(
        accepted,
        key=lambda group: hashlib.sha256(
            f"{profile.review_seed}:{group.record_id}".encode()
        ).hexdigest(),
    )
    selected = {group.record_id: group for group in rejected}
    selected.update(
        {group.record_id: group for group in ranked[: profile.overlay_review_sample_count]}
    )
    return [selected[key] for key in sorted(selected)]


def _render_overlays(
    groups: list[MaterializedStageAGroup],
    output_directory: Path,
    materialization_sha256: str,
    profile: StageAQualityProfile,
    cv2: Any,
) -> OverlayReviewIndex:
    selected = _overlay_selection(groups, profile)
    if not selected:
        raise ValueError("overlay review requires at least one selected group")
    overlay_directory = output_directory / "overlays"
    overlay_directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[OverlayArtifact] = []
    aggregate = hashlib.sha256()
    for group in selected:
        for view in group.views:
            image_path = Path(view.image_uri)
            image = (
                _read_bgr(image_path, cv2).copy()
                if view.image_uri and image_path.is_file()
                else np.zeros((view.height, view.width, 3), dtype=np.uint8)
            )
            for label in view.landmarks_2d:
                if label.point_pixels is not None:
                    cv2.circle(
                        image,
                        tuple(round(value) for value in label.point_pixels),
                        3,
                        (0, 0, 255),
                        -1,
                        lineType=cv2.LINE_8,
                    )
            if view.reprojected_points_pixels is not None:
                for label, projected in zip(
                    view.landmarks_2d, view.reprojected_points_pixels, strict=True
                ):
                    end = tuple(round(value) for value in projected)
                    cv2.circle(image, end, 2, (0, 255, 0), -1, lineType=cv2.LINE_8)
                    if label.point_pixels is not None:
                        cv2.line(
                            image,
                            tuple(round(value) for value in label.point_pixels),
                            end,
                            (0, 255, 255),
                            1,
                            lineType=cv2.LINE_8,
                        )
            cv2.putText(
                image,
                f"{group.status}:{','.join(group.failure_codes) or 'passed'}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_8,
            )
            relative = Path("overlays") / (
                f"{_safe_name(group.record_id)}-{_safe_name(view.camera_id)}.png"
            )
            success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if not success:
                raise RuntimeError("OpenCV failed to encode an overlay")
            payload = bytes(encoded)
            output = output_directory / relative
            output.write_bytes(payload)
            digest = _sha256_bytes(payload)
            relative_text = relative.as_posix()
            aggregate.update(relative_text.encode())
            aggregate.update(payload)
            artifacts.append(
                OverlayArtifact(
                    record_id=group.record_id,
                    camera_id=view.camera_id,
                    path=relative_text,
                    sha256=digest,
                )
            )
    return OverlayReviewIndex(
        materialization_sha256=materialization_sha256,
        overlay_sha256=aggregate.hexdigest(),
        sampled_record_ids=[group.record_id for group in selected],
        artifacts=artifacts,
    )


def materialize_stage_a_labels(
    capture_index: Path,
    *,
    teacher_descriptor: Path,
    model_asset: Path,
    calibration_path: Path,
    quality_profile_path: Path,
    license_registry_path: Path,
    capture_license_record_id: str,
    output_directory: Path,
    detector_factory: (
        Callable[[Path, TeacherModelDescriptor, StageAQualityProfile], LandmarkDetector] | None
    ) = None,
) -> StageAMaterializationSummary:
    descriptor = TeacherModelDescriptor.load(teacher_descriptor)
    descriptor.verify_model_asset(model_asset)
    if descriptor.framework_version != MEDIAPIPE_VERSION:
        raise ValueError("Stage A accepts only the pinned MediaPipe 0.10.35 descriptor")
    calibration = CameraRigCalibrationBundle.load(calibration_path)
    profile = StageAQualityProfile.load(quality_profile_path)
    if profile.pseudo_label_confidence > descriptor.default_label_confidence:
        raise ValueError("Stage A confidence cannot exceed the reviewed teacher descriptor")
    cv2 = _load_cv2(profile)
    cv2.setRNGSeed(profile.review_seed)
    registry = LicenseRegistry.load(license_registry_path)
    registry.verify_local_license_texts(license_registry_path)
    record_ids = [
        capture_license_record_id,
        descriptor.license_record_id,
        OPENCV_SOURCE_ID,
    ]
    for stage in ("teacher-labeling", "base-model-training"):
        registry.admit(
            record_ids,
            stage=cast(Any, stage),
            production=True,
            usage_tier="commercial",
        )
    groups = StageACaptureGroup.load_jsonl(capture_index)
    descriptor_digest = _sha256_file(teacher_descriptor)
    calibration_digest = _sha256_file(calibration_path)
    profile_digest = _sha256_file(quality_profile_path)
    factory = detector_factory or MediaPipeLandmarkDetector
    detector = factory(model_asset, descriptor, profile)
    try:
        materialized = [
            _materialize_group(
                group,
                index_path=capture_index,
                detector=detector,
                descriptor=descriptor,
                calibration=calibration,
                profile=profile,
                descriptor_digest=descriptor_digest,
                calibration_digest=calibration_digest,
                profile_digest=profile_digest,
                cv2=cv2,
            )
            for group in sorted(
                groups, key=lambda item: (item.capture_timestamp_ns, item.record_id)
            )
        ]
    finally:
        detector.close()
    candidates = output_directory / "stage-a-candidates.jsonl"
    materialization_digest = _write_groups(materialized, candidates)
    overlay_index = _render_overlays(
        materialized, output_directory, materialization_digest, profile, cv2
    )
    overlay_index_path = output_directory / "overlay-review-index.json"
    _write_json(overlay_index, overlay_index_path)
    failure_counts: dict[str, int] = {}
    for group in materialized:
        for code in group.failure_codes:
            failure_counts[code] = failure_counts.get(code, 0) + 1
    summary = StageAMaterializationSummary(
        record_count=len(materialized),
        accepted_count=sum(group.status == "accepted" for group in materialized),
        rejected_count=sum(group.status == "rejected" for group in materialized),
        failure_counts=dict(sorted(failure_counts.items())),
        materialization_sha256=materialization_digest,
        overlay_sha256=overlay_index.overlay_sha256,
        candidates=str(candidates),
        overlay_index=str(overlay_index_path),
    )
    _write_json(summary, output_directory / "quality-summary.json")
    return summary


def _relative_reference(path: Path, parent: Path) -> FileReference:
    resolved = path.resolve()
    try:
        reference_path = Path(os.path.relpath(resolved, parent.resolve())).as_posix()
    except ValueError:
        reference_path = resolved.as_posix()
    return FileReference(
        path=reference_path,
        sha256=_sha256_file(path),
    )


def _license_review(record: LicenseRecord) -> LicenseReview:
    return LicenseReview(
        license_id=record.record_id,
        scope=(
            "first_party"
            if record.kind == "first-party-capture"
            else "synthetic"
            if record.smoke_only
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


def _reject_split_overlap(
    splits: dict[str, SplitManifest] | dict[Literal["train", "validation", "test"], SplitManifest],
) -> None:
    for field in ("identities", "sessions", "devices", "camera_ids"):
        owners: dict[str, str] = {}
        for split_name, split in splits.items():
            for value in getattr(split, field):
                previous = owners.setdefault(value, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"Stage A {field[:-1]} {value!r} appears in both "
                        f"{previous!r} and {split_name!r}"
                    )


def _split_for_group(group: MaterializedStageAGroup, plan: StageASplitPlan) -> str:
    matches = [
        name
        for name, split in plan.splits.items()
        if group.identity_id in split.identities
        and group.session_id in split.sessions
        and {view.device_id for view in group.views} == set(split.devices)
        and {view.camera_id for view in group.views} == set(split.camera_ids)
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage A group {group.record_id!r} does not match exactly one split")
    return matches[0]


def build_stage_a_manifest(
    candidates_path: Path,
    *,
    overlay_index_path: Path,
    overlay_review_path: Path,
    split_plan_path: Path,
    teacher_descriptor_path: Path,
    calibration_path: Path,
    quality_profile_path: Path,
    license_registry_path: Path,
    label_catalog_path: Path,
    training_recipe_path: Path,
    capture_license_record_id: str,
    data_revision: str,
    output_path: Path,
) -> DatasetManifest:
    groups = MaterializedStageAGroup.load_jsonl(candidates_path)
    materialization_digest = _sha256_file(candidates_path)
    index = OverlayReviewIndex.load(overlay_index_path)
    index.verify_files(overlay_index_path)
    review = OverlayReviewDecision.load(overlay_review_path)
    if (
        index.materialization_sha256 != materialization_digest
        or review.materialization_sha256 != materialization_digest
        or review.overlay_sha256 != index.overlay_sha256
    ):
        raise ValueError("overlay review does not bind the materialization and overlay digests")
    if not set(index.sampled_record_ids).issubset(review.reviewed_record_ids):
        raise ValueError("approved overlay review does not cover every sampled record")
    descriptor = TeacherModelDescriptor.load(teacher_descriptor_path)
    CameraRigCalibrationBundle.load(calibration_path)
    profile = StageAQualityProfile.load(quality_profile_path)
    plan = StageASplitPlan.load(split_plan_path)
    registry = LicenseRegistry.load(license_registry_path)
    registry.verify_local_license_texts(license_registry_path)
    record_ids = sorted(
        {
            capture_license_record_id,
            descriptor.license_record_id,
            OPENCV_SOURCE_ID,
        }
    )
    for stage in ("teacher-labeling", "base-model-training"):
        registry.admit(
            record_ids,
            stage=cast(Any, stage),
            production=True,
            usage_tier="commercial",
        )
    admitted = registry.admit(
        record_ids,
        stage="base-model-training",
        production=True,
        usage_tier="commercial",
    )
    expected_digests = (
        _sha256_file(teacher_descriptor_path),
        _sha256_file(calibration_path),
        _sha256_file(quality_profile_path),
    )
    accepted: list[MaterializedStageAGroup] = []
    seen: set[str] = set()
    for group in groups:
        if group.record_id in seen:
            raise ValueError(f"duplicate Stage A materialization ID: {group.record_id}")
        seen.add(group.record_id)
        if (
            group.teacher_descriptor_sha256,
            group.calibration_sha256,
            group.quality_profile_sha256,
        ) != expected_digests:
            raise ValueError(f"Stage A group provenance digest drift: {group.record_id}")
        if group.status != "accepted":
            continue
        split_name = _split_for_group(group, plan)
        if split_name == "train" and not 4.5 <= group.effective_fps <= 5.5:
            raise ValueError("Stage A training groups must be sampled at approximately 5 FPS")
        if split_name != "train" and group.effective_fps not in {15.0, 30.0}:
            raise ValueError("Stage A validation/test groups must retain continuous 15 or 30 FPS")
        accepted.append(group.model_copy(update={"review_status": "approved"}))
    if not accepted:
        raise ValueError("no approved Stage A groups remain for the training manifest")
    final_shard = output_path.with_name(f"{output_path.stem}-records.jsonl")
    _write_groups(accepted, final_shard)
    root = output_path.parent.resolve()
    catalog_payload = json.loads(label_catalog_path.read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "schema_version": "ntp-dataset/3.0.0",
        "capture_schema_version": "nana-stage-a-materialization/1.0.0",
        "data_revision": data_revision,
        "digest": "0" * 64,
        "ntp_schema_revision": catalog_payload["ntp_schema_revision"],
        "signal_registry_revision": catalog_payload["signal_registry_revision"],
        "normalization_revision": catalog_payload["normalization_revision"],
        "calibration_revision": catalog_payload["calibration_revision"],
        "feature_revision": catalog_payload["feature_revision"],
        "pipeline_stage": "base-model-training",
        "usage_tier": "commercial",
        "label_catalog": _relative_reference(label_catalog_path, root).model_dump(mode="json"),
        "license_registry": _relative_reference(license_registry_path, root).model_dump(
            mode="json"
        ),
        "license_record_ids": record_ids,
        "anchor_mapping": _relative_reference(teacher_descriptor_path, root).model_dump(
            mode="json"
        ),
        "teacher_model": _relative_reference(teacher_descriptor_path, root).model_dump(mode="json"),
        "calibration_bundle": _relative_reference(calibration_path, root).model_dump(mode="json"),
        "quality_profile": _relative_reference(quality_profile_path, root).model_dump(mode="json"),
        "overlay_review": _relative_reference(overlay_review_path, root).model_dump(mode="json"),
        "training_recipe": _relative_reference(training_recipe_path, root).model_dump(mode="json"),
        "split_plan": _relative_reference(split_plan_path, root).model_dump(mode="json"),
        "record_files": [
            {
                **_relative_reference(final_shard, root).model_dump(mode="json"),
                "record_count": len(accepted),
            }
        ],
        "teacher_sources": [
            TeacherSource(
                source_id=descriptor.teacher_source_id,
                source_type="offline_face",
                version=descriptor.output_contract_revision,
                license_id=descriptor.license_record_id,
            ).model_dump(mode="json"),
            TeacherSource(
                source_id=OPENCV_SOURCE_ID,
                source_type="multiview_pose",
                version=profile.geometry_derivation_revision,
                license_id=OPENCV_SOURCE_ID,
            ).model_dump(mode="json"),
        ],
        "license_reviews": [_license_review(record).model_dump(mode="json") for record in admitted],
        "synchronization": SynchronizationPolicy(
            max_teacher_skew_ns=profile.max_view_skew_ns,
            max_depth_skew_ns=min(profile.max_view_skew_ns, 2_000_000),
        ).model_dump(mode="json"),
        "splits": {name: split.model_dump(mode="json") for name, split in plan.splits.items()},
        "smoke_only": False,
    }
    candidate = DatasetManifest.model_validate(payload)
    payload["digest"] = dataset_digest(candidate)
    manifest = DatasetManifest.model_validate(payload)
    manifest.save(output_path)
    manifest.verify_files(output_path)
    return manifest


def validate_stage_a_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = DatasetManifest.load(manifest_path)
    manifest.verify_files(manifest_path)
    if manifest.capture_schema_version != "nana-stage-a-materialization/1.0.0":
        raise ValueError("manifest does not contain first-party Stage A materialization")
    records = [
        record
        for reference in manifest.record_files
        for record in MaterializedStageAGroup.load_jsonl(manifest.resolve(manifest_path, reference))
    ]
    split_counts = {name: 0 for name in manifest.splits}
    for record in records:
        if record.status != "accepted" or record.review_status != "approved":
            raise ValueError(
                f"Stage A training shard contains an unapproved group: {record.record_id}"
            )
        split_name = _split_for_group(
            record,
            StageASplitPlan(splits=cast(Any, manifest.splits)),
        )
        split_counts[split_name] += 1
    if any(count == 0 for count in split_counts.values()):
        raise ValueError("every Stage A split requires at least one approved group")
    return {
        "schema_version": manifest.capture_schema_version,
        "usage_tier": manifest.usage_tier,
        "data_revision": manifest.data_revision,
        "record_count": len(records),
        "split_record_counts": split_counts,
        "training_ready": True,
        "limitations": (
            "Commercial development labels only; locked-test and release evidence remain required."
        ),
    }


@dataclass(frozen=True, slots=True)
class _StageASample:
    images: Tensor
    targets: dict[str, Tensor]
    confidence: dict[str, Tensor]
    intrinsics: Tensor
    camera_to_capture: Tensor
    record: MaterializedStageAGroup


def _crop_view(
    view: MaterializedStageAView,
    *,
    output_height: int,
    output_width: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if view.face_bbox_xyxy is None:
        raise ValueError("approved Stage A view is missing its face bounding box")
    image = decode_image(view.image_uri, mode=ImageReadMode.RGB).to(torch.float32) / 255.0
    if image.shape[1:] != (view.height, view.width):
        raise ValueError("Stage A image dimensions do not match the materialized view")
    left, top, right, bottom = view.face_bbox_xyxy
    side = max(right - left, bottom - top) * 1.25
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    crop_left = max(0, math.floor(center_x - side * 0.5))
    crop_top = max(0, math.floor(center_y - side * 0.5))
    crop_side = max(
        1,
        min(
            math.ceil(side),
            view.width - crop_left,
            view.height - crop_top,
        ),
    )
    resized = resize(
        crop(image, crop_top, crop_left, crop_side, crop_side),
        [output_height, output_width],
        antialias=True,
    )
    scale_x = output_width / crop_side
    scale_y = output_height / crop_side
    intrinsics = torch.tensor(view.intrinsics, dtype=torch.float32)
    intrinsics[0, 0] *= scale_x
    intrinsics[1, 1] *= scale_y
    intrinsics[0, 2] = (intrinsics[0, 2] - crop_left) * scale_x
    intrinsics[1, 2] = (intrinsics[1, 2] - crop_top) * scale_y
    points = torch.tensor([label.point_pixels for label in view.landmarks_2d], dtype=torch.float32)
    points[:, 0] = ((points[:, 0] - crop_left) * scale_x) / output_width * 2.0 - 1.0
    points[:, 1] = ((points[:, 1] - crop_top) * scale_y) / output_height * 2.0 - 1.0
    confidence = torch.tensor(
        [[label.confidence, label.confidence] for label in view.landmarks_2d],
        dtype=torch.float32,
    )
    inside = (points.abs() <= 1.0).all(dim=-1, keepdim=True)
    confidence = confidence * inside
    return resized, points, confidence, intrinsics


class FirstPartyStageADataset(Dataset[_StageASample]):
    def __init__(self, config: ExperimentConfig, *, split: str) -> None:
        if config.data.manifest is None:
            raise ValueError("first-party Stage A requires data.manifest")
        self._config = config
        self._manifest_path = config.data.manifest.resolve()
        manifest = DatasetManifest.load(self._manifest_path)
        manifest.verify_files(self._manifest_path)
        if manifest.capture_schema_version != "nana-stage-a-materialization/1.0.0":
            raise ValueError("Stage A configuration requires a first-party multiview manifest")
        if split not in manifest.splits:
            raise ValueError(f"Stage A manifest has no {split!r} split")
        selected = manifest.splits[split]
        records = [
            record
            for reference in manifest.record_files
            for record in MaterializedStageAGroup.load_jsonl(
                manifest.resolve(self._manifest_path, reference)
            )
        ]
        self._records = [
            record
            for record in records
            if record.identity_id in selected.identities
            and record.session_id in selected.sessions
            and {view.device_id for view in record.views} == set(selected.devices)
            and {view.camera_id for view in record.views} == set(selected.camera_ids)
        ]
        self._identity_indices = {
            identity: index for index, identity in enumerate(sorted(selected.identities))
        }
        if not self._records:
            raise ValueError(f"no approved Stage A groups remain in split {split!r}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> _StageASample:
        record = self._records[index]
        geometry_points = record.canonical_geometry.points_head_relative
        if geometry_points is None:
            raise ValueError("approved Stage A group lacks canonical geometry")
        images: list[Tensor] = []
        landmarks: list[Tensor] = []
        landmark_weights: list[Tensor] = []
        intrinsics: list[Tensor] = []
        poses: list[Tensor] = []
        pose_weights: list[Tensor] = []
        camera_to_capture: list[Tensor] = []
        for view in record.views:
            image, points, point_confidence, camera_intrinsics = _crop_view(
                view,
                output_height=self._config.model.input_height,
                output_width=self._config.model.input_width,
            )
            if view.pose.translation_head_relative is None or view.pose.orientation_xyzw is None:
                raise ValueError("approved Stage A view lacks pose")
            images.append(image)
            landmarks.append(points)
            landmark_weights.append(point_confidence)
            intrinsics.append(camera_intrinsics)
            poses.append(
                torch.tensor(
                    [
                        *view.pose.translation_head_relative,
                        *view.pose.orientation_xyzw,
                    ],
                    dtype=torch.float32,
                )
            )
            pose_weights.append(torch.full((7,), view.pose.confidence))
            camera_to_capture.append(torch.tensor(view.camera_to_capture, dtype=torch.float32))
        geometry = torch.tensor(geometry_points, dtype=torch.float32).repeat(3, 1, 1)
        geometry_weight = torch.full_like(geometry, record.canonical_geometry.confidence)
        identity = self._identity_indices[record.identity_id]
        return _StageASample(
            images=torch.stack(images),
            targets={
                "landmarks": torch.stack(landmarks),
                "canonical_geometry": geometry,
                "pose": torch.stack(poses),
                "visibility": torch.tensor(
                    [view.visibility for view in record.views], dtype=torch.long
                ),
                "identity": torch.full((3,), identity, dtype=torch.long),
            },
            confidence={
                "landmarks": torch.stack(landmark_weights),
                "canonical_geometry": geometry_weight,
                "pose": torch.stack(pose_weights),
                "visibility": torch.ones(3, 1),
                "identity": torch.ones(3, 1),
            },
            intrinsics=torch.stack(intrinsics),
            camera_to_capture=torch.stack(camera_to_capture),
            record=record,
        )


def _collate_stage_a(samples: list[_StageASample]) -> MultiViewTrackingBatch:
    return MultiViewTrackingBatch(
        images=torch.stack([sample.images for sample in samples]),
        targets={
            name: torch.stack([sample.targets[name] for sample in samples])
            for name in samples[0].targets
        },
        label_confidence={
            name: torch.stack([sample.confidence[name] for sample in samples])
            for name in samples[0].confidence
        },
        camera_intrinsics=torch.stack([sample.intrinsics for sample in samples]),
        camera_to_capture=torch.stack([sample.camera_to_capture for sample in samples]),
        sample_ids=tuple(sample.record.record_id for sample in samples),
        identity_ids=tuple(sample.record.identity_id for sample in samples),
        sequence_ids=tuple(sample.record.session_id for sample in samples),
        expressions=tuple(sample.record.expression for sample in samples),
        timestamps_ns=torch.tensor(
            [sample.record.capture_timestamp_ns for sample in samples], dtype=torch.int64
        ),
    )


def create_first_party_stage_a_loader(
    config: ExperimentConfig,
    *,
    split: str,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader[MultiViewTrackingBatch]:
    dataset = FirstPartyStageADataset(config, split=split)
    generator = torch.Generator().manual_seed(config.training.seed + seed_offset)
    return cast(
        DataLoader[MultiViewTrackingBatch],
        DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=_collate_stage_a,
            num_workers=0,
        ),
    )


__all__ = [
    "CameraRigCalibrationBundle",
    "FirstPartyStageADataset",
    "LandmarkDetector",
    "MaterializedStageAGroup",
    "MediaPipeLandmarkDetector",
    "OverlayReviewDecision",
    "OverlayReviewIndex",
    "ReviewApproval",
    "StageACaptureGroup",
    "StageAMaterializationSummary",
    "StageAQualityProfile",
    "StageASplitPlan",
    "build_stage_a_manifest",
    "create_first_party_stage_a_loader",
    "materialize_stage_a_labels",
    "validate_stage_a_manifest",
]
