"""Latest-frame-only ONNX FaceSpatial reference producer."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, cast
from uuid import UUID, uuid4

import numpy as np
import onnxruntime as ort

from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.export import verify_model_package
from nana_tracking.personalization import LevelACalibration
from nana_tracking.runtime.face_basic import (
    FaceBox,
    empty_skeleton,
    prepare_rgb_roi,
    region,
    unsupported_tracked,
)

_UNSIGNED_SLOTS = {8, 9, 12, 13, 14, 15, 16, 27, 29, 32, 33, 34, 35, 40}


@dataclass(frozen=True, slots=True)
class FaceSpatialPrediction:
    rig: tuple[float, ...]
    pose: tuple[float, ...]
    eye_origins: tuple[tuple[float, float, float], tuple[float, float, float]]
    eye_directions: tuple[tuple[float, float, float], tuple[float, float, float]]
    look_at_head: tuple[float, float, float]
    face_geometry: tuple[tuple[float, float, float], ...]
    visibility: int
    tongue_visible: bool
    confidence: tuple[float, ...]


class FaceSpatialBackend(Protocol):
    input_height: int
    input_width: int

    def infer(self, image: np.ndarray) -> FaceSpatialPrediction: ...


class OrtFaceSpatialBackend:
    """Verified ORT adapter; arrays and provider types do not cross the producer boundary."""

    _OUTPUTS: ClassVar[tuple[str, ...]] = (
        "rig",
        "pose",
        "eye_origins",
        "eye_directions",
        "look_at_head",
        "face_geometry",
        "visibility",
        "tongue_visibility",
        "confidence",
    )

    def __init__(
        self,
        package_dir: Path,
        *,
        providers: list[str] | None = None,
        tensorrt_fp16: bool = False,
    ) -> None:
        verify_model_package(package_dir)
        self.metadata = ModelPackageMetadata.model_validate_json(
            (package_dir / "runtime-metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.supported_signals != list(range(1, 42)):
            raise ValueError("model package does not declare the complete SpatialSet")
        required_structures = {
            "head_geometry",
            "eye_geometry",
            "look_at_point",
            "face_geometry",
        }
        if set(self.metadata.supported_structures) != required_structures:
            raise ValueError("model package does not declare every Spatial structure")
        if not self.metadata.geometry_topology_revision:
            raise ValueError("FaceSpatial package lacks a geometry topology revision")
        requested = providers or ["CPUExecutionProvider"]
        if tensorrt_fp16 and "TensorrtExecutionProvider" not in requested:
            raise ValueError("TensorRT FP16 requires TensorrtExecutionProvider")
        available = cast(list[str], ort.get_available_providers())
        unavailable = set(requested).difference(available)
        if unavailable:
            raise RuntimeError(f"requested ONNX Runtime providers are unavailable: {unavailable}")
        provider_specs: list[str | tuple[str, dict[str, str]]] = [
            (provider, {"trt_fp16_enable": "1"})
            if provider == "TensorrtExecutionProvider" and tensorrt_fp16
            else provider
            for provider in requested
        ]
        self._session = ort.InferenceSession(
            str(package_dir / "model.onnx"), providers=provider_specs
        )
        self.input_height = self.metadata.input_shape[2]
        self.input_width = self.metadata.input_shape[3]

    @property
    def active_providers(self) -> list[str]:
        return list(self._session.get_providers())

    def infer(self, image: np.ndarray) -> FaceSpatialPrediction:
        if image.shape != (1, 3, self.input_height, self.input_width):
            raise ValueError("backend input does not match the fixed model ROI shape")
        outputs = cast(list[np.ndarray], self._session.run(self._OUTPUTS, {"image": image}))
        (
            rig,
            pose,
            eye_origins,
            eye_directions,
            look_at_head,
            face_geometry,
            visibility,
            tongue_visibility,
            confidence,
        ) = outputs
        return FaceSpatialPrediction(
            rig=tuple(float(value) for value in rig[0]),
            pose=tuple(float(value) for value in pose[0]),
            eye_origins=tuple(
                (float(eye[0]), float(eye[1]), float(eye[2])) for eye in eye_origins[0]
            ),  # type: ignore[arg-type]
            eye_directions=tuple(
                (float(eye[0]), float(eye[1]), float(eye[2])) for eye in eye_directions[0]
            ),  # type: ignore[arg-type]
            look_at_head=(
                float(look_at_head[0, 0]),
                float(look_at_head[0, 1]),
                float(look_at_head[0, 2]),
            ),
            face_geometry=tuple(
                (float(point[0]), float(point[1]), float(point[2])) for point in face_geometry[0]
            ),
            visibility=int(np.argmax(visibility[0])),
            tongue_visible=bool(np.argmax(tongue_visibility[0])),
            confidence=tuple(float(value) for value in confidence[0]),
        )


def _tracked(
    value: object | None,
    confidence: float,
    state: str,
    capture_timestamp_ns: int,
) -> dict[str, object]:
    return {
        "value": value,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "state": state,
        "sample_capture_timestamp_ns": capture_timestamp_ns,
        "prediction_horizon_ns": 0,
    }


def _vector(value: np.ndarray) -> dict[str, object]:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def _normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("geometry vectors must contain three finite values")
    norm = float(np.linalg.norm(value))
    return fallback.copy() if norm < 1e-6 else value / norm


def _rotate_by_quaternion(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    xyz = quaternion[:3]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + quaternion[3] * vector)


class FaceSpatialProducer:
    """Convert one RGB Spatial inference into a framework-neutral NTP diagnostic event."""

    def __init__(
        self,
        backend: FaceSpatialBackend,
        *,
        calibration: LevelACalibration | None = None,
        session_id: UUID | None = None,
        generation: int = 0,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.backend = backend
        self.calibration = calibration
        self.session_id = session_id or uuid4()
        self.generation = generation
        self._clock = clock
        self._input = np.empty((1, 3, backend.input_height, backend.input_width), dtype=np.float32)
        self.last_stage_timings_ns: dict[str, int] = {}

    @property
    def descriptor_event(self) -> dict[str, object]:
        return {
            "kind": "descriptor",
            "value": {
                "revisions": {
                    "protocol": {"major": 1, "minor": 0},
                    "schema_revision": 1,
                    "signal_registry": {"major": 1, "minor": 0, "patch": 0},
                    "normalization": {"major": 1, "minor": 0, "patch": 0},
                    "calibration": {"major": 1, "minor": 0, "patch": 0},
                    "features": {"major": 1, "minor": 0, "patch": 0},
                },
                "guaranteed_profile": "Spatial",
                "supported_signals": list(range(1, 42)),
                "supported_structures": 15,
                "features": 0,
            },
        }

    def produce(
        self,
        frame: np.ndarray,
        *,
        roi: FaceBox | None,
        capture_timestamp_ns: int,
        sequence: int,
    ) -> dict[str, object]:
        preprocess_started = time.perf_counter_ns()
        prepare_rgb_roi(frame, roi, self._input)
        preprocess_ns = time.perf_counter_ns() - preprocess_started
        inference_started = time.perf_counter_ns()
        prediction = self.backend.infer(self._input)
        inference_ns = time.perf_counter_ns() - inference_started
        readback_started = time.perf_counter_ns()
        values = np.asarray(prediction.rig, dtype=np.float32)
        confidence = np.asarray(prediction.confidence, dtype=np.float32)
        if values.shape != (41,) or confidence.shape != (41,):
            raise ValueError("backend must return exactly 41 Spatial values and confidences")
        if not np.isfinite(values).all() or not np.isfinite(confidence).all():
            raise ValueError("backend Spatial output must be finite")
        if self.calibration is not None:
            values[:36] = self.calibration.apply(values[:36])
        for slot in range(41):
            if slot in {36, 38}:
                values[slot] = np.clip(values[slot], -1.2, 1.2)
            elif slot in {37, 39}:
                values[slot] = np.clip(values[slot], -0.8, 0.8)
            else:
                values[slot] = np.clip(values[slot], 0.0 if slot in _UNSIGNED_SLOTS else -1.0, 1.0)

        if prediction.visibility not in {0, 1, 2}:
            raise ValueError("visibility must be Observed, Occluded, or OutOfFrame")
        face_state = ("Observed", "Occluded", "OutOfFrame")[prediction.visibility]
        tongue_state = (
            "Observed"
            if face_state == "Observed" and prediction.tongue_visible
            else "Occluded"
            if face_state == "Observed"
            else face_state
        )
        slots: list[dict[str, object]] = []
        for slot in range(88):
            if slot >= 41:
                slots.append(unsupported_tracked())
                continue
            state = tongue_state if slot == 40 else face_state
            slots.append(
                _tracked(
                    float(values[slot]) if state == "Observed" else None,
                    float(confidence[slot]),
                    state,
                    capture_timestamp_ns,
                )
            )

        pose = np.asarray(prediction.pose, dtype=np.float32)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("backend pose must be seven finite xyz+xyzw values")
        quaternion = pose[3:]
        quaternion_norm = float(np.linalg.norm(quaternion))
        quaternion = (
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if quaternion_norm < 1e-6
            else quaternion / quaternion_norm
        )
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        carries_face = face_state == "Observed"
        overall = float(np.clip(confidence.mean(), 0.0, 1.0))
        head_pose_value = (
            {
                "parent_space": "Camera",
                "length_basis": "HeadRelative",
                "position": _vector(pose[:3]),
                "orientation_xyzw": {
                    "x": float(quaternion[0]),
                    "y": float(quaternion[1]),
                    "z": float(quaternion[2]),
                    "w": float(quaternion[3]),
                },
            }
            if carries_face
            else None
        )

        eye_origins = np.asarray(prediction.eye_origins, dtype=np.float32)
        eye_directions = np.asarray(prediction.eye_directions, dtype=np.float32)
        look_at_head = np.asarray(prediction.look_at_head, dtype=np.float32)
        face_geometry = np.asarray(prediction.face_geometry, dtype=np.float32)
        if eye_origins.shape != (2, 3) or not np.isfinite(eye_origins).all():
            raise ValueError("eye origins must be finite [2, 3] head-relative coordinates")
        if eye_directions.shape != (2, 3) or not np.isfinite(eye_directions).all():
            raise ValueError("eye directions must be finite [2, 3] vectors")
        if look_at_head.shape != (3,) or not np.isfinite(look_at_head).all():
            raise ValueError("look-at must be a finite three-value head-relative vector")
        if (
            face_geometry.ndim != 2
            or face_geometry.shape[0] == 0
            or face_geometry.shape[1] != 3
            or not np.isfinite(face_geometry).all()
        ):
            raise ValueError("face geometry must be finite [points, 3] canonical coordinates")
        normalized_directions = np.stack(
            [
                _normalize(direction, np.array([0.0, 0.0, 1.0], dtype=np.float32))
                for direction in eye_directions
            ]
        )
        camera_look_at = pose[:3] + _rotate_by_quaternion(look_at_head, quaternion)
        eyes: dict[str, object] = {}
        for side, index in (("left", 0), ("right", 1)):
            eyes[side] = {
                "origin_head": _tracked(
                    {
                        "space": "HeadLocal",
                        "length_basis": "HeadRelative",
                        "value": _vector(eye_origins[index]),
                    }
                    if carries_face
                    else None,
                    overall,
                    face_state,
                    capture_timestamp_ns,
                ),
                "direction_head": _tracked(
                    {"space": "HeadLocal", "value": _vector(normalized_directions[index])}
                    if carries_face
                    else None,
                    overall,
                    face_state,
                    capture_timestamp_ns,
                ),
            }
        result = {
            "session_id": list(self.session_id.bytes),
            "generation": self.generation,
            "sequence": sequence,
            "capture_timestamp_ns": capture_timestamp_ns,
            "produced_timestamp_ns": max(int(self._clock()), capture_timestamp_ns),
            "rig": {"slots": slots},
            "geometry": {
                "head_camera_pose": _tracked(
                    head_pose_value, overall, face_state, capture_timestamp_ns
                ),
                "eyes": eyes,
                "look_at_camera": _tracked(
                    {
                        "space": "Camera",
                        "length_basis": "HeadRelative",
                        "value": _vector(camera_look_at),
                    }
                    if carries_face
                    else None,
                    overall,
                    face_state,
                    capture_timestamp_ns,
                ),
                # Signal Registry 1.0 assigns no stable landmark semantic IDs. The versioned model
                # geometry remains an artifact output; vendor or topology indices never enter NTP.
                "face_geometry_state": face_state,
                "face_landmarks": [],
            },
            "skeleton": empty_skeleton(),
            "quality": {
                "overall_confidence": overall,
                "face": region(overall, face_state),
                "eyes": region(float(confidence[36:40].mean()), face_state),
                "torso": region(),
                "arm": {"left": region(), "right": region()},
                "auricle": {"left": region(), "right": region()},
                "stabilization_revision": {"major": 1, "minor": 0, "patch": 0},
            },
        }
        self.last_stage_timings_ns = {
            "preprocess": preprocess_ns,
            "inference": inference_ns,
            "readback": time.perf_counter_ns() - readback_started,
        }
        return {"kind": "result", "value": result}
