"""Latest-frame-only ONNX FaceBasic reference producer."""

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID, uuid4

import numpy as np
import onnxruntime as ort

from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.export import verify_model_package
from nana_tracking.personalization import LevelACalibration

_UNSIGNED_SLOTS = {8, 9, 12, 13, 14, 15, 16, 27, 29, 32, 33, 34, 35}


@dataclass(frozen=True, slots=True)
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int

    def clamp(self, width: int, height: int) -> FaceBox:
        left = min(max(self.left, 0), width - 1)
        top = min(max(self.top, 0), height - 1)
        right = min(max(self.right, left + 1), width)
        bottom = min(max(self.bottom, top + 1), height)
        return FaceBox(left, top, right, bottom)


@dataclass(frozen=True, slots=True)
class FaceBasicPrediction:
    rig: tuple[float, ...]
    pose: tuple[float, ...]
    visibility: int
    confidence: tuple[float, ...]


class FaceBasicBackend(Protocol):
    input_height: int
    input_width: int

    def infer(self, image: np.ndarray) -> FaceBasicPrediction: ...


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[tuple[FaceBox, float]]: ...


class FaceRoiTracker:
    """Keep a stable square ROI between bounded detector refreshes."""

    def __init__(
        self,
        detector: FaceDetector,
        *,
        detection_interval: int = 5,
        minimum_score: float = 0.6,
        smoothing: float = 0.7,
        margin: float = 0.2,
        maximum_missed: int = 3,
    ) -> None:
        if detection_interval < 1 or maximum_missed < 0:
            raise ValueError("ROI tracker intervals must be bounded and non-negative")
        if not 0.0 <= minimum_score <= 1.0 or not 0.0 <= smoothing < 1.0:
            raise ValueError("ROI tracker score and smoothing must be in range")
        if margin < 0.0:
            raise ValueError("ROI margin must be non-negative")
        self._detector = detector
        self._detection_interval = detection_interval
        self._minimum_score = minimum_score
        self._smoothing = smoothing
        self._margin = margin
        self._maximum_missed = maximum_missed
        self._current: FaceBox | None = None
        self._missed = 0

    def update(self, frame: np.ndarray, sequence: int) -> FaceBox | None:
        if self._current is not None and sequence % self._detection_interval != 0:
            return self._current
        candidates = [
            (box, score)
            for box, score in self._detector.detect(frame)
            if score >= self._minimum_score
        ]
        if not candidates:
            self._missed += 1
            if self._missed > self._maximum_missed:
                self._current = None
            return self._current
        self._missed = 0
        selected = max(candidates, key=lambda item: (self._selection_score(item[0]), item[1]))[0]
        selected = self._square_with_margin(selected, frame.shape[1], frame.shape[0])
        if self._current is None:
            self._current = selected
        else:
            keep = self._smoothing
            take = 1.0 - keep
            self._current = FaceBox(
                round(self._current.left * keep + selected.left * take),
                round(self._current.top * keep + selected.top * take),
                round(self._current.right * keep + selected.right * take),
                round(self._current.bottom * keep + selected.bottom * take),
            ).clamp(frame.shape[1], frame.shape[0])
        return self._current

    def reset(self) -> None:
        self._current = None
        self._missed = 0

    def _selection_score(self, candidate: FaceBox) -> float:
        area = max(0, candidate.right - candidate.left) * max(0, candidate.bottom - candidate.top)
        if self._current is None:
            return float(area)
        intersection_width = max(
            0, min(candidate.right, self._current.right) - max(candidate.left, self._current.left)
        )
        intersection_height = max(
            0, min(candidate.bottom, self._current.bottom) - max(candidate.top, self._current.top)
        )
        intersection = intersection_width * intersection_height
        current_area = (self._current.right - self._current.left) * (
            self._current.bottom - self._current.top
        )
        union = area + current_area - intersection
        return float(intersection / max(union, 1))

    def _square_with_margin(self, box: FaceBox, width: int, height: int) -> FaceBox:
        center_x = (box.left + box.right) / 2.0
        center_y = (box.top + box.bottom) / 2.0
        side = max(box.right - box.left, box.bottom - box.top) * (1.0 + 2.0 * self._margin)
        half = side / 2.0
        return FaceBox(
            round(center_x - half),
            round(center_y - half),
            round(center_x + half),
            round(center_y + half),
        ).clamp(width, height)


class OrtFaceBasicBackend:
    """Verified ONNX Runtime backend; backend arrays never escape this adapter."""

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
        if self.metadata.supported_signals != list(range(1, 37)):
            raise ValueError("model package does not declare the complete BasicSet")
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
        self._names = ["rig", "pose", "visibility", "confidence"]

    @property
    def active_providers(self) -> list[str]:
        return list(self._session.get_providers())

    def infer(self, image: np.ndarray) -> FaceBasicPrediction:
        if image.shape != (1, 3, self.input_height, self.input_width):
            raise ValueError("backend input does not match the fixed model ROI shape")
        outputs = cast(list[np.ndarray], self._session.run(self._names, {"image": image}))
        rig, pose, visibility, confidence = outputs
        return FaceBasicPrediction(
            rig=tuple(float(value) for value in rig[0]),
            pose=tuple(float(value) for value in pose[0]),
            visibility=int(np.argmax(visibility[0])),
            confidence=tuple(float(value) for value in confidence[0]),
        )


def _unsupported_tracked() -> dict[str, object]:
    return {
        "value": None,
        "confidence": 0.0,
        "state": "Unsupported",
        "sample_capture_timestamp_ns": 0,
        "prediction_horizon_ns": 0,
    }


def _unsupported_side_map() -> dict[str, object]:
    return {"left": _unsupported_tracked(), "right": _unsupported_tracked()}


def _empty_skeleton() -> dict[str, object]:
    return {
        "torso_camera_pose": _unsupported_tracked(),
        "shoulder": _unsupported_side_map(),
        "elbow": _unsupported_side_map(),
        "wrist": _unsupported_side_map(),
        "upper_arm_direction_torso": _unsupported_side_map(),
        "forearm_direction_torso": _unsupported_side_map(),
        "upper_arm_twist": _unsupported_side_map(),
        "forearm_twist": _unsupported_side_map(),
    }


def _region(confidence: float = 0.0, state: str = "Unsupported") -> dict[str, object]:
    return {"confidence": confidence, "state": state}


class FaceBasicProducer:
    """Convert one model ROI prediction into a framework-neutral NTP diagnostic event."""

    def __init__(
        self,
        backend: FaceBasicBackend,
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
                "guaranteed_profile": "Basic",
                "supported_signals": list(range(1, 37)),
                "supported_structures": 1,
                "features": 0,
            },
        }

    def _prepare(self, frame: np.ndarray, roi: FaceBox | None) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frames must be uint8 HWC RGB arrays")
        height, width, _ = frame.shape
        box = (roi or FaceBox(0, 0, width, height)).clamp(width, height)
        ys = np.linspace(box.top, box.bottom - 1, self.backend.input_height).astype(np.intp)
        xs = np.linspace(box.left, box.right - 1, self.backend.input_width).astype(np.intp)
        for output_y, source_y in enumerate(ys):
            row = frame[source_y, xs, :]
            np.multiply(row.T, 1.0 / 255.0, out=self._input[0, :, output_y, :])

    def produce(
        self,
        frame: np.ndarray,
        *,
        roi: FaceBox | None,
        capture_timestamp_ns: int,
        sequence: int,
    ) -> dict[str, object]:
        self._prepare(frame, roi)
        prediction = self.backend.infer(self._input)
        values = np.asarray(prediction.rig, dtype=np.float32)
        if values.shape != (36,) or len(prediction.confidence) != 36:
            raise ValueError("backend must return exactly 36 values and confidences")
        if self.calibration is not None:
            values = self.calibration.apply(values)
        for slot in range(36):
            values[slot] = np.clip(values[slot], 0.0 if slot in _UNSIGNED_SLOTS else -1.0, 1.0)

        state = ("Observed", "Occluded", "OutOfFrame")[prediction.visibility]
        carries_value = state == "Observed"
        slots: list[dict[str, object]] = []
        for slot in range(88):
            if slot >= 36:
                slots.append(_unsupported_tracked())
                continue
            confidence = float(np.clip(prediction.confidence[slot], 0.0, 1.0))
            slots.append(
                {
                    "value": float(values[slot]) if carries_value else None,
                    "confidence": confidence,
                    "state": state,
                    "sample_capture_timestamp_ns": capture_timestamp_ns,
                    "prediction_horizon_ns": 0,
                }
            )

        pose = np.asarray(prediction.pose, dtype=np.float32)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("backend pose must be seven finite xyz+xyzw values")
        quaternion = pose[3:]
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-6:
            quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            quaternion = quaternion / norm
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        overall = float(np.mean(prediction.confidence))
        head_pose = {
            "value": (
                {
                    "parent_space": "Camera",
                    "length_basis": "HeadRelative",
                    "position": {
                        "x": float(pose[0]),
                        "y": float(pose[1]),
                        "z": float(pose[2]),
                    },
                    "orientation_xyzw": {
                        "x": float(quaternion[0]),
                        "y": float(quaternion[1]),
                        "z": float(quaternion[2]),
                        "w": float(quaternion[3]),
                    },
                }
                if carries_value
                else None
            ),
            "confidence": overall,
            "state": state,
            "sample_capture_timestamp_ns": capture_timestamp_ns,
            "prediction_horizon_ns": 0,
        }
        produced_timestamp_ns = max(int(self._clock()), capture_timestamp_ns)
        result = {
            "session_id": list(self.session_id.bytes),
            "generation": self.generation,
            "sequence": sequence,
            "capture_timestamp_ns": capture_timestamp_ns,
            "produced_timestamp_ns": produced_timestamp_ns,
            "rig": {"slots": slots},
            "geometry": {
                "head_camera_pose": head_pose,
                "eyes": {
                    "left": {
                        "origin_head": _unsupported_tracked(),
                        "direction_head": _unsupported_tracked(),
                    },
                    "right": {
                        "origin_head": _unsupported_tracked(),
                        "direction_head": _unsupported_tracked(),
                    },
                },
                "look_at_camera": _unsupported_tracked(),
                "face_geometry_state": "Unsupported",
                "face_landmarks": [],
            },
            "skeleton": _empty_skeleton(),
            "quality": {
                "overall_confidence": overall,
                "face": _region(overall, state),
                "eyes": _region(overall, state),
                "torso": _region(),
                "arm": {"left": _region(), "right": _region()},
                "auricle": {"left": _region(), "right": _region()},
                "stabilization_revision": {"major": 1, "minor": 0, "patch": 0},
            },
        }
        return {"kind": "result", "value": result}


@dataclass(frozen=True, slots=True)
class _Submission:
    frame: np.ndarray
    roi: FaceBox | None
    capture_timestamp_ns: int
    sequence: int


class LatestFrameRuntime:
    """Bounded single-slot worker that replaces stale pending frames."""

    def __init__(
        self,
        producer: FaceBasicProducer,
        *,
        roi_tracker: FaceRoiTracker | None = None,
    ) -> None:
        self._producer = producer
        self._roi_tracker = roi_tracker
        self._condition = threading.Condition()
        self._pending: _Submission | None = None
        self._latest: dict[str, object] | None = None
        self._closed = False
        self._error: BaseException | None = None
        self._sequence = 0
        self.dropped_frames = 0
        self._worker = threading.Thread(target=self._run, name="face-basic-latest", daemon=True)
        self._worker.start()

    def submit(
        self,
        frame: np.ndarray,
        *,
        capture_timestamp_ns: int,
        roi: FaceBox | None = None,
    ) -> int:
        with self._condition:
            self._raise_if_error()
            if self._closed:
                raise RuntimeError("runtime is closed")
            self._sequence += 1
            if self._pending is not None:
                self.dropped_frames += 1
            self._pending = _Submission(frame, roi, capture_timestamp_ns, self._sequence)
            self._condition.notify()
            return self._sequence

    def poll_latest(self) -> dict[str, object] | None:
        with self._condition:
            self._raise_if_error()
            result = self._latest
            self._latest = None
            return result

    def wait_latest(self, timeout: float = 1.0) -> dict[str, object] | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None and not self._closed:
                self._raise_if_error()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            self._raise_if_error()
            result = self._latest
            self._latest = None
            return result

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            raise RuntimeError("FaceBasic runtime worker did not stop")
        self._raise_if_error()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    submission = self._pending
                    self._pending = None
                assert submission is not None
                roi = submission.roi
                if roi is None and self._roi_tracker is not None:
                    roi = self._roi_tracker.update(submission.frame, submission.sequence)
                result = self._producer.produce(
                    submission.frame,
                    roi=roi,
                    capture_timestamp_ns=submission.capture_timestamp_ns,
                    sequence=submission.sequence,
                )
                with self._condition:
                    self._latest = result
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._closed = True
                self._condition.notify_all()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("FaceBasic runtime worker failed") from self._error


def write_diagnostic_stream(
    path: Path,
    descriptor: dict[str, object],
    results: list[dict[str, object]],
) -> None:
    lines = [json.dumps(descriptor, separators=(",", ":"))]
    lines.extend(json.dumps(result, separators=(",", ":")) for result in results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
