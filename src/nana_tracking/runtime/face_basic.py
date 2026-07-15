"""Latest-frame-only ONNX FaceBasic reference producer."""

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID, uuid4

import numpy as np
import onnxruntime as ort

from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.export import verify_model_package
from nana_tracking.personalization import LevelACalibration
from nana_tracking.runtime.temporal import CausalTemporalRefiner, TemporalSample, TemporalState

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


class NtpFrameProducer(Protocol):
    def produce(
        self,
        frame: np.ndarray,
        *,
        roi: FaceBox | None,
        capture_timestamp_ns: int,
        sequence: int,
    ) -> dict[str, object]: ...


class NumericalAdapter(Protocol):
    def apply(self, base_values: tuple[float, ...]) -> tuple[float, ...]: ...


class RuntimeMode(StrEnum):
    PERFORMANCE = "Performance"
    QUALITY = "Quality"


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    mode: RuntimeMode
    guaranteed_profile: str
    latest_frame_only: bool
    temporal_refiner: bool
    stage_telemetry: bool


@dataclass(frozen=True, slots=True)
class RuntimeTelemetry:
    samples: int
    submitted_frames: int
    dropped_frames: int
    mailbox_wait_ms: dict[str, float]
    preprocess_ms: dict[str, float]
    inference_ms: dict[str, float]
    readback_ms: dict[str, float]
    producer_total_ms: dict[str, float]
    result_age_ms: dict[str, float]


def _latency_summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    ordered = sorted(values)

    def percentile(value: float) -> float:
        return ordered[min(round((len(ordered) - 1) * value), len(ordered) - 1)] / 1_000_000.0

    return {
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": sum(values) / len(values) / 1_000_000.0,
    }


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


def unsupported_tracked() -> dict[str, object]:
    return {
        "value": None,
        "confidence": 0.0,
        "state": "Unsupported",
        "sample_capture_timestamp_ns": 0,
        "prediction_horizon_ns": 0,
    }


def unsupported_side_map() -> dict[str, object]:
    return {"left": unsupported_tracked(), "right": unsupported_tracked()}


def empty_skeleton() -> dict[str, object]:
    return {
        "torso_camera_pose": unsupported_tracked(),
        "shoulder": unsupported_side_map(),
        "elbow": unsupported_side_map(),
        "wrist": unsupported_side_map(),
        "upper_arm_direction_torso": unsupported_side_map(),
        "forearm_direction_torso": unsupported_side_map(),
        "upper_arm_twist": unsupported_side_map(),
        "forearm_twist": unsupported_side_map(),
    }


def region(confidence: float = 0.0, state: str = "Unsupported") -> dict[str, object]:
    return {"confidence": confidence, "state": state}


class RgbRoiWorkspace:
    """Reusable nearest-neighbour RGB ROI resize storage.

    The largest scratch array is one resized RGB row. Moving ROIs and source-resolution changes do
    not allocate an intermediate image-sized tensor.
    """

    def __init__(self, output_height: int, output_width: int) -> None:
        if output_height < 1 or output_width < 1:
            raise ValueError("ROI output dimensions must be positive")
        self.output_height = output_height
        self.output_width = output_width
        self._x_fraction = np.linspace(0.0, 1.0, output_width, dtype=np.float64)
        self._y_fraction = np.linspace(0.0, 1.0, output_height, dtype=np.float64)
        self._x_work = np.empty(output_width, dtype=np.float64)
        self._y_work = np.empty(output_height, dtype=np.float64)
        self._x_indices = np.empty(output_width, dtype=np.intp)
        self._y_indices = np.empty(output_height, dtype=np.intp)
        self._row_rgb = np.empty((output_width, 3), dtype=np.uint8)
        self._row_chw = self._row_rgb.T
        self._scale = np.float32(1.0 / 255.0)
        self._source_width = -1
        self._source_height = -1
        self._box = FaceBox(-1, -1, -1, -1)
        self._output_owner: np.ndarray | None = None
        self._output_rows: tuple[np.ndarray, ...] = ()

    @property
    def workspace_bytes(self) -> int:
        arrays = (
            self._x_fraction,
            self._y_fraction,
            self._x_work,
            self._y_work,
            self._x_indices,
            self._y_indices,
            self._row_rgb,
        )
        return sum(array.nbytes for array in arrays)

    def prepare(self, frame: np.ndarray, roi: FaceBox | None, output: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frames must be uint8 HWC RGB arrays")
        expected_shape = (1, 3, self.output_height, self.output_width)
        if output.shape != expected_shape or output.dtype != np.float32:
            raise ValueError(f"output must be a preallocated {expected_shape} float32 array")
        height, width, _ = frame.shape
        if height < 1 or width < 1:
            raise ValueError("frames must have positive height and width")
        box = (roi or FaceBox(0, 0, width, height)).clamp(width, height)
        if width != self._source_width or height != self._source_height or box != self._box:
            self._fill_indices(
                self._x_fraction,
                self._x_work,
                self._x_indices,
                box.left,
                box.right,
            )
            self._fill_indices(
                self._y_fraction,
                self._y_work,
                self._y_indices,
                box.top,
                box.bottom,
            )
            self._source_width = width
            self._source_height = height
            self._box = box
        if output is not self._output_owner:
            self._output_owner = output
            self._output_rows = tuple(
                output[0, :, output_y, :] for output_y in range(self.output_height)
            )
        for source_y, output_row in zip(self._y_indices, self._output_rows, strict=True):
            np.take(frame[source_y], self._x_indices, axis=0, out=self._row_rgb)
            np.multiply(self._row_chw, self._scale, out=output_row)

    @staticmethod
    def _fill_indices(
        fraction: np.ndarray,
        work: np.ndarray,
        indices: np.ndarray,
        start: int,
        stop: int,
    ) -> None:
        np.multiply(fraction, stop - start - 1, out=work)
        np.add(work, start, out=work)
        np.copyto(indices, work, casting="unsafe")


def prepare_rgb_roi(
    frame: np.ndarray,
    roi: FaceBox | None,
    output: np.ndarray,
    *,
    workspace: RgbRoiWorkspace | None = None,
) -> None:
    """Resize one uint8 RGB ROI into a preallocated NCHW float32 tensor."""

    if output.ndim != 4 or output.shape[0:2] != (1, 3) or output.dtype != np.float32:
        raise ValueError("output must be a preallocated [1, 3, height, width] float32 array")
    active = workspace or RgbRoiWorkspace(output.shape[2], output.shape[3])
    active.prepare(frame, roi, output)


class FaceBasicProducer:
    """Convert one model ROI prediction into a framework-neutral NTP diagnostic event."""

    def __init__(
        self,
        backend: FaceBasicBackend,
        *,
        calibration: LevelACalibration | None = None,
        level_b_adapter: NumericalAdapter | None = None,
        session_id: UUID | None = None,
        generation: int = 0,
        temporal_refiner: CausalTemporalRefiner | None = None,
        camera_id: str = "default-camera",
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.backend = backend
        self.calibration = calibration
        self.level_b_adapter = level_b_adapter
        self.session_id = session_id or uuid4()
        self.generation = generation
        if temporal_refiner is not None and temporal_refiner.signal_ids != tuple(range(1, 37)):
            raise ValueError("FaceBasic temporal refiner must cover Signal IDs 1..36")
        self.temporal_refiner = temporal_refiner
        self.camera_id = camera_id
        self._clock = clock
        self._input = np.empty((1, 3, backend.input_height, backend.input_width), dtype=np.float32)
        self._roi_workspace = RgbRoiWorkspace(backend.input_height, backend.input_width)
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
                "guaranteed_profile": "Basic",
                "supported_signals": list(range(1, 37)),
                "supported_structures": 1,
                "features": 0,
            },
        }

    def _prepare(self, frame: np.ndarray, roi: FaceBox | None) -> None:
        prepare_rgb_roi(frame, roi, self._input, workspace=self._roi_workspace)

    def produce(
        self,
        frame: np.ndarray,
        *,
        roi: FaceBox | None,
        capture_timestamp_ns: int,
        sequence: int,
    ) -> dict[str, object]:
        preprocess_started = time.perf_counter_ns()
        self._prepare(frame, roi)
        preprocess_ns = time.perf_counter_ns() - preprocess_started
        inference_started = time.perf_counter_ns()
        prediction = self.backend.infer(self._input)
        inference_ns = time.perf_counter_ns() - inference_started
        readback_started = time.perf_counter_ns()
        values = np.asarray(prediction.rig, dtype=np.float32)
        if values.shape != (36,) or len(prediction.confidence) != 36:
            raise ValueError("backend must return exactly 36 values and confidences")
        if self.calibration is not None:
            values = self.calibration.apply(values)
        if self.level_b_adapter is not None:
            values = np.asarray(
                self.level_b_adapter.apply(tuple(float(value) for value in values)),
                dtype=np.float32,
            )
            if values.shape != (36,) or not np.isfinite(values).all():
                raise ValueError("Level B adapter must return 36 finite Basic values")
        for slot in range(36):
            values[slot] = np.clip(values[slot], 0.0 if slot in _UNSIGNED_SLOTS else -1.0, 1.0)

        observation_state = ("Observed", "Occluded", "OutOfFrame")[prediction.visibility]
        temporal: tuple[TemporalSample, ...] | None = None
        if self.temporal_refiner is not None:
            calibration_revision = (
                self.calibration.calibration_revision
                if self.calibration is not None
                else "uncalibrated"
            )
            adapter_digest = getattr(
                getattr(self.level_b_adapter, "metadata", None),
                "adapter_digest",
                "no-adapter",
            )
            temporal = self.temporal_refiner.process(
                [
                    TemporalSample(
                        float(values[slot]) if observation_state == "Observed" else None,
                        float(np.clip(prediction.confidence[slot], 0.0, 1.0)),
                        TemporalState(observation_state),
                        capture_timestamp_ns,
                    )
                    for slot in range(36)
                ],
                capture_timestamp_ns=capture_timestamp_ns,
                session_id=self.session_id.bytes,
                generation=self.generation,
                camera_id=self.camera_id,
                calibration_revision=f"{calibration_revision}:{adapter_digest}",
            )
        slots: list[dict[str, object]] = []
        for slot in range(88):
            if slot >= 36:
                slots.append(unsupported_tracked())
                continue
            refined = temporal[slot] if temporal is not None else None
            state = refined.state.value if refined is not None else observation_state
            confidence = (
                refined.confidence
                if refined is not None
                else float(np.clip(prediction.confidence[slot], 0.0, 1.0))
            )
            slots.append(
                {
                    "value": refined.value
                    if refined is not None
                    else float(values[slot])
                    if observation_state == "Observed"
                    else None,
                    "confidence": confidence,
                    "state": state,
                    "sample_capture_timestamp_ns": refined.sample_capture_timestamp_ns
                    if refined is not None
                    else capture_timestamp_ns,
                    "prediction_horizon_ns": refined.prediction_horizon_ns
                    if refined is not None
                    else 0,
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
        overall = float(
            np.mean([sample.confidence for sample in temporal])
            if temporal is not None
            else np.mean(prediction.confidence)
        )
        carries_value = observation_state == "Observed"
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
            "state": observation_state,
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
                        "origin_head": unsupported_tracked(),
                        "direction_head": unsupported_tracked(),
                    },
                    "right": {
                        "origin_head": unsupported_tracked(),
                        "direction_head": unsupported_tracked(),
                    },
                },
                "look_at_camera": unsupported_tracked(),
                "face_geometry_state": "Unsupported",
                "face_landmarks": [],
            },
            "skeleton": empty_skeleton(),
            "quality": {
                "overall_confidence": overall,
                "face": region(
                    overall,
                    temporal[0].state.value if temporal is not None else observation_state,
                ),
                "eyes": region(
                    overall,
                    temporal[6].state.value if temporal is not None else observation_state,
                ),
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


@dataclass(frozen=True, slots=True)
class _Submission:
    frame: np.ndarray
    roi: FaceBox | None
    capture_timestamp_ns: int
    sequence: int
    submitted_timestamp_ns: int


class LatestFrameRuntime:
    """Bounded single-slot worker that replaces stale pending frames."""

    def __init__(
        self,
        producer: NtpFrameProducer,
        *,
        roi_tracker: FaceRoiTracker | None = None,
        mode: RuntimeMode = RuntimeMode.PERFORMANCE,
        telemetry_window: int = 2048,
    ) -> None:
        if telemetry_window < 1:
            raise ValueError("telemetry window must be positive")
        self._producer = producer
        self._roi_tracker = roi_tracker
        self.mode = mode
        has_temporal = getattr(producer, "temporal_refiner", None) is not None
        if mode == RuntimeMode.QUALITY and not has_temporal:
            raise ValueError("Quality mode requires an active causal temporal refiner")
        descriptor = getattr(producer, "descriptor_event", {})
        guaranteed_profile = "Partial"
        if isinstance(descriptor, dict) and isinstance(descriptor.get("value"), dict):
            descriptor_value = cast(dict[str, object], descriptor["value"])
            guaranteed_profile = cast(str, descriptor_value.get("guaranteed_profile", "Partial"))
        self.capabilities = RuntimeCapabilities(
            mode=mode,
            guaranteed_profile=guaranteed_profile,
            latest_frame_only=True,
            temporal_refiner=has_temporal,
            stage_telemetry=True,
        )
        self._condition = threading.Condition()
        self._pending: _Submission | None = None
        self._latest: dict[str, object] | None = None
        self._closed = False
        self._error: BaseException | None = None
        self._sequence = 0
        self.dropped_frames = 0
        self._mailbox_wait_ns: deque[int] = deque(maxlen=telemetry_window)
        self._preprocess_ns: deque[int] = deque(maxlen=telemetry_window)
        self._inference_ns: deque[int] = deque(maxlen=telemetry_window)
        self._readback_ns: deque[int] = deque(maxlen=telemetry_window)
        self._producer_total_ns: deque[int] = deque(maxlen=telemetry_window)
        self._result_age_ns: deque[int] = deque(maxlen=telemetry_window)
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
            self._pending = _Submission(
                frame, roi, capture_timestamp_ns, self._sequence, time.monotonic_ns()
            )
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

    def telemetry_snapshot(self) -> RuntimeTelemetry:
        with self._condition:
            return RuntimeTelemetry(
                samples=len(self._producer_total_ns),
                submitted_frames=self._sequence,
                dropped_frames=self.dropped_frames,
                mailbox_wait_ms=_latency_summary(self._mailbox_wait_ns),
                preprocess_ms=_latency_summary(self._preprocess_ns),
                inference_ms=_latency_summary(self._inference_ns),
                readback_ms=_latency_summary(self._readback_ns),
                producer_total_ms=_latency_summary(self._producer_total_ns),
                result_age_ms=_latency_summary(self._result_age_ns),
            )

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
                producer_started = time.monotonic_ns()
                self._mailbox_wait_ns.append(producer_started - submission.submitted_timestamp_ns)
                roi = submission.roi
                if roi is None and self._roi_tracker is not None:
                    roi = self._roi_tracker.update(submission.frame, submission.sequence)
                result = self._producer.produce(
                    submission.frame,
                    roi=roi,
                    capture_timestamp_ns=submission.capture_timestamp_ns,
                    sequence=submission.sequence,
                )
                completed = time.monotonic_ns()
                self._producer_total_ns.append(completed - producer_started)
                self._result_age_ns.append(completed - submission.capture_timestamp_ns)
                stage_timings = getattr(self._producer, "last_stage_timings_ns", {})
                if isinstance(stage_timings, dict):
                    typed_timings = cast(dict[str, int], stage_timings)
                    self._preprocess_ns.append(typed_timings.get("preprocess", 0))
                    self._inference_ns.append(typed_timings.get("inference", 0))
                    self._readback_ns.append(typed_timings.get("readback", 0))
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
