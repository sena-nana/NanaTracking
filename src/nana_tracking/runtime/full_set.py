"""Low-cadence FullSet ONNX producer fused with a display-cadence Spatial result."""

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, cast

import numpy as np
import onnxruntime as ort

from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.export import verify_model_package
from nana_tracking.runtime.face_basic import FaceBox, prepare_rgb_roi, region


@dataclass(frozen=True, slots=True)
class FullSetPrediction:
    rig: tuple[float, ...]
    torso_pose: tuple[float, ...]
    joint_positions: tuple[tuple[tuple[float, float, float], ...], ...]
    joint_rotations: tuple[tuple[tuple[float, float, float, float], ...], ...]
    limb_directions: tuple[tuple[tuple[float, float, float], ...], ...]
    limb_twists: tuple[tuple[float, float], tuple[float, float]]
    bone_lengths: tuple[tuple[float, float], tuple[float, float]]
    # torso, left arm, right arm, left auricle, right auricle.
    visibility: tuple[int, int, int, int, int]
    confidence: tuple[float, ...]


class FullSetBackend(Protocol):
    input_height: int
    input_width: int

    def infer(self, image: np.ndarray) -> FullSetPrediction: ...


class OrtFullSetBackend:
    """Verified ORT adapter for a Full-only package."""

    _OUTPUTS: ClassVar[tuple[str, ...]] = (
        "rig",
        "torso_pose",
        "joint_positions",
        "joint_rotations",
        "limb_directions",
        "limb_twists",
        "bone_lengths",
        "visibility",
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
        if self.metadata.supported_signals != list(range(42, 77)):
            raise ValueError("model package does not declare all Full-only signals")
        if self.metadata.supported_structures != ["body_skeleton"]:
            raise ValueError("model package does not declare the body skeleton")
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

    def infer(self, image: np.ndarray) -> FullSetPrediction:
        if image.shape != (1, 3, self.input_height, self.input_width):
            raise ValueError("backend input does not match the fixed upper-body ROI shape")
        outputs = cast(list[np.ndarray], self._session.run(self._OUTPUTS, {"image": image}))
        rig, torso, joints, rotations, directions, twists, lengths, visibility, confidence = outputs
        return FullSetPrediction(
            rig=tuple(float(value) for value in rig[0]),
            torso_pose=tuple(float(value) for value in torso[0]),
            joint_positions=cast(
                tuple[tuple[tuple[float, float, float], ...], ...],
                tuple(
                    tuple(tuple(float(axis) for axis in joint) for joint in side)
                    for side in joints[0]
                ),
            ),
            joint_rotations=cast(
                tuple[tuple[tuple[float, float, float, float], ...], ...],
                tuple(
                    tuple(tuple(float(axis) for axis in joint) for joint in side)
                    for side in rotations[0]
                ),
            ),
            limb_directions=cast(
                tuple[tuple[tuple[float, float, float], ...], ...],
                tuple(
                    tuple(tuple(float(axis) for axis in limb) for limb in side)
                    for side in directions[0]
                ),
            ),
            limb_twists=cast(
                tuple[tuple[float, float], tuple[float, float]],
                tuple(tuple(float(value) for value in side) for side in twists[0]),
            ),
            bone_lengths=cast(
                tuple[tuple[float, float], tuple[float, float]],
                tuple(tuple(float(value) for value in side) for side in lengths[0]),
            ),
            visibility=cast(
                tuple[int, int, int, int, int], tuple(int(np.argmax(row)) for row in visibility[0])
            ),
            confidence=tuple(float(value) for value in confidence[0]),
        )


def _tracked(
    value: object | None,
    confidence: float,
    state: str,
    timestamp_ns: int,
) -> dict[str, object]:
    return {
        "value": value,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "state": state,
        "sample_capture_timestamp_ns": timestamp_ns,
        "prediction_horizon_ns": 0,
    }


def _vector(value: np.ndarray) -> dict[str, float]:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def _quaternion(value: np.ndarray) -> np.ndarray:
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(value))
    result = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) if norm < 1e-6 else value / norm
    return -result if result[3] < 0.0 else result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _quaternion(
        np.array(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float32,
        )
    )


def _rotate(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    xyz = quaternion[:3]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + quaternion[3] * vector)


def _euler_pitch_yaw_roll(quaternion: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = (float(value) for value in quaternion)
    pitch = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    yaw = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    roll = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return pitch, yaw, roll


def _pose(position: np.ndarray, quaternion: np.ndarray, parent: str) -> dict[str, object]:
    return {
        "parent_space": parent,
        "length_basis": "TorsoRelative",
        "position": _vector(position),
        "orientation_xyzw": {
            "x": float(quaternion[0]),
            "y": float(quaternion[1]),
            "z": float(quaternion[2]),
            "w": float(quaternion[3]),
        },
    }


_WIRE_STATE = {
    0: "Observed",
    1: "Occluded",  # Internal PartiallyOccluded remains distinguishable in model output.
    2: "Occluded",  # Internal SelfOccluded remains distinguishable in model output.
    3: "OutOfFrame",
    4: "Predicted",
    5: "TrackingLost",
}


@dataclass(slots=True)
class _BodySample:
    prediction: FullSetPrediction
    capture_timestamp_ns: int


class FullSetProducer:
    """Fuse low-rate body observations into current Spatial results without hiding sample age."""

    def __init__(
        self,
        backend: FullSetBackend,
        *,
        body_inference_interval: int = 2,
        maximum_body_age_ns: int = 150_000_000,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if body_inference_interval < 1 or maximum_body_age_ns < 0:
            raise ValueError("body cadence and maximum age must be bounded")
        self.backend = backend
        self.body_inference_interval = body_inference_interval
        self.maximum_body_age_ns = maximum_body_age_ns
        self._clock = clock
        self._input = np.empty((1, 3, backend.input_height, backend.input_width), dtype=np.float32)
        self._latest: _BodySample | None = None
        self._needs_refresh = True
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
                "guaranteed_profile": "Full",
                "supported_signals": list(range(1, 77)),
                "supported_structures": 31,
                "features": 0,
            },
        }

    def produce(
        self,
        frame: np.ndarray,
        *,
        face_event: dict[str, object],
        body_roi: FaceBox | None,
        body_visible: bool,
        capture_timestamp_ns: int,
        sequence: int,
    ) -> dict[str, object]:
        result = self._validated_face_result(face_event, capture_timestamp_ns, sequence)
        preprocess_ns = 0
        inference_ns = 0
        if body_visible and (
            self._needs_refresh
            or self._latest is None
            or sequence % self.body_inference_interval == 0
        ):
            preprocess_started = time.perf_counter_ns()
            prepare_rgb_roi(frame, body_roi, self._input)
            preprocess_ns = time.perf_counter_ns() - preprocess_started
            inference_started = time.perf_counter_ns()
            self._latest = _BodySample(self.backend.infer(self._input), capture_timestamp_ns)
            inference_ns = time.perf_counter_ns() - inference_started
            self._needs_refresh = False
        elif not body_visible:
            self._needs_refresh = True
        readback_started = time.perf_counter_ns()
        sample = self._latest if body_visible else None
        if sample is None:
            self._write_unavailable(result, "OutOfFrame", capture_timestamp_ns)
        elif capture_timestamp_ns - sample.capture_timestamp_ns > self.maximum_body_age_ns:
            self._write_unavailable(result, "TrackingLost", sample.capture_timestamp_ns)
        else:
            self._write_prediction(result, sample, capture_timestamp_ns)
        result["produced_timestamp_ns"] = max(int(self._clock()), capture_timestamp_ns)
        self.last_stage_timings_ns = {
            "preprocess": preprocess_ns,
            "inference": inference_ns,
            "readback": time.perf_counter_ns() - readback_started,
        }
        return {"kind": "result", "value": result}

    @staticmethod
    def _validated_face_result(
        face_event: dict[str, object], capture_timestamp_ns: int, sequence: int
    ) -> dict[str, object]:
        if face_event.get("kind") != "result" or not isinstance(face_event.get("value"), dict):
            raise ValueError("FullSet fusion requires a Spatial result event")
        result = copy.deepcopy(cast(dict[str, object], face_event["value"]))
        if result.get("capture_timestamp_ns") != capture_timestamp_ns:
            raise ValueError("face and FullSet capture timestamps must match")
        if result.get("sequence") != sequence:
            raise ValueError("face and FullSet sequences must match")
        rig = cast(dict[str, object], result.get("rig"))
        slots = cast(list[dict[str, object]], rig.get("slots"))
        if len(slots) != 88 or any(slot["state"] == "Unsupported" for slot in slots[:41]):
            raise ValueError("FullSet fusion requires complete Spatial signal support")
        return result

    @staticmethod
    def _write_unavailable(result: dict[str, object], state: str, timestamp_ns: int) -> None:
        slots = cast(list[dict[str, object]], cast(dict[str, object], result["rig"])["slots"])
        for index in range(41, 76):
            slots[index] = _tracked(None, 0.0, state, timestamp_ns)
        unavailable = _tracked(None, 0.0, state, timestamp_ns)
        result["skeleton"] = {
            "torso_camera_pose": unavailable,
            "shoulder": {"left": unavailable, "right": unavailable},
            "elbow": {"left": unavailable, "right": unavailable},
            "wrist": {"left": unavailable, "right": unavailable},
            "upper_arm_direction_torso": {"left": unavailable, "right": unavailable},
            "forearm_direction_torso": {"left": unavailable, "right": unavailable},
            "upper_arm_twist": {"left": unavailable, "right": unavailable},
            "forearm_twist": {"left": unavailable, "right": unavailable},
        }
        quality = cast(dict[str, object], result["quality"])
        quality["torso"] = region(0.0, state)
        quality["arm"] = {"left": region(0.0, state), "right": region(0.0, state)}
        quality["auricle"] = {"left": region(0.0, state), "right": region(0.0, state)}

    def _write_prediction(
        self, result: dict[str, object], sample: _BodySample, capture_timestamp_ns: int
    ) -> None:
        prediction = sample.prediction
        values = np.asarray(prediction.rig, dtype=np.float32)
        confidence = np.asarray(prediction.confidence, dtype=np.float32)
        torso = np.asarray(prediction.torso_pose, dtype=np.float32)
        joints = np.asarray(prediction.joint_positions, dtype=np.float32)
        rotations = np.asarray(prediction.joint_rotations, dtype=np.float32)
        directions = np.asarray(prediction.limb_directions, dtype=np.float32)
        twists = np.asarray(prediction.limb_twists, dtype=np.float32)
        lengths = np.asarray(prediction.bone_lengths, dtype=np.float32)
        if values.shape != (35,) or confidence.shape != (35,):
            raise ValueError("backend must return all 35 Full-only values and confidences")
        if torso.shape != (7,) or joints.shape != (2, 3, 3) or rotations.shape != (2, 3, 4):
            raise ValueError("backend torso or joint geometry shape is invalid")
        if directions.shape != (2, 2, 3) or twists.shape != (2, 2) or lengths.shape != (2, 2):
            raise ValueError("backend limb geometry shape is invalid")
        arrays = (values, confidence, torso, joints, rotations, directions, twists, lengths)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("backend FullSet output must be finite")
        if len(prediction.visibility) != 5 or any(
            value not in _WIRE_STATE for value in prediction.visibility
        ):
            raise ValueError("backend visibility must use the six FullSet internal states")
        confidence = np.clip(confidence, 0.0, 1.0)
        values = np.clip(values, -1.0, 1.0)
        values[[28, 33]] = np.clip(values[[28, 33]], 0.0, 1.0)
        torso_quaternion = _quaternion(torso[3:])
        torso_angles = _euler_pitch_yaw_roll(torso_quaternion)
        values[:6] = np.asarray((*np.clip(torso[:3], -1.0, 1.0), *torso_angles), dtype=np.float32)
        # NTP skeleton geometry is authoritative. Derive directions and the corresponding Rig
        # scalars from the same joints so the two representations cannot drift independently.
        for side in range(2):
            upper_delta = joints[side, 1] - joints[side, 0]
            forearm_delta = joints[side, 2] - joints[side, 1]
            upper_norm = math.sqrt(sum(float(upper_delta[axis]) ** 2 for axis in range(3)))
            forearm_norm = math.sqrt(sum(float(forearm_delta[axis]) ** 2 for axis in range(3)))
            if upper_norm < 1e-6 or forearm_norm < 1e-6:
                raise ValueError("backend joints must define non-zero upper-arm and forearm bones")
            directions[side, 0] = upper_delta / upper_norm
            directions[side, 1] = forearm_delta / forearm_norm
            start = 25 if side == 0 else 30
            values[start] = directions[side, 0, 2]
            values[start + 1] = directions[side, 0, 0] * (-1.0 if side == 0 else 1.0)
            values[start + 2] = twists[side, 0]
            values[start + 3] = np.clip(
                (1.0 - float(np.dot(directions[side, 0], directions[side, 1]))) * 0.5,
                0.0,
                1.0,
            )
            values[start + 4] = twists[side, 1]
        states = [_WIRE_STATE[value] for value in prediction.visibility]
        if sample.capture_timestamp_ns != capture_timestamp_ns:
            states = ["Fused" if state == "Observed" else state for state in states]
        slots = cast(list[dict[str, object]], cast(dict[str, object], result["rig"])["slots"])
        face_quality = cast(dict[str, object], cast(dict[str, object], result["quality"])["face"])
        face_state = cast(str, face_quality["state"])
        face_confidence = float(cast(float, face_quality["confidence"]))
        signal_states = [states[0]] * 12 + [face_state] * 3 + [states[3], states[4]] * 3
        signal_states += [states[1], states[2], states[1], states[2]]
        signal_states += [states[1]] * 5 + [states[2]] * 5
        self._write_head_relative(
            result, values, confidence, torso, torso_quaternion, states[0], sample
        )
        for offset, state in enumerate(signal_states):
            if 6 <= offset <= 11:
                continue
            carries = state in {"Observed", "Fused", "Predicted"}
            slots[41 + offset] = _tracked(
                float(values[offset]) if carries else None,
                min(float(confidence[offset]), face_confidence)
                if 12 <= offset <= 14
                else float(confidence[offset]),
                state,
                sample.capture_timestamp_ns,
            )
        self._write_skeleton(
            result,
            torso,
            torso_quaternion,
            joints,
            rotations,
            directions,
            twists,
            confidence,
            states,
            sample.capture_timestamp_ns,
        )
        quality = cast(dict[str, object], result["quality"])
        quality["torso"] = region(float(confidence[:12].mean()), states[0])
        quality["arm"] = {
            "left": region(float(confidence[21:30].mean()), states[1]),
            "right": region(float(confidence[[22, 24, 30, 31, 32, 33, 34]].mean()), states[2]),
        }
        quality["auricle"] = {
            "left": region(float(confidence[[15, 17, 19]].mean()), states[3]),
            "right": region(float(confidence[[16, 18, 20]].mean()), states[4]),
        }
        prior = float(cast(float, quality["overall_confidence"]))
        quality["overall_confidence"] = min(prior, float(confidence.mean()))

    @staticmethod
    def _write_head_relative(
        result: dict[str, object],
        values: np.ndarray,
        confidence: np.ndarray,
        torso: np.ndarray,
        torso_quaternion: np.ndarray,
        torso_state: str,
        sample: _BodySample,
    ) -> None:
        slots = cast(list[dict[str, object]], cast(dict[str, object], result["rig"])["slots"])
        geometry = cast(dict[str, object], result["geometry"])
        head = cast(dict[str, object], geometry["head_camera_pose"])
        head_state = cast(str, head["state"])
        head_value = head["value"]
        if torso_state not in {"Observed", "Fused", "Predicted"} or not isinstance(
            head_value, dict
        ):
            state = (
                torso_state if torso_state not in {"Observed", "Fused", "Predicted"} else head_state
            )
            for index in range(6, 12):
                slots[41 + index] = _tracked(None, 0.0, state, sample.capture_timestamp_ns)
            return
        head_position_map = cast(dict[str, float], head_value["position"])
        head_quaternion_map = cast(dict[str, float], head_value["orientation_xyzw"])
        head_position = np.array([head_position_map[axis] for axis in "xyz"], dtype=np.float32)
        head_quaternion = _quaternion(
            np.array([head_quaternion_map[axis] for axis in "xyzw"], dtype=np.float32)
        )
        inverse_torso = torso_quaternion.copy()
        inverse_torso[:3] *= -1.0
        relative_position = _rotate(head_position - torso[:3], inverse_torso)
        relative_quaternion = _quaternion_multiply(inverse_torso, head_quaternion)
        relative = (
            *np.clip(relative_position, -1.0, 1.0),
            *_euler_pitch_yaw_roll(relative_quaternion),
        )
        values[6:12] = np.asarray(relative, dtype=np.float32)
        state = "Fused"
        timestamp_ns = min(
            sample.capture_timestamp_ns, cast(int, head["sample_capture_timestamp_ns"])
        )
        combined_confidence = min(float(confidence[6:12].mean()), cast(float, head["confidence"]))
        for offset, value in enumerate(relative):
            slots[47 + offset] = _tracked(float(value), combined_confidence, state, timestamp_ns)

    @staticmethod
    def _write_skeleton(
        result: dict[str, object],
        torso: np.ndarray,
        torso_quaternion: np.ndarray,
        joints: np.ndarray,
        rotations: np.ndarray,
        directions: np.ndarray,
        twists: np.ndarray,
        confidence: np.ndarray,
        states: list[str],
        timestamp_ns: int,
    ) -> None:
        def carries(state: str) -> bool:
            return state in {"Observed", "Fused", "Predicted"}

        torso_value = _pose(torso[:3], torso_quaternion, "Camera") if carries(states[0]) else None
        skeleton: dict[str, object] = {
            "torso_camera_pose": _tracked(
                torso_value, float(confidence[:6].mean()), states[0], timestamp_ns
            )
        }
        for joint_index, joint_name in enumerate(("shoulder", "elbow", "wrist")):
            side_map: dict[str, object] = {}
            for side_index, side_name in enumerate(("left", "right")):
                state = states[side_index + 1]
                value = (
                    _pose(
                        joints[side_index, joint_index],
                        _quaternion(rotations[side_index, joint_index]),
                        "TorsoLocal",
                    )
                    if carries(state)
                    else None
                )
                side_map[side_name] = _tracked(value, float(confidence.mean()), state, timestamp_ns)
            skeleton[joint_name] = side_map
        for limb_index, name in enumerate(("upper_arm_direction_torso", "forearm_direction_torso")):
            side_map = {}
            for side_index, side_name in enumerate(("left", "right")):
                state = states[side_index + 1]
                vector = directions[side_index, limb_index]
                value = (
                    {"space": "TorsoLocal", "value": _vector(vector)} if carries(state) else None
                )
                side_map[side_name] = _tracked(value, float(confidence.mean()), state, timestamp_ns)
            skeleton[name] = side_map
        for limb_index, name in enumerate(("upper_arm_twist", "forearm_twist")):
            side_map = {}
            for side_index, side_name in enumerate(("left", "right")):
                state = states[side_index + 1]
                value = float(twists[side_index, limb_index] * math.pi) if carries(state) else None
                side_map[side_name] = _tracked(value, float(confidence.mean()), state, timestamp_ns)
            skeleton[name] = side_map
        result["skeleton"] = skeleton
