import json
import subprocess
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from nana_tracking.runtime import CausalTemporalRefiner, RuntimeMode
from nana_tracking.runtime.face_basic import (
    FaceBasicPrediction,
    FaceBasicProducer,
    LatestFrameRuntime,
    write_diagnostic_stream,
)


class DeterministicBackend:
    input_height = 64
    input_width = 64

    def infer(self, image: np.ndarray) -> FaceBasicPrediction:
        assert image.shape == (1, 3, 64, 64)
        time.sleep(0.005)
        return FaceBasicPrediction(
            rig=tuple(0.0 for _ in range(36)),
            pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            visibility=0,
            confidence=tuple(0.9 for _ in range(36)),
        )


class MutableBackend:
    input_height = 64
    input_width = 64

    def __init__(self) -> None:
        self.value = 0.0
        self.visibility = 0

    def infer(self, image: np.ndarray) -> FaceBasicPrediction:
        return FaceBasicPrediction(
            rig=tuple(self.value for _ in range(36)),
            pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            visibility=self.visibility,
            confidence=tuple(0.9 for _ in range(36)),
        )


class OffsetAdapter:
    def apply(self, base_values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(value + 0.1 for value in base_values)


def test_latest_frame_runtime_stays_bounded_and_passes_ntp_conformance(tmp_path: Path) -> None:
    producer = FaceBasicProducer(DeterministicBackend())
    runtime = LatestFrameRuntime(producer)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(12):
        runtime.submit(frame, capture_timestamp_ns=time.monotonic_ns())
    result = runtime.wait_latest(timeout=2.0)
    assert result is not None
    assert runtime.dropped_frames > 0
    value = result["value"]
    assert isinstance(value, dict)
    assert value["sequence"] == 12
    assert value["produced_timestamp_ns"] >= value["capture_timestamp_ns"]
    rig = cast(dict[str, object], value["rig"])
    slots = cast(list[dict[str, object]], rig["slots"])
    assert all(sample["state"] != "Unsupported" for sample in slots[:36])
    assert all(sample["state"] == "Unsupported" for sample in slots[36:])

    results = [result]
    ages_ms: list[float] = []
    for _ in range(10):
        captured = time.monotonic_ns()
        runtime.submit(frame, capture_timestamp_ns=captured)
        continuous = runtime.wait_latest(timeout=2.0)
        assert continuous is not None
        continuous_value = continuous["value"]
        assert isinstance(continuous_value, dict)
        continuous_capture = continuous_value["capture_timestamp_ns"]
        assert isinstance(continuous_capture, int)
        ages_ms.append((time.monotonic_ns() - continuous_capture) / 1e6)
        results.append(continuous)
    runtime.close()
    assert max(ages_ms) < 100.0
    assert ages_ms[-1] < ages_ms[0] + 25.0
    telemetry = runtime.telemetry_snapshot()
    assert telemetry.samples == len(results)
    assert telemetry.dropped_frames > 0
    assert telemetry.preprocess_ms["p95"] >= 0.0
    assert telemetry.inference_ms["p95"] > 0.0
    assert telemetry.result_age_ms["p99"] < 100.0

    stream = tmp_path / "face-basic.jsonl"
    write_diagnostic_stream(stream, producer.descriptor_event, results)
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "ntp-conformance",
            "--",
            "--input-format",
            "jsonl",
            "--output",
            "json",
            str(stream),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["certified_profile"] == "Basic"


def test_face_basic_temporal_prediction_is_complete_and_conformant(tmp_path: Path) -> None:
    backend = MutableBackend()
    producer = FaceBasicProducer(backend, temporal_refiner=CausalTemporalRefiner(range(1, 37)))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    events: list[dict[str, object]] = []
    for sequence, value in ((1, 0.0), (2, 0.2)):
        backend.value = value
        events.append(
            producer.produce(
                frame,
                roi=None,
                capture_timestamp_ns=sequence * 10_000_000,
                sequence=sequence,
            )
        )
    backend.visibility = 1
    events.append(
        producer.produce(
            frame,
            roi=None,
            capture_timestamp_ns=35_000_000,
            sequence=3,
        )
    )
    predicted = cast(dict[str, object], events[-1]["value"])
    slots = cast(list[dict[str, object]], cast(dict[str, object], predicted["rig"])["slots"])
    assert all(slot["state"] == "Predicted" for slot in slots[:36])
    assert all(slot["prediction_horizon_ns"] == 15_000_000 for slot in slots[:36])
    stream = tmp_path / "face-basic-temporal.jsonl"
    write_diagnostic_stream(stream, producer.descriptor_event, events)
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "ntp-conformance",
            "--",
            "--input-format",
            "jsonl",
            "--output",
            "json",
            str(stream),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_runtime_modes_report_actual_capability() -> None:
    with pytest.raises(ValueError, match="Quality mode"):
        LatestFrameRuntime(FaceBasicProducer(MutableBackend()), mode=RuntimeMode.QUALITY)
    runtime = LatestFrameRuntime(
        FaceBasicProducer(MutableBackend(), temporal_refiner=CausalTemporalRefiner(range(1, 37))),
        mode=RuntimeMode.QUALITY,
    )
    assert runtime.capabilities.mode == RuntimeMode.QUALITY
    assert runtime.capabilities.temporal_refiner is True
    assert runtime.capabilities.guaranteed_profile == "Basic"
    runtime.close()


def test_face_basic_applies_level_b_before_protocol_clamping() -> None:
    producer = FaceBasicProducer(MutableBackend(), level_b_adapter=OffsetAdapter())
    event = producer.produce(
        np.zeros((64, 64, 3), dtype=np.uint8),
        roi=None,
        capture_timestamp_ns=1,
        sequence=1,
    )
    value = cast(dict[str, object], event["value"])
    slots = cast(list[dict[str, object]], cast(dict[str, object], value["rig"])["slots"])
    assert all(slot["value"] == pytest.approx(0.1) for slot in slots[:36])
