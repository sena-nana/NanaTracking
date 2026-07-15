import json
import subprocess
import time
from pathlib import Path
from typing import cast

import numpy as np

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
