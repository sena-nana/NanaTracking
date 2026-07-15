import json
import subprocess
import time
from pathlib import Path
from typing import cast

import numpy as np

from nana_tracking.runtime import LatestFrameRuntime
from nana_tracking.runtime.face_basic import write_diagnostic_stream
from nana_tracking.runtime.face_spatial import FaceSpatialPrediction, FaceSpatialProducer


class DeterministicSpatialBackend:
    input_height = 64
    input_width = 64

    def __init__(self, *, tongue_visible: bool = True) -> None:
        self.tongue_visible = tongue_visible

    def infer(self, image: np.ndarray) -> FaceSpatialPrediction:
        assert image.shape == (1, 3, 64, 64)
        return FaceSpatialPrediction(
            rig=tuple(0.1 if slot == 40 else 0.0 for slot in range(41)),
            pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            eye_origins=((-0.15, 0.05, 0.0), (0.15, 0.05, 0.0)),
            eye_directions=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            look_at_head=(0.0, 0.0, 1.0),
            face_geometry=tuple((0.0, 0.0, 0.0) for _ in range(16)),
            visibility=0,
            tongue_visible=self.tongue_visible,
            confidence=tuple(0.9 for _ in range(41)),
        )


def test_spatial_producer_is_bounded_and_passes_conformance(tmp_path: Path) -> None:
    producer = FaceSpatialProducer(DeterministicSpatialBackend())
    runtime = LatestFrameRuntime(producer)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    results: list[dict[str, object]] = []
    for _ in range(6):
        runtime.submit(frame, capture_timestamp_ns=time.monotonic_ns())
    latest = runtime.wait_latest(timeout=2.0)
    assert latest is not None
    results.append(latest)
    value = cast(dict[str, object], latest["value"])
    rig = cast(dict[str, object], value["rig"])
    slots = cast(list[dict[str, object]], rig["slots"])
    assert all(sample["state"] != "Unsupported" for sample in slots[:41])
    assert all(sample["state"] == "Unsupported" for sample in slots[41:])
    geometry = cast(dict[str, object], value["geometry"])
    assert geometry["face_geometry_state"] == "Observed"
    assert geometry["face_landmarks"] == []
    assert runtime.dropped_frames > 0
    runtime.close()

    stream = tmp_path / "face-spatial.jsonl"
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
    assert report["certified_profile"] == "Spatial"


def test_invisible_tongue_is_occluded_without_a_fake_zero() -> None:
    producer = FaceSpatialProducer(DeterministicSpatialBackend(tongue_visible=False))
    event = producer.produce(
        np.zeros((64, 64, 3), dtype=np.uint8),
        roi=None,
        capture_timestamp_ns=time.monotonic_ns(),
        sequence=1,
    )
    value = cast(dict[str, object], event["value"])
    rig = cast(dict[str, object], value["rig"])
    slots = cast(list[dict[str, object]], rig["slots"])
    assert slots[40]["state"] == "Occluded"
    assert slots[40]["value"] is None
