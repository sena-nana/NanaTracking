import json
import subprocess
from pathlib import Path
from typing import cast
from uuid import UUID

import numpy as np

from nana_tracking.runtime.face_basic import write_diagnostic_stream
from nana_tracking.runtime.face_spatial import FaceSpatialPrediction, FaceSpatialProducer
from nana_tracking.runtime.full_set import FullSetPrediction, FullSetProducer


class DeterministicSpatialBackend:
    input_height = 64
    input_width = 64

    def infer(self, image: np.ndarray) -> FaceSpatialPrediction:
        return FaceSpatialPrediction(
            rig=tuple(0.0 for _ in range(41)),
            pose=(0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0),
            eye_origins=((-0.15, 0.05, 0.0), (0.15, 0.05, 0.0)),
            eye_directions=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            look_at_head=(0.0, 0.0, 1.0),
            face_geometry=tuple((0.0, 0.0, 0.0) for _ in range(16)),
            visibility=0,
            tongue_visible=True,
            confidence=tuple(0.9 for _ in range(41)),
        )


class DeterministicFullBackend:
    input_height = 96
    input_width = 96

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, image: np.ndarray) -> FullSetPrediction:
        self.calls += 1
        return FullSetPrediction(
            rig=tuple(0.1 for _ in range(35)),
            torso_pose=(0.1, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0),
            joint_positions=(
                ((-0.2, -0.1, 0.0), (-0.4, 0.1, 0.0), (-0.5, 0.3, 0.0)),
                ((0.2, -0.1, 0.0), (0.4, 0.1, 0.0), (0.5, 0.3, 0.0)),
            ),
            joint_rotations=tuple(tuple((0.0, 0.0, 0.0, 1.0) for _ in range(3)) for _ in range(2)),  # type: ignore[arg-type]
            limb_directions=tuple(tuple((0.0, 1.0, 0.0) for _ in range(2)) for _ in range(2)),  # type: ignore[arg-type]
            limb_twists=((0.0, 0.0), (0.0, 0.0)),
            bone_lengths=((0.45, 0.4), (0.45, 0.4)),
            visibility=(0, 0, 0, 0, 0),
            confidence=tuple(0.85 for _ in range(35)),
        )


def _face_event(
    producer: FaceSpatialProducer, frame: np.ndarray, timestamp: int, sequence: int
) -> dict[str, object]:
    return producer.produce(frame, roi=None, capture_timestamp_ns=timestamp, sequence=sequence)


def test_full_producer_preserves_observation_age_and_passes_conformance(tmp_path: Path) -> None:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    session_id = UUID("12345678-1234-5678-1234-567812345678")
    face = FaceSpatialProducer(DeterministicSpatialBackend(), session_id=session_id)
    backend = DeterministicFullBackend()
    full = FullSetProducer(backend, body_inference_interval=2, clock=lambda: 2_000)
    first = full.produce(
        frame,
        face_event=_face_event(face, frame, 1_000, 1),
        body_roi=None,
        body_visible=True,
        capture_timestamp_ns=1_000,
        sequence=1,
    )
    second = full.produce(
        frame,
        face_event=_face_event(face, frame, 1_100, 2),
        body_roi=None,
        body_visible=True,
        capture_timestamp_ns=1_100,
        sequence=2,
    )
    third = full.produce(
        frame,
        face_event=_face_event(face, frame, 1_200, 3),
        body_roi=None,
        body_visible=True,
        capture_timestamp_ns=1_200,
        sequence=3,
    )
    assert backend.calls == 2
    second_value = cast(dict[str, object], second["value"])
    second_slots = cast(
        list[dict[str, object]], cast(dict[str, object], second_value["rig"])["slots"]
    )
    assert second_slots[41]["state"] == "Fused"
    assert second_slots[41]["sample_capture_timestamp_ns"] == 1_000
    third_value = cast(dict[str, object], third["value"])
    slots = cast(list[dict[str, object]], cast(dict[str, object], third_value["rig"])["slots"])
    assert slots[41]["state"] == "Observed"
    assert slots[41]["sample_capture_timestamp_ns"] == 1_200
    np.testing.assert_allclose(
        cast(list[float], [slots[index]["value"] for index in range(47, 50)]),
        [0.1, 0.1, 0.2],
        atol=1e-6,
    )

    stream = tmp_path / "full.jsonl"
    write_diagnostic_stream(stream, full.descriptor_event, [first, second, third])
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
    assert report["certified_profile"] == "Full"


def test_face_only_capture_remains_full_and_reports_body_out_of_frame() -> None:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    face = FaceSpatialProducer(DeterministicSpatialBackend())
    full = FullSetProducer(DeterministicFullBackend())
    event = full.produce(
        frame,
        face_event=_face_event(face, frame, 1_000, 1),
        body_roi=None,
        body_visible=False,
        capture_timestamp_ns=1_000,
        sequence=1,
    )
    value = cast(dict[str, object], event["value"])
    slots = cast(list[dict[str, object]], cast(dict[str, object], value["rig"])["slots"])
    assert all(slot["state"] != "Unsupported" for slot in slots[:76])
    assert all(slot["state"] == "OutOfFrame" and slot["value"] is None for slot in slots[41:76])
    skeleton = cast(dict[str, object], value["skeleton"])
    assert cast(dict[str, object], skeleton["wrist"])["left"]["state"] == "OutOfFrame"  # type: ignore[index]


def test_body_reappearance_forces_a_fresh_observation() -> None:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    face = FaceSpatialProducer(DeterministicSpatialBackend())
    backend = DeterministicFullBackend()
    full = FullSetProducer(backend, body_inference_interval=10)
    for sequence, visible in ((1, True), (2, False), (3, True)):
        full.produce(
            frame,
            face_event=_face_event(face, frame, 1_000 + sequence, sequence),
            body_roi=None,
            body_visible=visible,
            capture_timestamp_ns=1_000 + sequence,
            sequence=sequence,
        )
    assert backend.calls == 2


def test_body_cadence_is_relative_and_stream_reconfigure_invalidates_cache() -> None:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    session_id = UUID("12345678-1234-5678-1234-567812345678")
    face = FaceSpatialProducer(DeterministicSpatialBackend(), session_id=session_id)
    backend = DeterministicFullBackend()
    full = FullSetProducer(backend, body_inference_interval=2)
    for sequence in (5, 6, 7):
        full.produce(
            frame,
            face_event=_face_event(face, frame, 1_000 + sequence, sequence),
            body_roi=None,
            body_visible=True,
            capture_timestamp_ns=1_000 + sequence,
            sequence=sequence,
        )
    assert backend.calls == 2

    reconfigured = FaceSpatialProducer(
        DeterministicSpatialBackend(), session_id=session_id, generation=1
    )
    full.produce(
        frame,
        face_event=_face_event(reconfigured, frame, 1_008, 8),
        body_roi=None,
        body_visible=True,
        capture_timestamp_ns=1_008,
        sequence=8,
    )
    assert backend.calls == 3


def test_stale_body_sample_requests_refresh_instead_of_remaining_lost() -> None:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    face = FaceSpatialProducer(DeterministicSpatialBackend())
    backend = DeterministicFullBackend()
    full = FullSetProducer(
        backend,
        body_inference_interval=100,
        maximum_body_age_ns=50,
    )
    first = full.produce(
        frame,
        face_event=_face_event(face, frame, 1_000, 1),
        body_roi=None,
        body_visible=True,
        capture_timestamp_ns=1_000,
        sequence=1,
    )
    stale = full.produce(
        frame,
        face_event=_face_event(face, frame, 1_100, 2),
        body_roi=None,
        body_visible=True,
        capture_timestamp_ns=1_100,
        sequence=2,
    )
    refreshed = full.produce(
        frame,
        face_event=_face_event(face, frame, 1_101, 3),
        body_roi=None,
        body_visible=True,
        capture_timestamp_ns=1_101,
        sequence=3,
    )
    assert backend.calls == 2
    for event, expected in ((first, "Observed"), (stale, "TrackingLost"), (refreshed, "Observed")):
        value = cast(dict[str, object], event["value"])
        slots = cast(list[dict[str, object]], cast(dict[str, object], value["rig"])["slots"])
        assert slots[41]["state"] == expected
