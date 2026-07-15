from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from nana_tracking.config import ModelConfig, load_config
from nana_tracking.evaluation import benchmark_rgb_roi_preprocessor
from nana_tracking.models import create_model, mirror_basic_rig
from nana_tracking.personalization import LevelACalibration, fit_level_a_calibration
from nana_tracking.runtime import FaceBox, FaceRoiTracker, RgbRoiWorkspace
from nana_tracking.runtime.face_basic import prepare_rgb_roi


class SequenceDetector:
    def __init__(self, detections: list[list[tuple[FaceBox, float]]]) -> None:
        self._detections = iter(detections)
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[tuple[FaceBox, float]]:
        del frame
        self.calls += 1
        return next(self._detections)


def test_face_basic_has_complete_single_pass_heads() -> None:
    config = load_config(Path("configs/face-basic-smoke.yaml"))
    model = create_model(config.model).eval()
    encoder_calls = 0

    def count_encoder_call(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = model.encoder.register_forward_hook(count_encoder_call)  # type: ignore[attr-defined]
    with torch.inference_mode():
        rig, pose, landmarks, visibility, identity, confidence = model(torch.zeros(2, 3, 64, 64))
    handle.remove()
    assert encoder_calls == 1
    assert rig.shape == (2, 36)
    assert pose.shape == (2, 7)
    assert landmarks.shape == (2, 16, 2)
    assert visibility.shape == (2, 3)
    assert identity.shape == (2, 2)
    assert confidence.shape == (2, 36)
    torch.testing.assert_close(torch.linalg.vector_norm(pose[:, 3:], dim=-1), torch.ones(2))


def test_face_basic_rejects_partial_basic_head() -> None:
    with pytest.raises(ValidationError, match="complete 36-signal BasicSet"):
        ModelConfig(
            name="face_basic",
            input_height=64,
            input_width=64,
            rig_dims=35,
            pose_dims=7,
        )


def test_anatomical_mirror_swaps_sides_and_negates_jaw_lateral() -> None:
    values = torch.arange(36, dtype=torch.float32).unsqueeze(0)
    mirrored = mirror_basic_rig(values)
    assert mirrored[0, 0] == values[0, 1]
    assert mirrored[0, 1] == values[0, 0]
    assert mirrored[0, 17] == -values[0, 17]
    torch.testing.assert_close(mirror_basic_rig(mirrored), values)


def test_level_a_calibration_is_complete_versioned_and_robust(tmp_path: Path) -> None:
    neutral = np.tile(np.linspace(-0.1, 0.1, 36, dtype=np.float32), (24, 1))
    movement = np.linspace(-1.0, 1.0, 48, dtype=np.float32)[:, None]
    ranges = neutral[0] + movement * np.ones((1, 36), dtype=np.float32)
    confidence = np.ones_like(ranges)
    profile = fit_level_a_calibration(
        neutral,
        ranges,
        confidence,
        user_slot="local-user-1",
        model_family="nana-face-basic",
        model_version="1.0.0",
        feature_revision="ntp-features/1.0.0",
        signal_registry_revision="ntp-signals/1.0.0",
        normalization_revision="ntp-normalization/1.0.0",
        calibration_revision="ntp-calibration/1.0.0",
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    restored = LevelACalibration.load_compatible(
        path,
        model_family="nana-face-basic",
        model_version="1.0.0",
        feature_revision="ntp-features/1.0.0",
        signal_registry_revision="ntp-signals/1.0.0",
    )
    calibrated_neutral = restored.apply(neutral[0])
    np.testing.assert_allclose(calibrated_neutral, np.zeros(36), atol=1e-6)
    assert [signal.signal_id for signal in restored.signals] == list(range(1, 37))


def test_roi_tracker_refreshes_at_a_bounded_interval_and_expires_missed_face() -> None:
    detector = SequenceDetector(
        [
            [(FaceBox(20, 20, 60, 60), 0.9)],
            [],
            [],
        ]
    )
    tracker = FaceRoiTracker(
        detector,
        detection_interval=2,
        smoothing=0.0,
        margin=0.0,
        maximum_missed=1,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert tracker.update(frame, 1) == FaceBox(20, 20, 60, 60)
    assert tracker.update(frame, 2) == FaceBox(20, 20, 60, 60)
    assert tracker.update(frame, 3) == FaceBox(20, 20, 60, 60)
    assert tracker.update(frame, 4) is None
    assert detector.calls == 3


def test_rgb_roi_workspace_preserves_sampling_with_bounded_reusable_storage() -> None:
    workspace = RgbRoiWorkspace(output_height=7, output_width=9)
    output = np.empty((1, 3, 7, 9), dtype=np.float32)
    random = np.random.default_rng(17)

    def expected(frame: np.ndarray, roi: FaceBox) -> np.ndarray:
        box = roi.clamp(frame.shape[1], frame.shape[0])
        ys = np.linspace(box.top, box.bottom - 1, 7).astype(np.intp)
        xs = np.linspace(box.left, box.right - 1, 9).astype(np.intp)
        return np.transpose(frame[ys[:, None], xs[None, :], :], (2, 0, 1))[None] / 255.0

    storage_bytes = workspace.workspace_bytes
    for _ in range(20):
        height = int(random.integers(10, 40))
        width = int(random.integers(10, 50))
        left = int(random.integers(-5, width - 1))
        top = int(random.integers(-5, height - 1))
        roi = FaceBox(
            left,
            top,
            int(random.integers(left + 1, width + 6)),
            int(random.integers(top + 1, height + 6)),
        )
        frame = random.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        prepare_rgb_roi(frame, roi, output, workspace=workspace)
        np.testing.assert_allclose(output, expected(frame, roi), rtol=0.0, atol=1e-7)
        assert workspace.workspace_bytes == storage_bytes


def test_rgb_roi_benchmark_records_bounded_workspace_evidence(tmp_path: Path) -> None:
    output = tmp_path / "roi-benchmark.json"
    report = benchmark_rgb_roi_preprocessor(
        output,
        input_width=64,
        input_height=48,
        roi_side=32,
        output_sizes=(8,),
        roi_positions=3,
        frames_per_roi=1,
        warmup=1,
        iterations=3,
    )
    result = cast(dict[str, object], cast(dict[str, object], report["results"])["8"])
    assert output.is_file()
    assert report["smoke_only"] is True
    assert cast(int, result["persistent_workspace_bytes"]) < 3 * 8 * 8 * 4
    assert cast(int, result["steady_tracemalloc_peak_bytes"]) < 64 * 1024
