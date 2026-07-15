from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from nana_tracking.config import ModelConfig, load_config
from nana_tracking.models import create_model, mirror_basic_rig
from nana_tracking.personalization import LevelACalibration, fit_level_a_calibration


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
