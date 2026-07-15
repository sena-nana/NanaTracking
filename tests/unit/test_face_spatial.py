from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from nana_tracking.config import ModelConfig, load_config
from nana_tracking.models import create_model, mirror_spatial_rig


def test_face_spatial_runs_all_heads_from_one_shared_encoder_pass() -> None:
    config = load_config(Path("configs/face-spatial-smoke.yaml"))
    model = create_model(config.model).eval()
    encoder_calls = 0

    def count_encoder_call(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = model.encoder.register_forward_hook(count_encoder_call)  # type: ignore[attr-defined]
    with torch.inference_mode():
        outputs = model(torch.zeros(2, 3, 64, 64))
    handle.remove()
    assert encoder_calls == 1
    assert [tuple(output.shape) for output in outputs] == [
        (2, 41),
        (2, 7),
        (2, 2, 3),
        (2, 2, 3),
        (2, 3),
        (2, 16, 3),
        (2, 3),
        (2, 2),
        (2, 2),
        (2, 41),
    ]
    torch.testing.assert_close(torch.linalg.vector_norm(outputs[1][:, 3:], dim=-1), torch.ones(2))
    torch.testing.assert_close(torch.linalg.vector_norm(outputs[3], dim=-1), torch.ones(2, 2))


def test_face_spatial_rejects_partial_profile_head() -> None:
    with pytest.raises(ValidationError, match="complete 41-signal SpatialSet"):
        ModelConfig(
            name="face_spatial",
            input_height=64,
            input_width=64,
            rig_dims=40,
            pose_dims=7,
        )


def test_spatial_mirror_swaps_eyes_and_negates_gaze_yaw() -> None:
    values = torch.arange(41, dtype=torch.float32).unsqueeze(0)
    mirrored = mirror_spatial_rig(values)
    assert mirrored[0, 36] == -values[0, 38]
    assert mirrored[0, 37] == values[0, 39]
    assert mirrored[0, 38] == -values[0, 36]
    assert mirrored[0, 39] == values[0, 37]
    assert mirrored[0, 40] == values[0, 40]
    torch.testing.assert_close(mirror_spatial_rig(mirrored), values)
