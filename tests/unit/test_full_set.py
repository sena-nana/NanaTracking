from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from nana_tracking.config import ModelConfig, load_config
from nana_tracking.models import create_model, mirror_full_rig


def test_full_set_runs_geometry_heads_from_one_encoder_pass() -> None:
    config = load_config(Path("configs/full-set-smoke.yaml"))
    model = create_model(config.model).eval()
    encoder_calls = 0

    def count_encoder_call(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = model.encoder.register_forward_hook(count_encoder_call)  # type: ignore[attr-defined]
    with torch.inference_mode():
        outputs = model(torch.zeros(2, 3, 96, 96))
    handle.remove()
    assert encoder_calls == 1
    assert [tuple(output.shape) for output in outputs] == [
        (2, 35),
        (2, 7),
        (2, 2, 3, 3),
        (2, 2, 3, 4),
        (2, 2, 2, 3),
        (2, 2, 2),
        (2, 2, 2),
        (2, 5, 6),
        (2, 2),
        (2, 35),
    ]
    torch.testing.assert_close(torch.linalg.vector_norm(outputs[1][:, 3:], dim=-1), torch.ones(2))
    torch.testing.assert_close(torch.linalg.vector_norm(outputs[3], dim=-1), torch.ones(2, 2, 3))
    torch.testing.assert_close(torch.linalg.vector_norm(outputs[4], dim=-1), torch.ones(2, 2, 2))


def test_full_set_rejects_partial_extension_head() -> None:
    with pytest.raises(ValidationError, match="complete 35-signal FullSet extension"):
        ModelConfig(
            name="full_set",
            input_height=96,
            input_width=96,
            rig_dims=34,
            pose_dims=7,
        )


def test_full_mirror_is_an_involution_and_swaps_arm_blocks() -> None:
    values = torch.arange(35, dtype=torch.float32).unsqueeze(0)
    mirrored = mirror_full_rig(values)
    assert mirrored[0, 25] == values[0, 30]
    assert mirrored[0, 30] == values[0, 25]
    assert mirrored[0, 27] == -values[0, 32]
    assert mirrored[0, 32] == -values[0, 27]
    torch.testing.assert_close(mirror_full_rig(mirrored), values)
