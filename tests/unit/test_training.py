from pathlib import Path

import pytest
import torch

from nana_tracking.config import load_config
from nana_tracking.data.loaders import create_loader
from nana_tracking.models import create_model
from nana_tracking.training import train
from nana_tracking.training.checkpoint import load_checkpoint


def test_checkpoint_resume_continues_step(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    direct_config = config.model_copy(
        update={
            "reproducibility": config.reproducibility.model_copy(
                update={"output_dir": tmp_path / "direct"}
            ),
            "training": config.training.model_copy(update={"max_steps": 2}),
        }
    )
    direct = train(direct_config)
    staged_config = config.model_copy(
        update={
            "reproducibility": config.reproducibility.model_copy(
                update={"output_dir": tmp_path / "resumed"}
            ),
            "training": config.training.model_copy(update={"max_steps": 1}),
        }
    )
    first = train(staged_config)
    resumed_config = staged_config.model_copy(
        update={"training": staged_config.training.model_copy(update={"max_steps": 2})}
    )
    resumed = train(resumed_config, resume=first.checkpoint)
    assert resumed.final_step == 2
    assert resumed.run_dir == first.run_dir
    direct_model = create_model(config.model)
    resumed_model = create_model(config.model)
    load_checkpoint(direct.checkpoint, model=direct_model)
    metadata = load_checkpoint(resumed.checkpoint, model=resumed_model)
    for name, direct_value in direct_model.state_dict().items():
        assert torch.equal(direct_value, resumed_model.state_dict()[name])
    assert (direct.run_dir / "metrics.jsonl").read_bytes() == (
        resumed.run_dir / "metrics.jsonl"
    ).read_bytes()
    assert metadata.step == 2
    assert metadata.data_revision == "synthetic-v1"
    assert metadata.device == "cpu"
    assert metadata.amp_enabled is False


def test_resume_rejects_training_semantic_drift(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke.yaml")).model_copy(
        update={
            "reproducibility": load_config(Path("configs/smoke.yaml")).reproducibility.model_copy(
                update={"output_dir": tmp_path}
            )
        }
    )
    first = train(
        config.model_copy(update={"training": config.training.model_copy(update={"max_steps": 1})})
    )
    incompatible = config.model_copy(
        update={
            "training": config.training.model_copy(update={"max_steps": 2, "learning_rate": 0.123})
        }
    )
    with pytest.raises(ValueError, match="incompatible"):
        train(incompatible, resume=first.checkpoint)


def test_epoch_shuffle_does_not_change_synthetic_samples() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    first = list(
        create_loader(
            config,
            split="train",
            shuffle=True,
            seed_offset=0,
            shuffle_seed_offset=0,
        )
    )
    second = list(
        create_loader(
            config,
            split="train",
            shuffle=True,
            seed_offset=0,
            shuffle_seed_offset=1,
        )
    )
    first_ids = tuple(sample_id for batch in first for sample_id in batch.sample_ids)
    second_ids = tuple(sample_id for batch in second for sample_id in batch.sample_ids)
    first_images = {
        sample_id: image
        for batch in first
        for sample_id, image in zip(batch.sample_ids, batch.images, strict=True)
    }
    second_images = {
        sample_id: image
        for batch in second
        for sample_id, image in zip(batch.sample_ids, batch.images, strict=True)
    }

    assert first_ids != second_ids
    assert first_images.keys() == second_images.keys()
    for sample_id in first_images:
        assert torch.equal(first_images[sample_id], second_images[sample_id])
