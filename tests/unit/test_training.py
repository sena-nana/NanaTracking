from pathlib import Path

from nana_tracking.config import load_config
from nana_tracking.models import create_model
from nana_tracking.training import train
from nana_tracking.training.checkpoint import load_checkpoint


def test_checkpoint_resume_continues_step(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    config = config.model_copy(
        update={
            "reproducibility": config.reproducibility.model_copy(update={"output_dir": tmp_path}),
            "training": config.training.model_copy(update={"max_steps": 1}),
        }
    )
    first = train(config)
    resumed_config = config.model_copy(
        update={"training": config.training.model_copy(update={"max_steps": 2})}
    )
    resumed = train(resumed_config, resume=first.checkpoint)
    assert resumed.final_step == 2
    assert resumed.run_dir == first.run_dir
    metadata = load_checkpoint(resumed.checkpoint, model=create_model(config.model))
    assert metadata.step == 2
    assert metadata.data_revision == "synthetic-v1"
    assert metadata.device == "cpu"
    assert metadata.amp_enabled is False
