from pathlib import Path

from nana_tracking.config import ExperimentConfig, load_config
from nana_tracking.data.capture import run_capture_pipeline_smoke
from nana_tracking.training import train


def test_frozen_capture_is_the_verified_training_entrypoint(tmp_path: Path) -> None:
    report = run_capture_pipeline_smoke(
        tmp_path / "capture",
        mapping_path=Path("configs/data/arkit-to-ntp-v1-smoke.json"),
        license_registry=Path("configs/data/license-registry.json"),
        label_catalog_path=Path("configs/data/ntp-v1-label-catalog.json"),
    )
    frozen_path = Path(str(report["dataset"])).resolve()
    manifest_path = Path(str(report["training_manifest"])).resolve()
    base = load_config(Path("configs/face-basic-smoke.yaml"))
    payload = base.model_dump(mode="python")
    payload["data"].update(
        {
            "dataset": "frozen_capture",
            "manifest": manifest_path,
            "frozen_capture": frozen_path,
            "require_complete_basic": False,
        }
    )
    payload["reproducibility"].update(
        {
            "output_dir": tmp_path / "runs",
            "data_revision": "capture-pipeline-smoke-v1",
        }
    )
    config = ExperimentConfig.model_validate(payload)

    result = train(config)

    assert result.final_step == 1
    assert result.checkpoint.is_file()
