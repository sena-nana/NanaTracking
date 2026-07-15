from pathlib import Path

import pytest
from pydantic import ValidationError

from nana_tracking.config import ExperimentConfig, load_config


def test_smoke_config_loads() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    assert config.training.device == "cpu"
    assert config.reproducibility.ntp_schema_revision == "smoke-ntp-v0"


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentConfig.model_validate(
            {
                "unexpected": True,
                "export": {"model_family": "smoke", "model_version": "v0"},
                "reproducibility": {
                    "data_revision": "data-v1",
                    "ntp_schema_revision": "ntp-v1",
                    "signal_registry_revision": "signals-v1",
                    "feature_revision": "features-v1",
                },
            }
        )


def test_parallel_executor_requires_workers() -> None:
    with pytest.raises(ValidationError, match="workers >= 1"):
        ExperimentConfig.model_validate(
            {
                "data": {"executor": "interpreter", "workers": 0},
                "export": {"model_family": "smoke", "model_version": "v0"},
                "reproducibility": {
                    "data_revision": "data-v1",
                    "ntp_schema_revision": "ntp-v1",
                    "signal_registry_revision": "signals-v1",
                    "feature_revision": "features-v1",
                },
            }
        )


def test_synthetic_data_cannot_claim_non_smoke_artifact() -> None:
    config = load_config(Path("configs/face-basic-smoke.yaml")).model_dump(mode="json")
    config["export"]["smoke_only"] = False
    with pytest.raises(ValidationError, match="reviewed manifest"):
        ExperimentConfig.model_validate(config)
