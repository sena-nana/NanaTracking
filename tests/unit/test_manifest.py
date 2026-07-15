from pathlib import Path

import pytest
from pydantic import ValidationError

from nana_tracking.data.manifest import DatasetManifest


def test_example_manifest_is_identity_safe() -> None:
    manifest = DatasetManifest.load(Path("examples/manifests/synthetic-v1.json"))
    assert manifest.splits["train"].identities == ["synthetic-train"]


def test_identity_leakage_is_rejected() -> None:
    base = DatasetManifest.load(Path("examples/manifests/synthetic-v1.json")).model_dump()
    base["splits"]["validation"]["identities"] = ["synthetic-train"]
    with pytest.raises(ValidationError, match="appears in both"):
        DatasetManifest.model_validate(base)


def test_missing_revision_is_rejected() -> None:
    base = DatasetManifest.load(Path("examples/manifests/synthetic-v1.json")).model_dump()
    base["data_revision"] = ""
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(base)
