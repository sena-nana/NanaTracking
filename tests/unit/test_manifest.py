from pathlib import Path

import pytest
from pydantic import ValidationError

from nana_tracking.data.labeling import LabelCatalog, materialize_dataset, write_materialized_labels
from nana_tracking.data.manifest import DatasetManifest, dataset_digest


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


def test_teacher_without_complete_license_permission_is_rejected() -> None:
    base = DatasetManifest.load(Path("examples/manifests/synthetic-v1.json")).model_dump()
    base["license_reviews"][0]["permissions"]["commercial_training"] = False
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(base)


def test_dataset_digest_covers_split_and_policy_metadata() -> None:
    manifest = DatasetManifest.load(Path("examples/manifests/synthetic-v1.json"))
    payload = manifest.model_dump(mode="json")
    payload["splits"]["train"]["devices"] = ["replacement-device"]
    changed = DatasetManifest.model_validate(payload)
    assert dataset_digest(manifest) != dataset_digest(changed)


def test_label_catalog_covers_all_stable_signals_with_core_sources() -> None:
    catalog = LabelCatalog.load(Path("configs/data/ntp-v1-label-catalog.json"))
    assert [signal.signal_id for signal in catalog.signals] == list(range(1, 89))
    for signal in catalog.signals[:76]:
        strategy = catalog.strategies[signal.strategy]
        assert strategy.evidence in {"teacher_label", "geometry", "derived"}
        assert strategy.method


def test_materialization_is_deterministic_and_preserves_unavailable_truth(
    tmp_path: Path,
) -> None:
    manifest_path = Path("examples/manifests/synthetic-v1.json")
    first = materialize_dataset(manifest_path)
    second = materialize_dataset(manifest_path)
    assert first == second
    assert first.quality.error_count == 0

    train = first.records[0]
    labels = {label.stable_name: label for label in train.labels}
    assert labels["jaw.open"].state == "available"
    assert labels["eye.left.aperture"].state == "unavailable"
    assert labels["tongue.extension"].state == "unavailable"
    assert labels["auricle.left.elevation"].state == "unavailable"
    assert train.depth.state == "unavailable"
    assert first.records[1].depth.state == "available"
    test_labels = {label.stable_name: label for label in first.records[2].labels}
    assert test_labels["tongue.extension"].state == "available"

    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    write_materialized_labels(first, first_path)
    write_materialized_labels(second, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
