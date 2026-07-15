import hashlib
from pathlib import Path
from typing import cast

import pytest

from nana_tracking.data.manifest import FileReference
from nana_tracking.data.schema import CaptureRecord
from nana_tracking.data.strategy import (
    ActorSplit,
    ActorSplitManifest,
    ClipReference,
    ExpressionCacheManifest,
    ExpressionCacheRecord,
    FrozenFContract,
    LicenseRegistry,
    expression_cache_digest,
    split_captures,
    split_clips_by_actor,
)
from nana_tracking.evaluation.expression import (
    ExpressionAblationConfig,
    run_expression_ablation_smoke,
)


def test_license_registry_fails_closed_for_unapproved_or_smoke_production() -> None:
    registry = LicenseRegistry.load(Path("configs/data/license-registry.json"))
    with pytest.raises(ValueError, match="not approved"):
        registry.admit(["crema-d-odbl-dbcl"], stage="expression-model-training", production=True)
    with pytest.raises(ValueError, match="smoke-only"):
        registry.admit(["nana-synthetic-smoke"], stage="model-release", production=True)
    admitted = registry.admit(
        ["nana-synthetic-smoke"], stage="base-model-training", production=False
    )
    assert [record.record_id for record in admitted] == ["nana-synthetic-smoke"]


def test_f_and_expression_splits_preserve_identity_and_actor_groups() -> None:
    captures = CaptureRecord.load_jsonl(Path("examples/records/synthetic-capture-v1.jsonl"))
    capture_plan = split_captures(
        captures,
        seed=7,
        held_out_test_devices={"synthetic-rgb-device"},
        validation_identities=1,
    )
    assert capture_plan.splits["test"].identities == ["synthetic-test"]
    assert not (
        set(capture_plan.splits["test"].devices) & set(capture_plan.splits["train"].devices)
    )

    clips = [
        ClipReference(clip_id=f"actor-{actor}-clip-{clip}", actor_id=f"actor-{actor}")
        for actor in range(5)
        for clip in range(2)
    ]
    first = split_clips_by_actor(clips, seed=11, validation_actors=1, test_actors=1)
    second = split_clips_by_actor(reversed(clips), seed=11, validation_actors=1, test_actors=1)
    assert first == second
    actor_sets = [set(split.actors) for split in first.splits.values()]
    assert all(
        not left & right
        for index, left in enumerate(actor_sets)
        for right in actor_sets[index + 1 :]
    )


def test_frozen_f_expression_smoke_runs_complete_ablation_without_f_weights(
    tmp_path: Path,
) -> None:
    config = ExpressionAblationConfig.load(Path("configs/expression/ablation-v1.json"))
    config = config.model_copy(update={"steps": 4, "hidden_dims": 8})
    report = run_expression_ablation_smoke(config, tmp_path / "report.json")
    assert report["f_weights_trainable"] is False
    results = report["results"]
    assert isinstance(results, dict)
    typed_results = cast(dict[str, dict[str, float]], results)
    assert {ablation.name for ablation in config.ablations} == set(typed_results)
    assert all(
        "confidence_mae" in metrics and "validation_confidence_mae" in metrics
        for metrics in typed_results.values()
    )
    splits = report["actor_split"]
    assert isinstance(splits, dict)
    typed_splits = cast(dict[str, list[int]], splits)
    assert not set(typed_splits["train"]) & set(typed_splits["test"])
    assert report["resolved_config"] == config.model_dump(mode="json")
    data = cast(dict[str, object], report["data"])
    assert data["revision"] == "synthetic-frozen-f-expression/1.0.0"
    assert isinstance(data["sha256"], str)


def test_expression_cache_verifies_frozen_f_license_digests_and_actor_splits(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "license-registry.json"
    registry_path.write_bytes(Path("configs/data/license-registry.json").read_bytes())
    license_dir = tmp_path / "licenses"
    license_dir.mkdir()
    (license_dir / "nana-synthetic-smoke.txt").write_bytes(
        Path("configs/data/licenses/nana-synthetic-smoke.txt").read_bytes()
    )
    records = [
        ExpressionCacheRecord(
            actor_id=f"actor-{index}",
            clip_id=f"clip-{index}",
            timestamps_ns=[1, 2],
            parameters=[[0.0] * 36, [0.1] * 36],
            confidence=[[1.0] * 36, [1.0] * 36],
            visibility=[1.0, 1.0],
            head_pose=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2,
            frame_quality=[1.0, 1.0],
            emotion_distribution=[1.0, 0.0],
            intensity=0.0,
            label_source="synthetic-votes",
        )
        for index in range(3)
    ]
    shard_path = tmp_path / "cache.jsonl"
    shard_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8"
    )
    splits = ActorSplitManifest(
        seed=1,
        splits={
            "train": ActorSplit(actors=["actor-0"], clips=["clip-0"]),
            "validation": ActorSplit(actors=["actor-1"], clips=["clip-1"]),
            "test": ActorSplit(actors=["actor-2"], clips=["clip-2"]),
        },
    )
    manifest = ExpressionCacheManifest(
        cache_revision="synthetic-cache-v1.1",
        cache_digest="0" * 64,
        source_dataset="NanaTracking generated smoke fixtures",
        source_dataset_revision="1.0.0",
        source_dataset_license_record_id="nana-synthetic-smoke",
        license_registry=FileReference(
            path=registry_path.name,
            sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        ),
        license_record_ids=["nana-synthetic-smoke"],
        frozen_f=FrozenFContract(
            model_family="face-basic",
            model_version="1.0.0",
            model_digest="a" * 64,
            ntp_schema_revision="ntp/1.0",
            signal_registry_revision="ntp-signals/1.0.0",
            feature_revision="ntp-features/1.0.0",
        ),
        parameter_signal_ids=list(range(1, 37)),
        shard_files=[
            FileReference(
                path=shard_path.name,
                sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
            )
        ],
        actor_splits=splits,
        emotion_labels=["neutral", "happy"],
        label_source="synthetic-votes",
        smoke_only=True,
    )
    manifest = manifest.model_copy(update={"cache_digest": expression_cache_digest(manifest)})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    assert len(ExpressionCacheManifest.load(manifest_path).verify_files(manifest_path)) == 3

    mismatched = manifest.model_copy(update={"source_dataset": "unrelated approved dataset"})
    mismatched = mismatched.model_copy(update={"cache_digest": expression_cache_digest(mismatched)})
    mismatched_path = tmp_path / "mismatched-manifest.json"
    mismatched_path.write_text(mismatched.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="identity does not match"):
        ExpressionCacheManifest.load(mismatched_path).verify_files(mismatched_path)
