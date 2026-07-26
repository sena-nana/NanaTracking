import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torchvision.io import write_png

from nana_tracking.config import ExperimentConfig, load_config
from nana_tracking.data.loaders import create_loader
from nana_tracking.data.manifest import DatasetManifest, dataset_digest
from nana_tracking.data.mediapipe_mapping import render_anchor_review_overlays
from nana_tracking.data.multiface import (
    SEMANTIC_ANCHORS,
    AnchorCorrespondence,
    AnchorMappingReview,
    AnchorVote,
    MultifaceCamera,
    SemanticAnchorMapping,
    build_anchor_mapping_proposal,
    build_multiface_split_plan,
    preflight_multiface,
)
from nana_tracking.data.strategy import LicenseRegistry
from nana_tracking.models import create_model, output_names
from nana_tracking.training import train
from nana_tracking.training.checkpoint import load_checkpoint
from nana_tracking.training.recipe import TrainingRecipe
from nana_tracking.training.stage_a import (
    compute_stage_a_loss_components,
    quaternion_geodesic,
    stage_a_parameters,
)


def _stage_a_outputs(
    config: ExperimentConfig,
    model: torch.nn.Module,
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    groups, views, channels, height, width = images.shape
    values = model(images.reshape(groups * views, channels, height, width))
    return {
        name: value.reshape(groups, views, *value.shape[1:])
        for name, value in zip(output_names(config.model), values, strict=True)
    }


def test_recipe_and_pending_anchor_map_are_fail_closed() -> None:
    recipe = TrainingRecipe.load(Path("configs/training/nana-training-recipe-1.0.0.json"))
    assert recipe.promotion_status == "research-candidate"
    with pytest.raises(ValueError, match="approved"):
        SemanticAnchorMapping.load(
            Path("configs/data/multiface-semantic-anchors-v1.pending.json"),
            require_approved=True,
        )


def test_mediapipe_votes_only_create_a_pending_mapping() -> None:
    votes = [
        AnchorVote(
            semantic_name=semantic,
            mediapipe_index=semantic_index,
            multiface_vertex_id=semantic_index,
            distance=0.01 + identity_index * 0.001,
            identity_id=f"identity-{identity_index}",
            expression=f"expression-{expression_index}",
            camera_id=f"camera-{camera_index}",
            sample_id=(f"{semantic_index}-{identity_index}-{expression_index}-{camera_index}"),
        )
        for semantic_index, semantic in enumerate(SEMANTIC_ANCHORS)
        for identity_index in range(3)
        for expression_index in range(4)
        for camera_index in range(3)
    ]
    proposal = build_anchor_mapping_proposal(
        votes,
        mapping_revision="test-mapping/1.0.0",
        mediapipe_version="0.10.35",
        mediapipe_model_sha256="a" * 64,
    )
    assert proposal.status == "pending"
    assert [item.semantic_name for item in proposal.correspondences] == list(SEMANTIC_ANCHORS)
    approved = proposal.model_copy(
        update={
            "status": "approved",
            "review": AnchorMappingReview(
                approved_by="unit-test-reviewer",
                approved_at=datetime(2026, 7, 26, tzinfo=UTC),
                identities=[f"identity-{index}" for index in range(3)],
                expressions=[f"expression-{index}" for index in range(4)],
                camera_ids=[f"camera-{index}" for index in range(3)],
                reviewed_sample_ids=[f"sample-{index}" for index in range(12)],
                overlay_sha256="b" * 64,
            ),
        }
    )
    assert SemanticAnchorMapping.model_validate(approved.model_dump()).status == "approved"


def test_anchor_overlay_evidence_is_digest_pinned(tmp_path: Path) -> None:
    mapping = SemanticAnchorMapping(
        schema_version="nana-semantic-anchor-map/1.0.0",
        mapping_revision="overlay-test/1.0.0",
        source_topology="multiface-tracked-mesh-7306",
        mediapipe_version="0.10.35",
        mediapipe_model_sha256="a" * 64,
        status="pending",
        correspondences=[
            AnchorCorrespondence(
                semantic_name=name,
                mediapipe_index=index,
                multiface_vertex_id=index,
            )
            for index, name in enumerate(SEMANTIC_ANCHORS)
        ],
        review=None,
        note="test-only pending candidate",
    )
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(mapping.model_dump_json(), encoding="utf-8")
    image_path = tmp_path / "image.png"
    write_png(torch.zeros(3, 32, 32, dtype=torch.uint8), str(image_path))
    mesh = np.zeros((7_306, 2), dtype=np.float32)
    mesh[:16, 0] = np.linspace(0.2, 0.8, 16)
    mesh[:16, 1] = np.linspace(0.8, 0.2, 16)
    mesh_path = tmp_path / "mesh.npy"
    np.save(mesh_path, mesh, allow_pickle=False)
    index_path = tmp_path / "review.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample/one",
                    "identity_id": "identity",
                    "expression": "neutral",
                    "camera_id": "camera",
                    "image": str(image_path),
                    "projected_mesh": str(mesh_path),
                }
            ]
        ),
        encoding="utf-8",
    )
    report = render_anchor_review_overlays(
        review_index=index_path,
        mapping_path=mapping_path,
        output_directory=tmp_path / "overlays",
    )
    assert report["sample_count"] == 1
    assert len(str(report["overlay_sha256"])) == 64
    assert (tmp_path / "overlays" / "sample-one.png").is_file()


def test_multiface_preflight_rejects_pending_license_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network must not run before license admission")

    monkeypatch.setattr(
        "nana_tracking.data.multiface.urlopen",
        unexpected_network,
    )
    with pytest.raises(ValueError, match="not approved"):
        preflight_multiface(
            registry_path=Path("configs/data/license-registry.json"),
        )


def test_multiface_split_is_identity_deterministic_and_camera_disjoint() -> None:
    cameras: list[MultifaceCamera] = []
    for index, yaw in enumerate((-60, -45, -30, -15, 0, 15, 30, 45, 60)):
        radians = math.radians(yaw)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        cameras.append(
            MultifaceCamera(
                camera_id=f"camera-{index}",
                camera_to_capture=(
                    (cosine, 0.0, sine, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (-sine, 0.0, cosine, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        )
    identities = [f"identity-{index}" for index in range(8)]
    first = build_multiface_split_plan(identities, cameras)
    second = build_multiface_split_plan(reversed(identities), reversed(cameras))
    assert first == second
    identity_sets = [set(split.identities) for split in first.splits.values()]
    camera_sets = [set(split.devices) for split in first.splits.values()]
    assert set.union(*identity_sets) == set(identities)
    assert all(
        not left & right
        for index, left in enumerate(identity_sets)
        for right in identity_sets[index + 1 :]
    )
    assert all(
        not left & right
        for index, left in enumerate(camera_sets)
        for right in camera_sets[index + 1 :]
    )
    assert first.classification == "research-cohort"


def test_research_license_permissions_are_independent_from_commercial_use(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path("configs/data/license-registry.json").read_text(encoding="utf-8"))
    record = next(
        item for item in payload["records"] if item["record_id"] == "research-multiface-cc-by-nc"
    )
    license_directory = tmp_path / "licenses"
    license_directory.mkdir()
    shutil.copy2(
        Path("configs/data/licenses/nana-synthetic-smoke.txt"),
        license_directory / "nana-synthetic-smoke.txt",
    )
    license_text = license_directory / "multiface.txt"
    license_text.write_text("test-only pinned CC BY-NC review fixture\n", encoding="utf-8")
    record.update(
        {
            "license": "licenses/multiface.txt",
            "license_text_sha256": hashlib.sha256(license_text.read_bytes()).hexdigest(),
            "review_status": "approved",
            "allowed_pipeline_stages": [
                "research-mapping",
                "research-model-training",
                "research-evaluation",
            ],
        }
    )
    record["permissions"].update(
        {
            "noncommercial_research_training_allowed": True,
            "research_derivative_labels_allowed": True,
            "research_checkpoint_local_use_allowed": True,
        }
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = LicenseRegistry.load(registry_path)
    registry.verify_local_license_texts(registry_path)
    admitted = registry.admit(
        ["research-multiface-cc-by-nc"],
        stage="research-model-training",
        production=False,
        usage_tier="noncommercial-research",
    )
    assert admitted[0].record_id == "research-multiface-cc-by-nc"
    with pytest.raises(ValueError, match="does not allow base-model-training"):
        registry.admit(
            ["research-multiface-cc-by-nc"],
            stage="base-model-training",
            production=True,
            usage_tier="commercial",
        )


def test_manifest_v3_digest_covers_research_mapping_and_recipe() -> None:
    payload = DatasetManifest.load(Path("examples/manifests/synthetic-v1.json")).model_dump(
        mode="json"
    )
    payload.update(
        {
            "schema_version": "ntp-dataset/3.0.0",
            "pipeline_stage": "research-model-training",
            "usage_tier": "noncommercial-research",
            "smoke_only": False,
            "anchor_mapping": {"path": "anchors.json", "sha256": "a" * 64},
            "training_recipe": {"path": "recipe.json", "sha256": "b" * 64},
            "split_plan": {"path": "splits.json", "sha256": "d" * 64},
        }
    )
    for review in payload["license_reviews"]:
        review["permissions"].update(
            {
                "noncommercial_research_training": True,
                "research_derivative_labels": True,
                "research_checkpoint_local_use": True,
            }
        )
    manifest = DatasetManifest.model_validate(payload)
    changed_payload = manifest.model_dump(mode="json")
    changed_payload["anchor_mapping"]["sha256"] = "c" * 64
    changed = DatasetManifest.model_validate(changed_payload)
    assert dataset_digest(manifest) != dataset_digest(changed)


def test_research_configuration_must_disable_export() -> None:
    payload = load_config(Path("configs/face-basic-stage-a-smoke.yaml")).model_dump(mode="json")
    payload["data"].update(
        {
            "dataset": "multiface",
            "usage_tier": "noncommercial-research",
            "manifest": "data/manifest.json",
            "anchor_mapping": "data/anchors.json",
        }
    )
    payload["export"]["smoke_only"] = False
    payload["export"]["enabled"] = True
    payload["reproducibility"].update(
        {
            "manifest_digest": "a" * 64,
            "license_registry_digest": "b" * 64,
            "anchor_mapping_digest": "c" * 64,
            "training_recipe_digest": "d" * 64,
        }
    )
    with pytest.raises(ValidationError, match="disable export"):
        ExperimentConfig.model_validate(payload)


def test_stage_a_loss_is_finite_and_rig_confidence_heads_do_not_change() -> None:
    config = load_config(Path("configs/face-basic-stage-a-smoke.yaml"))
    batch = next(iter(create_loader(config, split="train", shuffle=False, seed_offset=0)))
    model = create_model(config.model)
    rig_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("rig_head.", "confidence_head."))
    }
    outputs = _stage_a_outputs(config, model, batch.images)
    components = compute_stage_a_loss_components(config, outputs, batch)
    assert set(components) == {
        "landmarks",
        "canonical_geometry",
        "pose_rotation",
        "pose_translation",
        "reprojection",
        "multiview_geometry",
        "multiview_pose",
        "visibility",
        "identity_adversary",
    }
    assert all(torch.isfinite(value) for value in components.values())
    optimizer = torch.optim.AdamW(stage_a_parameters(model), lr=0.001)
    torch.stack(tuple(components.values())).sum().backward()
    optimizer.step()
    for name, before in rig_before.items():
        assert torch.equal(before, model.state_dict()[name])

    quaternion = torch.tensor([[0.1, -0.2, 0.3, 0.9]])
    torch.testing.assert_close(
        quaternion_geodesic(quaternion, -quaternion),
        torch.zeros(1),
        atol=1e-6,
        rtol=0.0,
    )


def test_stage_a_checkpoint_resume_is_exact_and_cross_tier_is_rejected(
    tmp_path: Path,
) -> None:
    base = load_config(Path("configs/face-basic-stage-a-smoke.yaml"))
    config = base.model_copy(
        update={
            "training": base.training.model_copy(
                update={
                    "max_steps": 2,
                    "validation_interval_steps": 1,
                    "checkpoint_interval_steps": 1,
                }
            ),
            "reproducibility": base.reproducibility.model_copy(
                update={"output_dir": tmp_path / "direct"}
            ),
        }
    )
    direct = train(config)
    intermediate = direct.run_dir / "checkpoints" / "step-00000001.pt"
    copied = tmp_path / "resume-copy" / direct.run_dir.name / "checkpoints" / intermediate.name
    copied.parent.mkdir(parents=True)
    shutil.copy2(intermediate, copied)
    shutil.copy2(intermediate.with_suffix(".json"), copied.with_suffix(".json"))
    resumed = train(config, resume=copied)

    direct_model = create_model(config.model)
    resumed_model = create_model(config.model)
    load_checkpoint(
        direct.checkpoint,
        model=direct_model,
        expected_usage_tier="synthetic-smoke",
    )
    metadata = load_checkpoint(
        resumed.checkpoint,
        model=resumed_model,
        expected_usage_tier="synthetic-smoke",
    )
    for name, value in direct_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name])
    assert metadata.training_stage == "real-geometry-pretrain"
    assert metadata.parent_checkpoint_digest is not None

    with pytest.raises(ValueError, match="usage tier mismatch"):
        load_checkpoint(
            resumed.checkpoint,
            model=create_model(config.model),
            expected_usage_tier="commercial",
        )
    summary = json.loads(resumed.summary_report.read_text(encoding="utf-8"))
    assert summary["usage_tier"] == "synthetic-smoke"
    assert "Synthetic smoke evidence only" in summary["limitations"]
