import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from nana_tracking.config import ExperimentConfig, load_config
from nana_tracking.data.loaders import create_loader
from nana_tracking.data.strategy import LicenseRegistry
from nana_tracking.data.teachers import SEMANTIC_ANCHORS, TeacherModelDescriptor
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


def test_commercial_recipe_and_pinned_mediapipe_descriptor(tmp_path: Path) -> None:
    recipe = TrainingRecipe.load(Path("configs/training/nana-training-recipe-1.0.0.json"))
    assert recipe.promotion_status == "pilot-only"
    assert recipe.usage_tier == "commercial"
    assert recipe.parent_checkpoint is None
    assert recipe.data_policy["existing_datasets_allowed"] is False
    assert recipe.data_policy["training_teachers"] == [
        "mediapipe-face-landmarker-v1",
        "opencv-calibrated-geometry-v1",
    ]
    assert recipe.data_policy["evaluation_only_references"] == ["apple-arkit-truedepth-teacher"]

    descriptor = TeacherModelDescriptor.load(Path("configs/data/mediapipe-face-landmarker-v1.json"))
    assert (
        tuple(binding.semantic_name for binding in descriptor.anchor_bindings) == SEMANTIC_ANCHORS
    )
    assert descriptor.admitted_outputs == {"semantic-landmarks-2d"}
    assert "NTP Basic 36 numeric truth" in descriptor.prohibited_outputs

    model_asset = tmp_path / "teacher.task"
    model_asset.write_bytes(b"pinned-test-model")
    test_descriptor = descriptor.model_copy(
        update={
            "model": descriptor.model.model_copy(
                update={"sha256": hashlib.sha256(model_asset.read_bytes()).hexdigest()}
            )
        }
    )
    test_descriptor.verify_model_asset(model_asset)
    model_asset.write_bytes(b"drifted")
    with pytest.raises(ValueError, match="digest mismatch"):
        test_descriptor.verify_model_asset(model_asset)


def test_registry_admits_only_training_teachers_and_arkit_evaluation() -> None:
    registry_path = Path("configs/data/license-registry.json")
    registry = LicenseRegistry.load(registry_path)
    registry.verify_local_license_texts(registry_path)

    training = registry.admit(
        ["mediapipe-face-landmarker-v1", "opencv-calibrated-geometry-v1"],
        stage="teacher-labeling",
        production=True,
        usage_tier="commercial",
    )
    assert {record.record_id for record in training} == {
        "mediapipe-face-landmarker-v1",
        "opencv-calibrated-geometry-v1",
    }
    assert next(
        record for record in training if record.record_id == "mediapipe-face-landmarker-v1"
    ).teacher_supervision_roles == {"pseudo-label"}
    assert next(
        record for record in training if record.record_id == "opencv-calibrated-geometry-v1"
    ).teacher_supervision_roles == {"geometry-derivation"}

    comparison = registry.admit(
        ["apple-arkit-truedepth-teacher"],
        stage="evaluation",
        production=True,
        usage_tier="commercial",
    )
    assert comparison[0].teacher_supervision_roles == {"evaluation-reference"}
    with pytest.raises(ValueError, match="does not allow base-model-training"):
        registry.admit(
            ["apple-arkit-truedepth-teacher"],
            stage="base-model-training",
            production=True,
            usage_tier="commercial",
        )

    for rejected_dataset in (
        "crema-d-odbl-dbcl",
        "ict-face-model-light",
        "research-multiface-cc-by-nc",
    ):
        with pytest.raises(ValueError, match="not approved"):
            registry.admit(
                [rejected_dataset],
                stage="base-model-training",
                production=True,
                usage_tier="commercial",
            )


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
