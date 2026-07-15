from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from nana_tracking.contracts import AdapterContract
from nana_tracking.personalization import (
    AffineResidualAdapter,
    BoundedOnlineCalibration,
    LevelACalibration,
    OrtLevelBAdapter,
    ProfileArtifact,
    ProfileCompatibility,
    SignalCalibration,
    UserProfileMetadata,
    ensure_adapter_compatible,
    profile_compatibility,
    train_level_b_adapter,
    verify_level_b_adapter,
)


def _user_profile() -> UserProfileMetadata:
    timestamp = datetime(2026, 7, 15, tzinfo=UTC)
    return UserProfileMetadata(
        user_slot="user-a",
        base_model_family="face-basic",
        base_model_version="1.2.3",
        base_model_digest="a" * 64,
        feature_revision="features-v1",
        signal_registry_revision="signals-v1",
        calibration_revision="calibration-v1",
        created_at=timestamp,
        updated_at=timestamp,
        artifacts=[
            ProfileArtifact(
                kind="level-a",
                relative_path="level-a.json",
                digest="b" * 64,
                runtime="native",
            )
        ],
    )


def test_user_profile_round_trip_and_compatibility_gates(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    _user_profile().save(path)
    profile = UserProfileMetadata.load(path)
    common = {
        "user_slot": "user-a",
        "base_model_family": "face-basic",
        "base_model_digest": "a" * 64,
        "feature_revision": "features-v1",
        "signal_registry_revision": "signals-v1",
        "calibration_revision": "calibration-v1",
    }
    assert (
        profile_compatibility(profile, base_model_version="1.2.3", **common)
        is ProfileCompatibility.EXACT
    )
    assert (
        profile_compatibility(profile, base_model_version="1.2.4", **common)
        is ProfileCompatibility.REVALIDATION_REQUIRED
    )
    assert (
        profile_compatibility(
            profile,
            base_model_version="1.2.3",
            **(common | {"base_model_digest": "c" * 64}),
        )
        is ProfileCompatibility.INCOMPATIBLE
    )


def test_adapter_starts_as_identity_residual() -> None:
    adapter = AffineResidualAdapter(3)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(adapter(values), values)


def test_adapter_revision_mismatch_fails_safe() -> None:
    contract = AdapterContract(
        adapter_type="affine-residual",
        base_model_family="face-basic",
        base_model_version="1.0.0",
        feature_revision="features-v1",
    )
    with pytest.raises(ValueError, match="incompatible"):
        ensure_adapter_compatible(
            contract,
            model_family="face-basic",
            model_version="1.0.0",
            feature_revision="features-v2",
        )


def _level_a() -> LevelACalibration:
    return LevelACalibration(
        user_slot="user-a",
        model_family="face-basic",
        model_version="1.0.0",
        feature_revision="features-v1",
        signal_registry_revision="signals-v1",
        normalization_revision="normalization-v1",
        calibration_revision="calibration-v1",
        signals=[
            SignalCalibration(
                signal_id=signal_id,
                neutral=0.0,
                negative_span=1.0,
                positive_span=1.0,
                deadzone=0.05,
            )
            for signal_id in range(1, 4)
        ],
    )


def test_level_a_supports_versioned_non_basic_profiles_and_deadzone() -> None:
    profile = _level_a()
    output = profile.apply(np.array([0.02, 0.5, -0.5], dtype=np.float32))
    assert output[0] == 0.0
    assert output[1] > 0.0
    assert output[2] < 0.0


def test_online_calibration_requires_explicit_stable_evidence_and_rolls_back() -> None:
    online = BoundedOnlineCalibration(_level_a(), minimum_stable_ns=100, maximum_step=0.01)
    values = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    confidence = np.ones(3, dtype=np.float32)
    assert not online.update(
        values,
        confidence,
        capture_timestamp_ns=1,
        stable_duration_ns=99,
        evidence="explicit_neutral",
    )
    assert online.current.signals[0].neutral == 0.0
    assert online.update(
        values,
        confidence,
        capture_timestamp_ns=2,
        stable_duration_ns=100,
        evidence="explicit_neutral",
    )
    assert online.current.signals[0].neutral == pytest.approx(0.01)
    assert online.rollback()
    assert online.current.signals[0].neutral == 0.0
    assert not online.rollback()


def test_level_b_adapter_is_offline_portable_and_user_isolated(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    base = rng.uniform(-0.5, 0.5, size=(32, 3)).astype(np.float32)
    target = base * np.array([1.1, 0.9, 1.0], dtype=np.float32) + np.array(
        [0.05, -0.03, 0.02], dtype=np.float32
    )
    confidence = np.ones_like(base)
    package = tmp_path / "adapter"
    metadata = train_level_b_adapter(
        base,
        target,
        confidence,
        package,
        user_slot="user-a",
        base_model_family="face-basic",
        base_model_version="1.0.0",
        base_model_digest="a" * 64,
        feature_revision="features-v1",
        signal_registry_revision="signals-v1",
        normalization_revision="normalization-v1",
        calibration_revision="calibration-v1",
        signal_ids=[1, 2, 3],
        steps=40,
    )
    assert metadata.user_slot == "user-a"
    parity = verify_level_b_adapter(
        package,
        user_slot="user-a",
        base_model_family="face-basic",
        base_model_version="1.0.0",
        base_model_digest="a" * 64,
        feature_revision="features-v1",
    )
    assert parity["max_abs"] <= 1e-6
    runtime = OrtLevelBAdapter(
        package,
        user_slot="user-a",
        base_model_family="face-basic",
        base_model_version="1.0.0",
        base_model_digest="a" * 64,
        feature_revision="features-v1",
    )
    adapted = np.asarray(runtime.apply(tuple(float(value) for value in base[0])))
    assert np.abs(adapted - target[0]).mean() < np.abs(base[0] - target[0]).mean()
    with pytest.raises(ValueError, match="active user"):
        verify_level_b_adapter(
            package,
            user_slot="user-b",
            base_model_family="face-basic",
            base_model_version="1.0.0",
            base_model_digest="a" * 64,
            feature_revision="features-v1",
        )
