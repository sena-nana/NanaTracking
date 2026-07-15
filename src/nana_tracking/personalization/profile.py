"""Versioned user-profile metadata kept separate from base model artifacts."""

import json
from datetime import datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProfileArtifact(_Contract):
    kind: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: str = Field(min_length=1)
    resettable: bool = True


class UserProfileMetadata(_Contract):
    schema_version: str = "nana-user-profile/1.0.0"
    user_slot: str = Field(min_length=1)
    base_model_family: str = Field(min_length=1)
    base_model_version: str = Field(min_length=1)
    base_model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    calibration_revision: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    rollback_generation: int = Field(default=0, ge=0)
    artifacts: list[ProfileArtifact]
    validation_metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        kinds = [artifact.kind for artifact in self.artifacts]
        if kinds != sorted(set(kinds)):
            raise ValueError("profile artifacts must use unique increasing kinds")
        if self.updated_at < self.created_at:
            raise ValueError("profile update time cannot precede creation")
        if any(not key or not isfinite(value) for key, value in self.validation_metrics.items()):
            raise ValueError("profile validation metrics must be named and finite")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class ProfileCompatibility(StrEnum):
    EXACT = "Exact"
    REVALIDATION_REQUIRED = "RevalidationRequired"
    INCOMPATIBLE = "Incompatible"


def profile_compatibility(
    profile: UserProfileMetadata,
    *,
    user_slot: str,
    base_model_family: str,
    base_model_version: str,
    base_model_digest: str,
    feature_revision: str,
    signal_registry_revision: str,
    calibration_revision: str,
) -> ProfileCompatibility:
    if (
        profile.user_slot != user_slot
        or profile.base_model_family != base_model_family
        or profile.feature_revision != feature_revision
        or profile.signal_registry_revision != signal_registry_revision
    ):
        return ProfileCompatibility.INCOMPATIBLE
    if profile.base_model_version == base_model_version:
        if profile.base_model_digest != base_model_digest:
            return ProfileCompatibility.INCOMPATIBLE
        if profile.calibration_revision != calibration_revision:
            return ProfileCompatibility.REVALIDATION_REQUIRED
        return ProfileCompatibility.EXACT
    current = _semantic_version(profile.base_model_version)
    requested = _semantic_version(base_model_version)
    if current is not None and requested is not None and current[:2] == requested[:2]:
        return ProfileCompatibility.REVALIDATION_REQUIRED
    return ProfileCompatibility.INCOMPATIBLE


def _semantic_version(value: str) -> tuple[int, int, int] | None:
    try:
        components = tuple(int(component) for component in value.split("."))
    except ValueError:
        return None
    return components if len(components) == 3 else None
