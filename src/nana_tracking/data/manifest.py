"""Versioned dataset manifest and identity-safe split validation."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitManifest(ManifestModel):
    identities: list[str] = Field(min_length=1)
    sessions: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)


class DatasetManifest(ManifestModel):
    schema_version: str = Field(min_length=1)
    data_revision: str = Field(min_length=1)
    digest: str = Field(min_length=16)
    ntp_schema_revision: str = Field(min_length=1)
    signal_registry_revision: str = Field(min_length=1)
    splits: dict[str, SplitManifest]

    @model_validator(mode="after")
    def validate_identity_isolation(self) -> "DatasetManifest":
        required = {"train", "validation", "test"}
        missing = required.difference(self.splits)
        if missing:
            raise ValueError(f"missing required splits: {sorted(missing)}")

        owners: dict[str, str] = {}
        for split_name, split in self.splits.items():
            for identity in split.identities:
                previous = owners.setdefault(identity, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"identity {identity!r} appears in both {previous!r} and {split_name!r}"
                    )
        return self

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
