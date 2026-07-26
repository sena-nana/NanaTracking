"""Pinned commercial teacher artifacts and supervision roles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SEMANTIC_ANCHORS = (
    "eye.left.inner",
    "eye.left.outer",
    "eye.right.inner",
    "eye.right.outer",
    "brow.left.inner",
    "brow.left.outer",
    "brow.right.inner",
    "brow.right.outer",
    "nose.tip",
    "nose.left.alar",
    "nose.right.alar",
    "mouth.left.corner",
    "mouth.right.corner",
    "mouth.upper.center",
    "mouth.lower.center",
    "chin.center",
)


class TeacherContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RemoteArtifact(TeacherContractModel):
    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SemanticAnchorBinding(TeacherContractModel):
    semantic_name: str = Field(min_length=1)
    teacher_index: int = Field(ge=0)


class TeacherModelDescriptor(TeacherContractModel):
    schema_version: Literal["nana-teacher-model/1.0.0"]
    teacher_source_id: str = Field(min_length=1)
    framework: str = Field(min_length=1)
    framework_version: str = Field(min_length=1)
    license_record_id: str = Field(min_length=1)
    model: RemoteArtifact
    model_cards: list[RemoteArtifact] = Field(min_length=1)
    output_contract_revision: str = Field(min_length=1)
    supervision_role: Literal["pseudo-label"]
    admitted_outputs: set[Literal["semantic-landmarks-2d"]] = Field(min_length=1)
    default_label_confidence: float = Field(gt=0.0, lt=1.0)
    anchor_bindings: list[SemanticAnchorBinding]
    prohibited_outputs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_anchor_contract(self) -> Self:
        names = [binding.semantic_name for binding in self.anchor_bindings]
        indices = [binding.teacher_index for binding in self.anchor_bindings]
        if tuple(names) != SEMANTIC_ANCHORS:
            raise ValueError("teacher descriptor must bind the ordered 16 semantic anchors")
        if len(indices) != len(set(indices)):
            raise ValueError("teacher semantic anchor indices must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def verify_model_asset(self, path: Path) -> None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.model.sha256:
            raise ValueError("teacher model asset digest mismatch")
