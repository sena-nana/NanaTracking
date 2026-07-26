"""Versioned training-recipe promotion contract."""

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["nana-training-recipe/1.0.0"]
    recipe_id: str = Field(min_length=1)
    promotion_status: Literal[
        "research-candidate",
        "commercial-reproduced",
        "release-eligible",
    ]
    usage_tier: Literal["noncommercial-research", "commercial"]
    parent_checkpoint: str | None
    seed: int = Field(ge=0)
    model: dict[str, Any]
    schedule: dict[str, Any]
    loss_weights: dict[str, float]
    data_policy: dict[str, Any]
    forbidden_inheritance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_promotion(self) -> Self:
        if self.promotion_status == "research-candidate":
            if self.usage_tier != "noncommercial-research":
                raise ValueError("research-candidate recipes must remain noncommercial")
            if self.parent_checkpoint is not None:
                raise ValueError("the Stage A research recipe must start without parent weights")
        required = {
            "multiface_weights",
            "ema",
            "optimizer_state",
            "rng_state",
            "normalization_statistics",
            "raw_data",
            "teacher_labels",
            "topology_vertex_ids",
        }
        missing = required.difference(self.forbidden_inheritance)
        if missing:
            raise ValueError(f"recipe omits forbidden research artifacts: {sorted(missing)}")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
