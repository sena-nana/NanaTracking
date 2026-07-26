"""Shared smoke and commercial artifact governance types."""

from typing import Literal

ArtifactUsageTier = Literal[
    "synthetic-smoke",
    "commercial",
]

TrainingStage = Literal[
    "standard",
    "real-geometry-pretrain",
    "synthetic-head-training",
    "dual-domain-finetune",
]

PipelineStage = Literal[
    "base-model-training",
    "expression-model-training",
    "teacher-labeling",
    "synthetic-rendering",
    "evaluation",
    "model-release",
]
