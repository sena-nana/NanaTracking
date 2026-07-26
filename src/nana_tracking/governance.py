"""Shared research, smoke, and commercial artifact governance types."""

from typing import Literal

ArtifactUsageTier = Literal[
    "synthetic-smoke",
    "noncommercial-research",
    "commercial",
]

TrainingStage = Literal[
    "standard",
    "real-geometry-pretrain",
    "synthetic-head-training",
    "dual-domain-finetune",
]

ResearchPipelineStage = Literal[
    "research-mapping",
    "research-model-training",
    "research-evaluation",
]

PipelineStage = Literal[
    "base-model-training",
    "expression-model-training",
    "teacher-labeling",
    "synthetic-rendering",
    "evaluation",
    "model-release",
    "research-mapping",
    "research-model-training",
    "research-evaluation",
]
