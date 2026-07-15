"""Privacy-aware failure-sample capture and local visualization."""

import html
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class FailureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FailureSample(FailureModel):
    schema_version: Literal["ntp-failure-sample/1.0.0"] = "ntp-failure-sample/1.0.0"
    sample_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    capture_timestamp_ns: int = Field(ge=0)
    image_uri: str = Field(min_length=1)
    categories: set[
        Literal[
            "signal_error",
            "jitter",
            "latency",
            "occlusion",
            "out_of_frame",
            "tracking_lost",
            "confidence",
            "geometry",
        ]
    ] = Field(min_length=1)
    signal_ids: list[int] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def load_jsonl(cls, path: Path) -> list[Self]:
        samples = [
            cls.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("failure sample IDs must be unique")
        return samples


def write_failure_samples(path: Path, samples: list[FailureSample]) -> None:
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("failure sample IDs must be unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(sample.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
            for sample in samples
        )
        + "\n",
        encoding="utf-8",
    )


def render_failure_report(samples: list[FailureSample], output: Path) -> None:
    """Render a local report that links images without embedding private recordings."""

    cards: list[str] = []
    for sample in samples:
        metrics = " ".join(
            f"<li>{html.escape(name)}: {value:.6g}</li>"
            for name, value in sorted(sample.metrics.items())
        )
        categories = ", ".join(sorted(sample.categories))
        signals = ", ".join(str(signal) for signal in sample.signal_ids) or "none"
        notes = f"<p>{html.escape(sample.notes)}</p>" if sample.notes else ""
        cards.append(
            "<article>"
            f"<h2>{html.escape(sample.sample_id)}</h2>"
            f'<img src="{html.escape(sample.image_uri, quote=True)}" alt="Failure sample">'
            f"<p>split: {sample.split}; categories: {html.escape(categories)}; "
            f"signals: {html.escape(signals)}</p>"
            f"<ul>{metrics}</ul>{notes}</article>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>FaceBasic failure samples</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:auto}}article{{border-bottom:1px solid
#ccc;padding:1rem 0}}img{{max-width:480px;max-height:320px;object-fit:contain}}</style></head>
<body><h1>FaceBasic failure samples</h1>{"".join(cards)}</body></html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
