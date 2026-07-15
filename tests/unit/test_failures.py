from pathlib import Path

from nana_tracking.evaluation.failures import (
    FailureSample,
    render_failure_report,
    write_failure_samples,
)


def test_failure_samples_round_trip_and_render_actual_images(tmp_path: Path) -> None:
    sample = FailureSample(
        sample_id="validation-frame-42",
        split="validation",
        capture_timestamp_ns=42,
        image_uri="frames/42.png",
        categories={"occlusion", "confidence"},
        signal_ids=[7, 8],
        metrics={"confidence_ece": 0.2},
        notes="hand over left eye",
    )
    jsonl = tmp_path / "failures.jsonl"
    write_failure_samples(jsonl, [sample])
    loaded = FailureSample.load_jsonl(jsonl)
    assert loaded == [sample]
    report = tmp_path / "failures.html"
    render_failure_report(loaded, report)
    html = report.read_text(encoding="utf-8")
    assert 'src="frames/42.png"' in html
    assert "validation-frame-42" in html
