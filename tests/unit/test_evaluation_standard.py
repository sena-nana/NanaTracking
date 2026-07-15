from pathlib import Path

import pytest
from pydantic import ValidationError

from nana_tracking.evaluation.standard import REQUIRED_METRICS, EvaluationStandard, MetricResult


def test_all_profiles_share_one_complete_evaluation_standard() -> None:
    path = Path("configs/evaluation/ntp-v1-standard.json")
    standard = EvaluationStandard.load(path)
    sequences, report = standard.validate_assets(path)

    suites = {suite.profile: suite for suite in standard.suites}
    assert suites["Basic"].required_signal_ids == list(range(1, 37))
    assert suites["Spatial"].required_signal_ids == list(range(1, 42))
    assert suites["Full"].required_signal_ids == list(range(1, 77))
    assert all(set(suite.metric_ids) == REQUIRED_METRICS for suite in suites.values())
    assert len(sequences.sequences) == 19
    report_metrics = {
        metric_id for family in report.output_families.values() for metric_id in family
    }
    assert report_metrics == REQUIRED_METRICS


def test_profile_cannot_silently_drop_a_required_metric() -> None:
    standard = EvaluationStandard.load(Path("configs/evaluation/ntp-v1-standard.json"))
    payload = standard.model_dump(mode="json")
    payload["suites"][0]["metric_ids"].pop()
    with pytest.raises(ValidationError):
        EvaluationStandard.model_validate(payload)


def test_measured_metric_keeps_named_aggregations_and_sample_count() -> None:
    result = MetricResult(
        status="measured",
        values={"signal-17.p50": 0.02, "signal-17.p95": 0.05},
        sample_count=120,
    )
    assert result.values is not None
    assert result.values["signal-17.p95"] == 0.05
    with pytest.raises(ValidationError):
        MetricResult(status="measured", values={"p50": 0.02})
