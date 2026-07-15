"""Tracking model evaluation APIs."""

from nana_tracking.evaluation.acceptance import validate_face_basic_acceptance
from nana_tracking.evaluation.confidence import (
    ConfidenceCalibration,
    ConfidenceCurve,
    fit_confidence_calibration,
)
from nana_tracking.evaluation.evaluator import evaluate
from nana_tracking.evaluation.expression import (
    ExpressionAblationConfig,
    ParameterExpressionModel,
    run_expression_ablation_smoke,
)
from nana_tracking.evaluation.failures import FailureSample, render_failure_report
from nana_tracking.evaluation.runtime import (
    benchmark_face_basic_package,
    benchmark_face_package,
    benchmark_face_spatial_package,
    benchmark_full_set_package,
    benchmark_rgb_roi_preprocessor,
)
from nana_tracking.evaluation.standard import EvaluationStandard
from nana_tracking.evaluation.temporal import benchmark_temporal_refiner

__all__ = [
    "ConfidenceCalibration",
    "ConfidenceCurve",
    "EvaluationStandard",
    "ExpressionAblationConfig",
    "FailureSample",
    "ParameterExpressionModel",
    "benchmark_face_basic_package",
    "benchmark_face_package",
    "benchmark_face_spatial_package",
    "benchmark_full_set_package",
    "benchmark_rgb_roi_preprocessor",
    "benchmark_temporal_refiner",
    "evaluate",
    "fit_confidence_calibration",
    "render_failure_report",
    "run_expression_ablation_smoke",
    "validate_face_basic_acceptance",
]
