from typing import List
from src.models.model_types import MetricItem
from . import (
    size,
    license_check,
    performance_metric,
    code_quality_metric,
    bus_factor_metric,
    dataset_quality,
    ramp_up_time,
    available_dataset_code,
    reproducibility,
    reviewedness,
    treescore,
)


def get_all_metrics() -> List[MetricItem]:
    return [
        ("ramp_up_time", ramp_up_time.metric),
        ("bus_factor", bus_factor_metric.metric),
        ("performance_claims", performance_metric.metric),
        ("license", license_check.metric),
        ("size_score", size.metric),
        ("dataset_and_code_score", available_dataset_code.metric),
        ("dataset_quality", dataset_quality.metric),
        ("code_quality", code_quality_metric.metric),
        ("reproducibility", reproducibility.metric),
        ("reviewedness", reviewedness.metric),
        ("tree_score", treescore.metric),
    ]
