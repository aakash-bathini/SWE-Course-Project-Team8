from typing import List
from src.models.types import MetricFn, MetricItem
from . import license_check, metric_b, metric_c, metric_d, performance_metric, code_quality_metric, bus_factor_metric, dataset_quality, ramp_up_time, available_dataset_code

def get_all_metrics() -> List[MetricItem]:
    return [
        ("performance_claims", performance_metric.metric),
        ("bus_factor", bus_factor_metric.metric),
        ("code_quality", code_quality_metric.metric),
        ("license", license_check.metric),
        ("dataset_quality", dataset_quality.metric),
        ("ramp_up_time", ramp_up_time.metric),
        ("dataset_and_code_score", available_dataset_code.metric)
    ]
