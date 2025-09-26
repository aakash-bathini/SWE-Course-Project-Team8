from typing import List
from src.models.types import MetricFn, MetricItem
from . import license_check, metric_b, metric_c, metric_d, performance_metric, code_quality_metric, bus_factor_metric, dataset_quality, ramp_up_time, available_dataset_code

def get_all_metrics() -> List[MetricItem]:
    return [
        ("performance", performance_metric.metric),
        ("bus factor", bus_factor_metric.metric),
        ("code quality", code_quality_metric.metric),
        ("license check", license_check.metric),
        ("dataset quality", dataset_quality.metric),
        ("ramp up time", ramp_up_time.metric),
        ("available dataset/code", available_dataset_code.metric),
        ("metric_b", metric_b.metric),
        ("metric_c", metric_c.metric),
        ("metric_d", metric_d.metric),
    ]
