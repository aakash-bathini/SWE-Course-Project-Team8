from typing import List
from src.models.types import MetricFn, MetricItem
from . import metric_a, metric_b, metric_c, metric_d

def get_all_metrics() -> List[MetricItem]:
    return [
        ("metric_a", metric_a.metric),
        ("metric_b", metric_b.metric),
        ("metric_c", metric_c.metric),
        ("metric_d", metric_d.metric),
    ]
