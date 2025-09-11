#src/models/types.py
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional, Tuple, Literal

#shared types
MetricId = str
Category = Literal["MODEL", "DATASET", "CODE"]

@dataclass
class EvalContext:
    url: str
    category: Optional[Category] = None
    #Fields for caching once API's are implemented
    hf_data: Optional[dict] = None
    gh_data: Optional[dict] = None
    repo_dir: Optional[str] = None

#Metrics are ASYNC and receive an EvalContext
MetricFn = Callable[[EvalContext], Awaitable[float]]
MetricItem = Tuple[MetricId, MetricFn]

@dataclass
class MetricRun:
    name: MetricId
    value: Optional[float]      # None on failure
    latency_ms: int
    error: Optional[str] = None

@dataclass
class OrchestrationReport:
    results: Dict[MetricId, MetricRun]  # each metric's outcome
    total_latency_ms: int               # wall-clock for the whole run

@dataclass
class ScoreBundle:
    subscores: Dict[MetricId, float]    # successful metric values only
    net_score: float
    net_score_latency_ms: int