# src/models/model_types.py
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional, Tuple, Literal, List, Union, Any

# shared types
MetricId = str
Category = Literal["MODEL", "DATASET", "CODE"]


@dataclass
class EvalContext:
    url: str
    category: Optional[Category] = None
    hf_data: Optional[List[Dict[str, Any]]] = None
    gh_data: Optional[List[Dict[str, Any]]] = None


# Metrics are ASYNC and receive an EvalContext
MetricFn = Callable[[EvalContext], Awaitable[float]]
MetricItem = Tuple[MetricId, MetricFn]


@dataclass
class MetricRun:
    name: MetricId
    # value can be:
    # - float (normal metrics)
    # - dict[str, float] (size_score)
    # - str (legacy best-device string)
    value: Optional[Union[float, Dict[str, float], str]]
    latency_ms: int
    error: Optional[str] = None


@dataclass
class OrchestrationReport:
    results: Dict[MetricId, MetricRun]
    total_latency_ms: int


@dataclass
class ScoreBundle:
    subscores: Dict[MetricId, float]
    net_score: float
    net_score_latency_ms: int
