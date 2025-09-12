#src/scoring/net_score.py
from time import perf_counter
from typing import Dict, Tuple
from src.models.types import MetricRun, OrchestrationReport, ScoreBundle

def subscores_from_results(results: Dict[str, MetricRun]) -> Dict[str, float]:
    return {k: r.value for k, r in results.items() if r.value is not None}

def net_score(
    subscores: Dict[str, float],
    weights: Dict[str, float],
    clamp: bool = False,
) -> Tuple[float, int]:
    """
    Weighted average over present metrics only:
        score = sum(v_i * w_i) / sum(w_i for present i)
    - Missing/negative weights are treated as 0 (ignored).
    - If total weight is 0 -> score = 0.0
    Returns (score, latency_ms).
    """
    t0 = perf_counter()
    total = 0.0
    total_w = 0.0

    for k, v in subscores.items():
        w = weights.get(k, 0.0)
        if w > 0:
            total += v * w
            total_w += w

    score = (total / total_w) if total_w > 0 else 0.0
    if clamp:
        score = 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)

    return score, int((perf_counter() - t0) * 1000)

def bundle_from_report(
    report: OrchestrationReport,
    weights: Dict[str, float],
    clamp: bool = False,
) -> ScoreBundle:
    """Convenience wrapper to build ScoreBundle from an OrchestrationReport."""
    subs = subscores_from_results(report.results)
    score, latency_ms = net_score(subs, weights, clamp=clamp)
    return ScoreBundle(subscores=subs, net_score=score, net_score_latency_ms=latency_ms)
