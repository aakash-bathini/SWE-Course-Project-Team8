# src/scoring/net_score.py
from typing import Dict
from src.models.model_types import MetricRun, OrchestrationReport, ScoreBundle


def subscores_from_results(results: Dict[str, MetricRun]) -> Dict[str, float]:
    subs: Dict[str, float] = {}
    for k, r in results.items():
        v = r.value
        if isinstance(v, (int, float)):
            subs[k] = float(v)
        elif isinstance(v, dict):  # special case for size_score
            # use the maximum score across devices
            try:
                subs[k] = max(float(x) for x in v.values())
            except Exception:
                subs[k] = 0.0
        # ignore strings and other types
    return subs


def net_score(
    subscores: Dict[str, float],
    weights: Dict[str, float],
    clamp: bool = False,
) -> float:
    """
    Weighted average over present metrics only:
        score = sum(v_i * w_i) / sum(w_i for present i)
    - Missing/negative weights are treated as 0 (ignored).
    - If total weight is 0 -> score = 0.0
    Returns (score, latency_ms).
    """
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

    return score


def bundle_from_report(
    report: OrchestrationReport,
    weights: Dict[str, float],
    clamp: bool = False,
) -> ScoreBundle:
    """Convenience wrapper to build ScoreBundle from an OrchestrationReport."""
    subs = subscores_from_results(report.results)
    score = net_score(subs, weights, clamp=clamp)
    latency_ms = report.total_latency_ms
    return ScoreBundle(subscores=subs, net_score=score, net_score_latency_ms=latency_ms)
