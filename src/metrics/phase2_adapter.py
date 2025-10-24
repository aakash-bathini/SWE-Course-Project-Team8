# src/metrics/phase2_adapter.py
"""
Phase 2 adapter for original Phase 1 metrics
Converts web-based model data to EvalContext format for original metrics
"""

import logging
from typing import Dict, Any, Optional, Literal
from src.models.model_types import EvalContext
from src.metrics.registry import get_all_metrics
from src.scoring.weights import calculate_net_score

logger = logging.getLogger(__name__)


def create_eval_context_from_model_data(model_data: Dict[str, Any]) -> EvalContext:
    """
    Convert Phase 2 model data to EvalContext for original metrics
    """
    try:
        # Extract URL from model data
        url = model_data.get("url", "")

        # Determine category
        category: Optional[Literal["MODEL", "DATASET", "CODE"]] = None
        if "huggingface.co" in url.lower():
            category = "MODEL"
        elif "github.com" in url.lower():
            category = "CODE"
        elif "datasets" in url.lower():
            category = "DATASET"

        # Extract HF and GH data if available
        hf_data = model_data.get("hf_data", [])
        gh_data = model_data.get("gh_data", [])

        return EvalContext(url=url, category=category, hf_data=hf_data, gh_data=gh_data)
    except Exception as e:
        logger.error(f"Failed to create EvalContext: {e}")
        return EvalContext(url=model_data.get("url", ""))


async def calculate_phase2_metrics(model_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate metrics using original Phase 1 metric functions
    """
    try:
        # Create EvalContext from model data
        eval_context = create_eval_context_from_model_data(model_data)

        # Get all original metrics
        metrics = get_all_metrics()

        # Calculate each metric
        results: Dict[str, float] = {}
        for metric_id, metric_fn in metrics:
            try:
                score = await metric_fn(eval_context)
                # Handle different return types
                if isinstance(score, dict):
                    # For metrics that return dictionaries (like size_score), use max value
                    if score:
                        # Filter to only numeric values to avoid type comparison errors
                        numeric_values = [
                            float(v) for v in score.values() if isinstance(v, (int, float))
                        ]
                        results[metric_id] = max(numeric_values) if numeric_values else 0.0
                    else:
                        results[metric_id] = 0.0
                elif isinstance(score, (int, float)):
                    results[metric_id] = float(score)
                else:
                    results[metric_id] = 0.0
            except Exception as e:
                logger.error(f"Failed to calculate {metric_id}: {e}")
                results[metric_id] = 0.0

        return results
    except Exception as e:
        logger.error(f"Failed to calculate Phase 2 metrics: {e}")
        return {}


def calculate_phase2_net_score(metrics: Dict[str, float]) -> float:
    """
    Calculate net score using original Phase 1 weights
    """
    try:
        # Handle size_score which might be a dict
        processed_metrics = {}
        for key, value in metrics.items():
            if key == "size_score" and isinstance(value, dict):
                # Use the best device score for size
                processed_metrics[key] = max(value.values()) if value else 0.0
            else:
                processed_metrics[key] = value

        return calculate_net_score(processed_metrics)
    except Exception as e:
        logger.error(f"Failed to calculate net score: {e}")
        return 0.0


async def orchestrate_phase2_metrics(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate metric calculation for Phase 2 using original Phase 1 metrics
    """
    try:
        # Calculate all metrics
        metrics = await calculate_phase2_metrics(model_data)

        # Calculate net score
        net_score = calculate_phase2_net_score(metrics)

        return {
            "net_score": net_score,
            "sub_scores": metrics,
            "confidence_intervals": {
                "net_score": {"lower": max(0, net_score - 0.1), "upper": min(1, net_score + 0.1)}
            },
            "latency_ms": 150,  # Mock latency for now
        }
    except Exception as e:
        logger.error(f"Phase 2 metric orchestration failed: {e}")
        return {"net_score": 0.0, "sub_scores": {}, "confidence_intervals": {}, "latency_ms": 0}
