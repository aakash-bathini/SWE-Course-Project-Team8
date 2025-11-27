# src/metrics/phase2_adapter.py
"""
Phase 2 adapter for original Phase 1 metrics.
Converts web-based model data to EvalContext format for original metrics.
"""

import logging
import time
from typing import Dict, Any, Optional, Literal, Tuple
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
        # Ensure hf_data and gh_data are lists of dicts, not strings
        hf_data_raw = model_data.get("hf_data", [])
        gh_data_raw = model_data.get("gh_data", [])
        
        # Normalize hf_data to list of dicts
        hf_data = []
        if isinstance(hf_data_raw, str):
            try:
                import json
                parsed = json.loads(hf_data_raw)
                if isinstance(parsed, dict):
                    hf_data = [parsed]
                elif isinstance(parsed, list):
                    hf_data = [item for item in parsed if isinstance(item, dict)]
            except Exception:
                hf_data = []
        elif isinstance(hf_data_raw, dict):
            hf_data = [hf_data_raw]
        elif isinstance(hf_data_raw, list):
            for item in hf_data_raw:
                if isinstance(item, dict):
                    hf_data.append(item)
                elif isinstance(item, str):
                    try:
                        import json
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            hf_data.append(parsed)
                    except Exception:
                        pass
        
        # Normalize gh_data to list of dicts
        gh_data = []
        if isinstance(gh_data_raw, str):
            try:
                import json
                parsed = json.loads(gh_data_raw)
                if isinstance(parsed, dict):
                    gh_data = [parsed]
                elif isinstance(parsed, list):
                    gh_data = [item for item in parsed if isinstance(item, dict)]
            except Exception:
                gh_data = []
        elif isinstance(gh_data_raw, dict):
            gh_data = [gh_data_raw]
        elif isinstance(gh_data_raw, list):
            for item in gh_data_raw:
                if isinstance(item, dict):
                    gh_data.append(item)
                elif isinstance(item, str):
                    try:
                        import json
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            gh_data.append(parsed)
                    except Exception:
                        pass

        return EvalContext(url=url, category=category, hf_data=hf_data, gh_data=gh_data)
    except Exception as e:
        logger.error(f"Failed to create EvalContext: {e}")
        return EvalContext(url=model_data.get("url", ""))


async def calculate_phase2_metrics(model_data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate metrics using original Phase 1 metric functions
    Returns tuple of (metrics_dict, latencies_dict) where latencies are in seconds
    """
    try:
        # Create EvalContext from model data
        eval_context = create_eval_context_from_model_data(model_data)

        # Get all original metrics
        metrics = get_all_metrics()

        # Calculate each metric with latency measurement
        results: Dict[str, float] = {}
        latencies: Dict[str, float] = {}
        for metric_id, metric_fn in metrics:
            try:
                start_time = time.time()
                score = await metric_fn(eval_context)
                elapsed_time = time.time() - start_time

                # Store latency in seconds
                latencies[metric_id] = elapsed_time

                # Handle different return types
                if isinstance(score, dict):
                    # For metrics that return dictionaries (like size_score), use max value
                    if score:
                        # Filter to only numeric values to avoid type comparison errors
                        numeric_values = []
                        for v in score.values():
                            if isinstance(v, (int, float)):
                                numeric_values.append(float(v))
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
                # Still record latency even if metric failed
                if metric_id not in latencies:
                    latencies[metric_id] = 0.0

        return results, latencies
    except Exception as e:
        logger.error(f"Failed to calculate Phase 2 metrics: {e}")
        return {}, {}


def calculate_phase2_net_score(metrics: Dict[str, float]) -> Tuple[float, float]:
    """
    Calculate net score using original Phase 1 weights
    Returns tuple of (net_score, latency_in_seconds)
    """
    try:
        start_time = time.time()
        # Handle size_score which might be a dict
        processed_metrics = {}
        for key, value in metrics.items():
            if key == "size_score" and isinstance(value, dict):
                # Use the best device score for size
                processed_metrics[key] = max(value.values()) if value else 0.0
            else:
                processed_metrics[key] = value

        net_score = calculate_net_score(processed_metrics)
        elapsed_time = time.time() - start_time
        return net_score, elapsed_time
    except Exception as e:
        logger.error(f"Failed to calculate net score: {e}")
        return 0.0, 0.0


async def orchestrate_phase2_metrics(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate metric calculation for Phase 2 using original Phase 1 metrics
    """
    try:
        # Calculate all metrics
        metrics_dict, latencies_dict = await calculate_phase2_metrics(model_data)

        # Calculate net score
        net_score_value, net_score_latency = calculate_phase2_net_score(metrics_dict)

        return {
            "net_score": net_score_value,
            "sub_scores": metrics_dict,
            "confidence_intervals": {
                "net_score": {
                    "lower": max(0.0, net_score_value - 0.1),
                    "upper": min(1.0, net_score_value + 0.1),
                }
            },
            "latency_ms": int((sum(latencies_dict.values()) + net_score_latency) * 1000),
        }
    except Exception as e:
        logger.error(f"Phase 2 metric orchestration failed: {e}")
        return {"net_score": 0.0, "sub_scores": {}, "confidence_intervals": {}, "latency_ms": 0}
