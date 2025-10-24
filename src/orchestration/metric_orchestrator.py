# src/orchestration/metric_orchestrator.py
"""
Phase 2 metric orchestrator using original Phase 1 metrics
"""

import logging
from typing import Dict, Any
from src.metrics.phase2_adapter import orchestrate_phase2_metrics

logger = logging.getLogger(__name__)


async def orchestrate(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate metric calculation for Phase 2 using original Phase 1 metrics
    """
    try:
        return await orchestrate_phase2_metrics(model_data)
    except Exception as e:
        logger.error(f"Metric orchestration failed: {e}")
        return {"net_score": 0.0, "sub_scores": {}, "confidence_intervals": {}, "latency_ms": 0}
