# src/scoring/net_score.py
"""
Net score calculation for Phase 2 using original Phase 1 weights
"""

import logging
from typing import Dict
from src.scoring.weights import calculate_net_score

logger = logging.getLogger(__name__)


def net_score(metrics: Dict[str, float]) -> float:
    """
    Calculate net score from metrics using original Phase 1 weights
    """
    try:
        return calculate_net_score(metrics)
    except Exception as e:
        logger.error(f"Net score calculation failed: {e}")
        return 0.0
