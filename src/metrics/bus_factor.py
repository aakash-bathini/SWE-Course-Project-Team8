"""
Bus factor calculation for Hugging Face models and datasets.
Measures knowledge distribution among contributors based on pull request activity.
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import asyncio
from src.models.types import EvalContext
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import requests
from urllib.parse import urlparse

from src.orchestration.error_handling import NetworkError, ProcessingError, ErrorContext


logger = logging.getLogger(__name__)


def calculate_bus_factor(self, url: str) -> Tuple[float, int]:
    """
    Calculate bus factor score for a Hugging Face model/dataset.
    
    Args:
        url: Hugging Face model or dataset URL
        
    Returns:
        Tuple of (bus_factor_score, calculation_latency_ms)
        
    Raises:
        NetworkError: If unable to fetch contributor data
        ProcessingError: If unable to calculate bus factor
    """
    start_time = time.time()
    
    try:
        # Extract repository info from Hugging Face URL
        repo_info = self._parse_huggingface_url(url)
        if not repo_info:
            logger.warning(f"Could not parse Hugging Face URL: {url}")
            return 0.0, int((time.time() - start_time) * 1000)
        
        # Get contributor statistics
        contributors = self._get_contributors(repo_info)
        
        if not contributors:
            logger.warning(f"No contributors found for {url}")
            return 0.0, int((time.time() - start_time) * 1000)
        
        # Calculate bus factor score
        score = self._calculate_distribution_score(contributors)
        
        latency = int((time.time() - start_time) * 1000)
        
        logger.info(f"Bus factor calculated for {url}: {score:.3f} "
                    f"({len(contributors)} contributors, {latency}ms)")
        
        return score, latency
        
    except Exception as e:
        latency = int((time.time() - start_time) * 1000)
        context = ErrorContext(url=url, operation="bus_factor_calculation")
        
        if isinstance(e, (NetworkError, ProcessingError)):
            raise e
        else:
            raise ProcessingError(f"Bus factor calculation failed: {str(e)}", context)


def _calculate_distribution_score(self, contributors: List[ContributorStats]) -> float:
    """
    Calculate bus factor score based on contributor distribution.
    
    Uses modified HHI approach where:
    - 0.0 = single contributor (maximum concentration)
    - 1.0 = perfectly equal distribution (minimum concentration)
    
    Args:
        contributors: List of contributor statistics
        
    Returns:
        Bus factor score between 0.0 and 1.0
    """
    if not contributors:
        return 0.0
    
    if len(contributors) == 1:
        return 0.0  # Single contributor = worst bus factor
    
    # Calculate total activity across all contributors
    total_activity = sum(c.activity_score for c in contributors)
    
    if total_activity == 0:
        return 0.0
    
    # Calculate concentration using HHI
    hhi = 0.0
    for contributor in contributors:
        share = contributor.activity_score / total_activity
        hhi += share * share
    
    # Convert HHI to bus factor score
    # HHI ranges from 1/n (perfect distribution) to 1 (monopoly)
    # We want to invert this so higher diversity = higher score
    
    num_contributors = len(contributors)
    min_hhi = 1.0 / num_contributors  # Perfect equal distribution
    max_hhi = 1.0  # Single contributor dominance
    
    # Normalize HHI to 0-1 scale and invert
    if max_hhi == min_hhi:
        normalized_score = 1.0
    else:
        normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi)
        normalized_score = 1.0 - normalized_hhi
    
    # Apply scaling based on number of contributors
    # More contributors generally means better bus factor
    contributor_bonus = min(0.2, (num_contributors - 1) * 0.05)
    final_score = min(1.0, normalized_score + contributor_bonus)
    
    logger.debug(f"Bus factor calculation: {num_contributors} contributors, "
                f"HHI={hhi:.3f}, score={final_score:.3f}")
    
    return final_score