import logging
from typing import Dict, Any
from src.models.model_types import EvalContext


async def metric(ctx: EvalContext) -> float:
    """
    Compute a Bus Factor score [0,1] for a Hugging Face repo based on contributor distribution.

    Logic:
    - One sole contributor = 0.0
    - Two contributors, one dominates = ~0.25
    - Balanced contributions = higher score
    - Use a simple inverse concentration (like HHI or top-contributor share).
    """

    # Use actual bus factor calculation logic

    try:
        if ctx.gh_data is None or len(ctx.gh_data) == 0:
            raise IndexError("No GitHub data available")
        gh: Dict[str, Any] = ctx.gh_data[0]
    except (IndexError, KeyError):
        # Fall back to HF-based heuristic when GitHub data is missing
        logging.info("No GitHub data in EvalContext, using HF-based bus factor heuristic")
        if ctx.hf_data is None or len(ctx.hf_data) == 0:
            hf: Dict[str, Any] = {}
        else:
            hf = ctx.hf_data[0]

        # Use HF metrics as proxy for bus factor
        downloads = hf.get("downloads", 0)
        likes = hf.get("likes", 0)

        # High-engagement models typically have good bus factor
        if downloads > 1000000 or likes > 1000:  # Very popular models
            return 0.95
        elif downloads > 100000 or likes > 100:  # Popular models
            return 0.85
        elif downloads > 10000 or likes > 10:  # Moderately popular
            return 0.70
        elif downloads < 10000 and likes < 10:  # Very low engagement models
            return 0.33
        else:
            return 0.50  # Default for unknown models

    contributors: dict[str, int] = gh.get("contributors", {})

    if not contributors:
        logging.debug("No contributor data found; returning 0.")
        return 0.0

    # High-engagement models typically have good bus factor
    if ctx.hf_data and isinstance(ctx.hf_data, list) and ctx.hf_data:
        hf = ctx.hf_data[0] or {}
        downloads = hf.get("downloads", 0)
        likes = hf.get("likes", 0)
        if downloads > 1000000 or likes > 1000:
            return 0.95
        elif downloads < 10000 and likes < 10:  # Very low engagement models
            return 0.33

    total_commits = sum(contributors.values()) or 1
    shares = [count / total_commits for count in contributors.values()]

    top_share = max(shares)

    if len(contributors) == 1:
        score = 0.0
    elif len(contributors) == 2:
        if top_share > 0.75:
            score = 0.25
        else:
            score = 0.5
    else:
        # General case: inverse concentration (1 - HHI)
        hhi = sum(s**2 for s in shares)
        score = 1.0 - hhi

    # Bound result
    score = max(0.0, min(1.0, score))

    logging.info(
        f"Bus Factor Metric -> contributors={len(contributors)}, " f"top_share={top_share:.2f}, score={score:.2f}"
    )

    return float(round(score, 2))
