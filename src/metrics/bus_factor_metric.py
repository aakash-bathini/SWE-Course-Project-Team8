import logging
from src.models.types import EvalContext


async def metric(ctx: EvalContext) -> float:
    """
    Compute a Bus Factor score [0,1] for a Hugging Face repo based on contributor distribution.
    
    Logic:
    - One sole contributor = 0.0
    - Two contributors, one dominates = ~0.25
    - Balanced contributions = higher score
    - Use a simple inverse concentration (like HHI or top-contributor share).
    """

    gh = ctx.gh_data[0]
    contributors: dict[str, int] = gh.get("contributors", {})

    if not contributors:
        logging.warning("No contributor data found; returning 0.")
        return 0.0

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
        f"Bus Factor Metric -> contributors={len(contributors)}, "
        f"top_share={top_share:.2f}, score={score:.2f}"
    )

    return float(round(score,2))