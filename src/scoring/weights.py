# src/scoring/weights.py
# Weights do NOT need to add to 1.0, normalized in the final calculation
# Missing keys imply weight 0
def get_weights() -> dict[str, float]:
    """
    Net score weights. Values do not need to sum to 1.0; calculation normalizes by total weight.
    Includes Phase 1 metrics and Phase 2 metrics (reproducibility, reviewedness, tree_score).
    """
    weights = {
        # Phase 1 metrics
        "size_score": 0.1,
        "license": 0.1,
        "ramp_up_time": 0.2,
        "bus_factor": 0.1,
        "dataset_and_code_score": 0.2,
        "dataset_quality": 0.1,
        "code_quality": 0.1,
        "performance_claims": 0.1,
        # Phase 2 metrics
        "reproducibility": 0.1,
        "reviewedness": 0.1,
        "tree_score": 0.1,
    }
    return weights


def calculate_net_score(metrics: dict[str, float]) -> float:
    """
    Calculate net score using the defined weights

    Note: Metrics with invalid values (e.g., -1.0 for reviewedness when no GitHub repo)
    are automatically excluded either because they are not present in weights or are < 0.
    """
    weights = get_weights()
    total_score = 0.0
    total_weight = 0.0

    for metric_name, weight in weights.items():
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            # Skip invalid values (e.g., -1.0 from reviewedness when no GitHub repo)
            if isinstance(metric_value, (int, float)) and metric_value >= 0.0:
                total_score += float(metric_value) * weight
                total_weight += weight

    if total_weight == 0:
        return 0.0

    return total_score / total_weight
