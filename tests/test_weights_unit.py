"""
Unit tests for scoring weights and net score calculation to increase coverage.
"""

from src.scoring.weights import get_weights, calculate_net_score


def test_get_weights_sum_and_nonnegative():
    weights = get_weights()
    assert isinstance(weights, dict)
    assert weights  # not empty
    # Weights should sum to approximately 1.0 per project plan comment
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9
    # All weights should be non-negative
    assert all(v >= 0 for v in weights.values())


def test_calculate_net_score_ignores_missing_and_negative_values():
    # Include a negative value and an unknown metric to ensure they are ignored
    metrics = {
        "size_score": 1.0,
        "license": 0.5,
        "ramp_up_time": 0.0,
        "unknown_metric": 1.0,  # should be ignored
        "reviewedness": -1.0,  # not in weights; even if it were, negative signals ignore
    }
    score = calculate_net_score(metrics)
    # Expected weighted average over included metrics only
    included = {
        "size_score": 1.0,
        "license": 0.5,
        "ramp_up_time": 0.0,
    }
    w = get_weights()
    expected = (
        included["size_score"] * w["size_score"]
        + included["license"] * w["license"]
        + included["ramp_up_time"] * w["ramp_up_time"]
    ) / (w["size_score"] + w["license"] + w["ramp_up_time"])
    assert abs(score - expected) < 1e-12


