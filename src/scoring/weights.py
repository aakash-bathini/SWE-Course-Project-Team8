#src/scoring/weights.py
#Weights do NOT need to add to 1.0, normalized in the final calculation
#Missing keys imply weight 0
DEFAULT_WEIGHTS = {
    "metric_a": 3.0,
    "metric_b": 1.0,
    "metric_c": 2.0,
    "metric_d": 2.0,
}

def get_weights() -> dict[str, float]:
    return DEFAULT_WEIGHTS
