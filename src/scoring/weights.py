#src/scoring/weights.py
#Weights do NOT need to add to 1.0, normalized in the final calculation
#Missing keys imply weight 0
def get_weights() -> dict[str, float]:
    weights = {
        "ramp_up_time": 2.0,
        "bus_factor": 2.0,
        "performance_claims": 1.5,
        "license": 1.5,
        "size_score": 1.5,
        "dataset_and_code_score": 1.0,
        "dataset_quality": 1.0,
        "code_quality": 0.5,
    }
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}
