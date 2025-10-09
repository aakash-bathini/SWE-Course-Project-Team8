#src/scoring/weights.py
#Weights do NOT need to add to 1.0, normalized in the final calculation
#Missing keys imply weight 0
def get_weights() -> dict[str, float]:
    # Based on the plan: Net_Score = 0.1 * size + 0.1 * license + 0.2 * ramp + 0.1 * bus + 0.2 * dataset+code_availability + 0.1 * dataset_quality + 0.1 * code_quality + 0.1 * performance
    weights = {
        "size_score": 0.1,
        "license": 0.1,
        "ramp_up_time": 0.2,
        "bus_factor": 0.1,
        "dataset_and_code_score": 0.2,
        "dataset_quality": 0.1,
        "code_quality": 0.1,
        "performance_claims": 0.1,
    }
    # Weights already sum to 1.0, no normalization needed
    return weights
