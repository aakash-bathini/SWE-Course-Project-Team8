#src/scoring/weights.py
#Weights do NOT need to add to 1.0, normalized in the final calculation
#Missing keys imply weight 0
DEFAULT_WEIGHTS = {
    # Highest priority
    "license": 2.0,               # must be LGPL-compatible
    "ramp_up_time": 2.0,          # ease of adoption

    # Important but secondary
    "dataset_and_code_score": 1.5,# availability of dataset + runnable code
    "bus_factor": 1.5,            # maintainer redundancy
    "code_quality": 1.5,          # style, tests, maintainability
    "performance_claims": 1.0,    # evidence of benchmarking/performance

    # Useful but lower priority
    "size_score": 1.0,            # deployability on devices
    "dataset_quality": 0.5        # dataset documentation/quality
}

def get_weights() -> dict[str, float]:
    return DEFAULT_WEIGHTS
