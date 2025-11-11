"""
Unit tests for size metric small helpers to raise coverage in a meaningful way.
"""

from src.metrics.size import _best_device


def test_best_device_tie_prefers_pi():
    # Tie across devices should pick raspberry_pi by design
    scores = {
        "raspberry_pi": 0.8,
        "jetson_nano": 0.8,
        "desktop_pc": 0.8,
        "aws_server": 0.8,
    }
    assert _best_device(scores) == "raspberry_pi"
