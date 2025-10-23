"""
SPDX helper data and functions for license parsing
"""

from typing import Tuple
from src.config_parsers_nlp.thresholds import (
    LICENSE_WHITELIST,
    LICENSE_BLACKLIST,
    LICENSE_AMBIGUOUS_03,
    LICENSE_AMBIGUOUS_07,
    LICENSE_ALIASES,
)


def normalize_license(text: str) -> str:
    """
    Normalize license text for SPDX matching
    """
    norm = text.strip().lower()
    for alias, spdx in LICENSE_ALIASES.items():
        if alias in norm:
            return spdx
    return norm


def classify_license(text: str) -> Tuple[float, str]:
    """
    Classify license text into whitelist/blacklist/ambiguous categories
    Returns (score, category)
    """
    norm = normalize_license(text)
    if norm in LICENSE_WHITELIST:
        return (1.0, f"Explicit SPDX whitelist: {norm}")
    if norm in LICENSE_BLACKLIST:
        return (0.0, f"Explicit SPDX blacklist: {norm}")
    if norm in LICENSE_AMBIGUOUS_03:
        return (0.3, f"Ambiguous license (vague wording): {norm}")
    if norm in LICENSE_AMBIGUOUS_07:
        return (0.7, f"Ambiguous license (Family match): {norm}")
    return (0.0, f"Unknown license: {norm}")
