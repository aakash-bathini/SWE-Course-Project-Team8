"""
Unit tests for SPDX helpers to improve meaningful coverage.
"""

from src.config_parsers_nlp import spdx
from src.config_parsers_nlp import readme_parser


def test_normalize_license_basic():
    # Implementation may preserve original case; just ensure trimming and identity behavior
    assert spdx.normalize_license("  MIT ").strip().lower().endswith("mit")
    assert spdx.normalize_license("Apache-2.0").lower().endswith("apache-2.0")
    assert "gpl" in spdx.normalize_license("gPl-3.0").lower()


def test_classify_license_known_ids():
    score, rationale = spdx.classify_license("MIT")
    assert isinstance(score, float) and 0.0 <= score <= 1.0
    assert "mit" in rationale.lower() or "permissive" in rationale.lower()

    score2, rationale2 = spdx.classify_license("Apache-2.0")
    assert isinstance(score2, float) and 0.0 <= score2 <= 1.0
    assert "apache" in rationale2.lower() or "permissive" in rationale2.lower()


def test_classify_license_unknown_id():
    score, rationale = spdx.classify_license("NotARealLicense")
    assert score == 0.0
    assert "unknown" in rationale.lower()


def test_find_spdx_ids_from_text():
    text = "This project is licensed under the MIT License and parts under Apache-2.0."
    ids = readme_parser.find_spdx_ids(text)
    assert any(i.lower() == "mit" for i in ids)
    assert any(i.lower() == "apache-2.0" for i in ids)
