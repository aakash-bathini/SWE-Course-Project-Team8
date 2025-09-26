import asyncio
from typing import Dict, List, Tuple

import pytest

from src.models.types import EvalContext
from src.metrics import license_check as lc
from src.config_parsers_nlp import spdx
from src.config_parsers_nlp.thresholds import (
    LICENSE_WHITELIST,
    LICENSE_BLACKLIST,
    LICENSE_AMBIGUOUS_03,
    LICENSE_AMBIGUOUS_07,
    LICENSE_ALIASES
)

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

# ...existing code...
def representative_ids() -> List[Tuple[str, str, float]]:
    picks: List[Tuple[str, str, float]] = []

    def _one(s):
        return next(iter(s))

    if LICENSE_WHITELIST:
        picks.append(("whitelist", _one(LICENSE_WHITELIST), 1.0))
    if LICENSE_BLACKLIST:
        picks.append(("blacklist", _one(LICENSE_BLACKLIST), 0.0))
    if LICENSE_AMBIGUOUS_03:
        picks.append(("ambiguous_03", _one(LICENSE_AMBIGUOUS_03), 0.3))
    if LICENSE_AMBIGUOUS_07:
        picks.append(("ambiguous_07", _one(LICENSE_AMBIGUOUS_07), 0.7))
    # Always include an unknown to exercise default
    picks.append(("unknown", "Unknown-NonSPDX-XYZ", 0.0))
    return picks
# ...existing code...

@pytest.mark.parametrize("label,spdx_id,expected", representative_ids())
def test_metric_scores_by_spdx_category_prints_rationale(monkeypatch, label: str, spdx_id: str, expected: float):
    # Force the extractor to return the selected SPDX id from README (no LICENSE file)
    monkeypatch.setattr(lc, "extract_license_evidence", lambda readme, lic: ("README", [spdx_id], [], []))

    # Prepare a minimal context with README text only
    ctx = EvalContext(
        url="https://github.com/example/repo",
        gh_data=[{"readme_text": f"README mentions {spdx_id}", "doc_texts": {}, "license_spdx": None}],
    )

    # Run metric and assert score matches expected for this category
    score = run(lc.metric(ctx))
    assert score == expected

    # Also print the rationale from the classifier for visibility
    cls_score, rationale = spdx.classify_license(spdx_id)
    print(f"[{label}] id={spdx_id} -> metric_score={score}, classify_score={cls_score}, rationale={rationale}")

def test_metric_prefers_license_file_over_readme_and_prints(monkeypatch):
    # Extractor should see the LICENSE text as primary source
    captured = {}
    def fake_extract(readme, lic):
        captured["readme"] = readme
        captured["license"] = lic
        return ("LICENSE", ["MIT"], [], [])
    monkeypatch.setattr(lc, "extract_license_evidence", fake_extract)

    ctx = EvalContext(
        url="https://github.com/example/repo",
        gh_data=[{"readme_text": "README mentions GPL-3.0-only", "doc_texts": {"LICENSE": "MIT license text"}, "license_spdx": None}],
    )

    score = run(lc.metric(ctx))
    # MIT is commonly whitelisted; accept either 1.0 (preferred) or whatever thresholds define
    cls_score, rationale = spdx.classify_license("MIT")
    print(f"[prefer_license] id=MIT -> metric_score={score}, classify_score={cls_score}, rationale={rationale}")
    assert score == cls_score
    assert captured["license"] == "MIT license text"

def test_metric_uses_readme_when_no_license_file_and_prints(monkeypatch):
    # No LICENSE in doc_texts; rely on README
    monkeypatch.setattr(lc, "extract_license_evidence", lambda readme, lic: ("README", ["Apache-2.0"], [], []))

    ctx = EvalContext(
        url="https://github.com/example/repo",
        gh_data=[{"readme_text": "README mentions Apache-2.0", "doc_texts": {}, "license_spdx": None}],
    )

    score = run(lc.metric(ctx))
    cls_score, rationale = spdx.classify_license("Apache-2.0")
    print(f"[readme_only] id=Apache-2.0 -> metric_score={score}, classify_score={cls_score}, rationale={rationale}")
    assert score == cls_score

def test_metric_falls_back_to_github_spdx_when_no_evidence_and_prints():
    # No evidence from extractor; use GitHub's SPDX
    ctx = EvalContext(
        url="https://github.com/example/repo",
        gh_data=[{"readme_text": "", "doc_texts": {}, "license_spdx": "BSD-3-Clause"}],
    )

    score = run(lc.metric(ctx))
    cls_score, rationale = spdx.classify_license("BSD-3-Clause")
    print(f"[gh_fallback] id=BSD-3-Clause -> metric_score={score}, classify_score={cls_score}, rationale={rationale}")
    assert score == cls_score

def test_metric_extractor_overrides_gh_spdx_when_both_present_and_prints(monkeypatch):
    # If extractor returns IDs, ignore gh_spdx fallback path
    monkeypatch.setattr(lc, "extract_license_evidence", lambda r, l: ("LICENSE", ["GPL-3.0-only"], [], []))

    ctx = EvalContext(
        url="https://github.com/example/repo",
        gh_data=[{"readme_text": "", "doc_texts": {"LICENSE": "GPL3 text"}, "license_spdx": "MIT"}],
    )

    score = run(lc.metric(ctx))
    cls_score, rationale = spdx.classify_license("GPL-3.0-only")
    print(f"[override_gh] id=GPL-3.0-only -> metric_score={score}, classify_score={cls_score}, rationale={rationale}")
    assert score == cls_score

def test_metric_unknown_license_and_prints(monkeypatch):
    # Unknown ID should classify to 0.0 with "Unknown license" rationale
    unknown_id = "Unknown-NonSPDX-XYZ"
    monkeypatch.setattr(lc, "extract_license_evidence", lambda r, l: ("README", [unknown_id], [], []))

    ctx = EvalContext(
        url="https://github.com/example/repo",
        gh_data=[{"readme_text": f"README mentions {unknown_id}", "doc_texts": {}, "license_spdx": None}],
    )

    score = run(lc.metric(ctx))
    cls_score, rationale = spdx.classify_license(unknown_id)
    print(f"[unknown] id={unknown_id} -> metric_score={score}, classify_score={cls_score}, rationale={rationale}")
    assert score == cls_score == 0.0
    assert "Unknown" in rationale

# -------- selection helper edge cases (also print selection results) --------

def test_select_license_text_exact_and_case_insensitive_prints():
    doc_texts = {
        "docs/license.md": "L1",
        "COPYING": "L2",
        "notes.txt": "N",
    }
    chosen = lc._select_license_text(doc_texts)
    print(f"[select_exact_case] chosen={chosen!r}")
    assert chosen == "L1"

def test_select_license_text_loose_basename_and_nested_prints():
    doc_texts = {
        "third_party/dep/LICENSE-APACHE.txt": "APACHE",
        "third_party/dep/readme.txt": "R",
    }
    chosen = lc._select_license_text(doc_texts)
    print(f"[select_loose_nested] chosen={chosen!r}")
    assert chosen == "APACHE"

def test_select_license_text_preference_order_prints():
    # LICENSE should be preferred over COPYING
    doc_texts = {"COPYING": "C", "LICENSE": "L"}
    chosen = lc._select_license_text(doc_texts)
    print(f"[select_preference] chosen={chosen!r}")
    assert chosen == "L"

@pytest.mark.parametrize(
    "doc_texts,expected_is_none,label",
    [
        ({}, True, "empty"),
        ({"notes.txt": "N"}, True, "no_license_files"),
        ({"COPYRIGHT": "Copyright text"}, False, "copyright_accepted"),
    ],
)
def test_select_license_text_edge_cases_prints(doc_texts, expected_is_none, label):
    chosen = lc._select_license_text(doc_texts)
    print(f"[select_edge:{label}] chosen={chosen!r}")
    assert (chosen is None) == expected_is_none

def test_metric_no_github_data_returns_zero_and_prints():
    ctx = EvalContext(url="https://github.com/example/repo", gh_data=None)
    score = run(lc.metric(ctx))
    print(f"[no_gh_data] metric_score={score}")
    assert score == 0.0