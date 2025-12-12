import os
import sys
import time

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import app  # noqa: E402


def test_is_dangerous_regex_detects_nested_quantifiers_and_large_ranges() -> None:
    assert app._is_dangerous_regex("(a+)+$") is True
    assert app._is_dangerous_regex("^(?:.+)+$") is True
    assert app._is_dangerous_regex("a{1,5000}") is True
    assert app._is_dangerous_regex("") is False
    assert app._is_dangerous_regex("^[a-z]+$") is False


def test_safe_eval_with_timeout_success_and_timeout() -> None:
    ok, res = app._safe_eval_with_timeout(lambda: 123, timeout_ms=50)
    assert ok is True
    assert res == 123

    ok, res = app._safe_eval_with_timeout(lambda: time.sleep(0.2), timeout_ms=10)
    assert ok is False
    assert res is None


def test_safe_name_match_exact_and_partial() -> None:
    pat_exact = app.re.compile(r"^Hello$")
    assert app._safe_name_match(pat_exact, "Hello", exact_match=True, raw_pattern=pat_exact.pattern) is True
    assert app._safe_name_match(pat_exact, "Hello!", exact_match=True, raw_pattern=pat_exact.pattern) is False

    pat_partial = app.re.compile(r"world", app.re.IGNORECASE)
    assert app._safe_name_match(pat_partial, "Hello World", exact_match=False, raw_pattern=pat_partial.pattern) is True


def test_safe_name_match_timeout_raises_http_exception() -> None:
    class SlowPattern:
        pattern = "slow"

        def fullmatch(self, _candidate: str):
            time.sleep(1.0)
            return True

        def search(self, _candidate: str):
            time.sleep(1.0)
            return True

    with pytest.raises(app.HTTPException) as exc:
        app._safe_name_match(SlowPattern(), "anything", exact_match=True, raw_pattern="(a+)+$")

    assert exc.value.status_code == 400


def test_safe_text_search_timeout_returns_false() -> None:
    class SlowSearch:
        pattern = "slowsearch"

        def search(self, _text: str):
            time.sleep(1.0)
            return True

    assert app._safe_text_search(SlowSearch(), "some text", raw_pattern="(a+)+$") is False


def test_regex_readme_cache_and_extractors() -> None:
    # cache get/set
    app._cache_regex_readme_text("a1", "hello")
    assert app._get_cached_regex_readme_text("a1") == "hello"
    assert app._get_cached_regex_readme_text("") is None
    app._cache_regex_readme_text("", "noop")  # no-op

    # extract from hf_data json string
    data_block = {
        "hf_data": '[{"readme_text": "hf readme"}]',
        "readme_text": "fallback readme",
    }
    assert app._extract_readme_from_data_block(data_block) == "hf readme"

    # ensure uses extract + caches
    app._REGEX_README_CACHE.clear()
    got = app._ensure_regex_readme_text("a2", data_block)
    assert got == "hf readme"
    assert app._get_cached_regex_readme_text("a2") == "hf readme"

    # ensure returns cached immediately (early return branch)
    assert app._ensure_regex_readme_text("a2", None) == "hf readme"


def test_ensure_regex_readme_text_scrapes_hf_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force scrape path: no readme in data block, huggingface URL, and scraper available.
    def fake_scrape(url: str):
        assert "huggingface.co" in url.lower()
        return {"readme_text": "scraped readme"}, None

    monkeypatch.setattr(app, "scrape_hf_url", fake_scrape, raising=False)
    app._REGEX_README_CACHE.clear()

    data_block = {"source_url": "https://huggingface.co/user/repo"}
    got = app._ensure_regex_readme_text("a3", data_block)
    assert got == "scraped readme"
    # Ensure we stored hf_data and cached it
    assert isinstance(data_block.get("hf_data"), list)
    assert data_block["hf_data"][0]["readme_text"] == "scraped readme"
    assert app._get_cached_regex_readme_text("a3") == "scraped readme"


def test_ensure_regex_readme_text_scrape_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_url: str):
        raise RuntimeError("nope")

    monkeypatch.setattr(app, "scrape_hf_url", boom, raising=False)
    app._REGEX_README_CACHE.clear()
    got = app._ensure_regex_readme_text("a4", {"url": "https://huggingface.co/user/repo"})
    assert got is None


def test_ensure_artifact_display_name_fallbacks() -> None:
    record = {"metadata": {"id": "x"}, "data": {"url": "https://example.com/foo"}}
    name = app._ensure_artifact_display_name(record)
    assert isinstance(name, str)
    assert record["metadata"].get("name")


def test_safe_helpers_empty_inputs() -> None:
    # candidate/text empty branches
    pat = app.re.compile("x")
    assert app._safe_name_match(pat, "", exact_match=False) is False
    assert app._safe_text_search(pat, "") is False
