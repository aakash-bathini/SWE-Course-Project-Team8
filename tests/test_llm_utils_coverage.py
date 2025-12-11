"""
Tests for src.metrics.llm_utils to ensure provider-agnostic LLM helpers
and JSON extraction behave as expected.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import llm_utils


class TestCachedLLMChat:
    """Validate the caching helper that wraps external LLM providers."""

    def setup_method(self):
        with llm_utils._CACHE_LOCK:
            llm_utils._LLM_RESPONSE_CACHE.clear()

    def test_cached_llm_chat_no_providers(self, monkeypatch):
        """Return None when no providers are configured."""
        monkeypatch.setattr(llm_utils, "_GEMINI_KEY", "")
        monkeypatch.setattr(llm_utils, "_PURDUE_KEY", "")

        result = llm_utils.cached_llm_chat(
            system_prompt="sys",
            user_prompt="user",
            cache_scope="test",
        )

        assert result is None

    def test_cached_llm_chat_uses_gemini_cache(self, monkeypatch):
        """Ensure Gemini is invoked once and subsequent calls hit cache."""
        monkeypatch.setattr(llm_utils, "_GEMINI_KEY", "fake-key")
        monkeypatch.setattr(llm_utils, "_PURDUE_KEY", "")

        call_count = {"value": 0}

        def fake_gemini(*_args, **_kwargs):
            call_count["value"] += 1
            return '{"ok": true}'

        monkeypatch.setattr(llm_utils, "_invoke_gemini", fake_gemini)

        first = llm_utils.cached_llm_chat(
            system_prompt="sys",
            user_prompt="user",
            cache_scope="cache-scope",
        )
        second = llm_utils.cached_llm_chat(
            system_prompt="sys",
            user_prompt="user",
            cache_scope="cache-scope",
        )

        assert first == second == '{"ok": true}'
        assert call_count["value"] == 1

    def test_cached_llm_chat_falls_back_to_purdue(self, monkeypatch):
        """If the first provider fails, use the next configured provider."""
        monkeypatch.setattr(llm_utils, "_GEMINI_KEY", "fake-key")
        monkeypatch.setattr(llm_utils, "_PURDUE_KEY", "purdue-key")

        monkeypatch.setattr(llm_utils, "_invoke_gemini", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(llm_utils, "_invoke_purdue", lambda *_args, **_kwargs: "fallback")

        result = llm_utils.cached_llm_chat(
            system_prompt="sys",
            user_prompt="user",
            cache_scope="fallback-scope",
        )

        assert result == "fallback"


class TestInvokePurdue:
    """Directly test Purdue helper to ensure response parsing."""

    def test_invoke_purdue_success(self, monkeypatch):
        monkeypatch.setattr(llm_utils, "_PURDUE_KEY", "purdue-key")

        class DummyResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "result"}}]}

        monkeypatch.setattr(llm_utils, "requests", type("Req", (), {"post": lambda *args, **kwargs: DummyResponse()}))

        result = llm_utils._invoke_purdue("sys", "user", 100, 0.1)

        assert result == "result"


class TestExtractJson:
    """Ensure JSON extraction trims code fences and whitespace."""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ("```json\n{\"a\": 1}\n```", '{"a": 1}'),
            ("No json here", None),
            ("prefix {\"nested\": true} suffix", '{"nested": true}'),
        ],
    )
    def test_extract_json_from_llm(self, payload, expected):
        assert llm_utils.extract_json_from_llm(payload) == expected



