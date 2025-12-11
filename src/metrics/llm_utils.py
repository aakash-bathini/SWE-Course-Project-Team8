import hashlib
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_CACHE_CAP = 64
_LLM_RESPONSE_CACHE: "OrderedDict[str, str]" = OrderedDict()

_DEFAULT_KEYWORD_HINTS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "benchmark",
    "dataset",
    "rouge",
    "bleu",
    "latency",
    "throughput",
    "evaluation",
    "claims",
    "evidence",
)

_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
_GEMINI_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
_PURDUE_KEY = os.getenv("GEN_AI_STUDIO_API_KEY")


def reduce_readme_for_llm(readme_text: str, max_chars: int = 4000) -> str:
    """
    Trim a README down to the most signal-rich sections before sending to an LLM.
    Keeps the first chunk, keyword-rich lines, and the tail of the document.
    """
    if not readme_text:
        return ""

    text = readme_text.strip()
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()
    intro = "\n".join(lines[:80])
    outro = "\n".join(lines[-40:]) if len(lines) > 40 else ""

    keyword_lines = []
    for line in lines:
        lower = line.lower()
        if any(token in lower for token in _DEFAULT_KEYWORD_HINTS):
            keyword_lines.append(line)
        if len(keyword_lines) >= 120:
            break

    combined_sections = [section for section in (intro, "\n".join(keyword_lines), outro) if section]
    condensed = "\n\n---\n\n".join(combined_sections)

    if not condensed:
        return text[:max_chars]

    return condensed[:max_chars]


def _cache_get(cache_key: str) -> Optional[str]:
    with _CACHE_LOCK:
        if cache_key in _LLM_RESPONSE_CACHE:
            _LLM_RESPONSE_CACHE.move_to_end(cache_key)
            return _LLM_RESPONSE_CACHE[cache_key]
    return None


def _cache_set(cache_key: str, value: str) -> None:
    with _CACHE_LOCK:
        _LLM_RESPONSE_CACHE[cache_key] = value
        _LLM_RESPONSE_CACHE.move_to_end(cache_key)
        if len(_LLM_RESPONSE_CACHE) > _CACHE_CAP:
            _LLM_RESPONSE_CACHE.popitem(last=False)


def cached_llm_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    cache_scope: str,
    max_tokens: int = 384,
    temperature: float = 0.15,
) -> Optional[str]:
    """
    Invoke an external LLM provider (Gemini or Purdue GenAI) with a small in-memory cache so
    repeated prompts (especially from concurrent /rate calls) do not hammer the provider.
    """

    digest = hashlib.sha256(
        f"{cache_scope}:{system_prompt}:{user_prompt}:{max_tokens}:{temperature}".encode("utf-8")
    ).hexdigest()
    cached = _cache_get(digest)
    if cached is not None:
        logger.debug("LLM cache hit: scope=%s key=%s", cache_scope, digest[:8])
        return cached

    providers = []
    if _GEMINI_KEY:
        providers.append(("Gemini", _invoke_gemini))
    if _PURDUE_KEY:
        providers.append(("Purdue", _invoke_purdue))

    if not providers:
        logger.debug("LLM provider unavailable for cache scope %s", cache_scope)
        return None

    for name, provider in providers:
        raw = provider(system_prompt, user_prompt, max_tokens, temperature)
        if raw:
            _cache_set(digest, raw)
            logger.info(
                "LLM invocation succeeded via %s: scope=%s prompt_chars=%d tokens=%d",
                name,
                cache_scope,
                len(user_prompt),
                max_tokens,
            )
            return raw
        logger.debug("LLM provider %s failed for scope %s", name, cache_scope)
    return None


_CODE_FENCE_START = re.compile(r"^```(?:json)?", re.IGNORECASE)
_CODE_FENCE_END = re.compile(r"```$")


def extract_json_from_llm(raw_text: str) -> Optional[str]:
    """
    Extract the first JSON object from an LLM response.
    Handles common cases where the model wraps JSON in code fences or adds commentary.
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    if _CODE_FENCE_START.match(text):
        text = _CODE_FENCE_START.sub("", text, count=1).strip()
        text = _CODE_FENCE_END.sub("", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    return candidate.strip() if candidate else None


def _invoke_gemini(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if not _GEMINI_KEY:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=_GEMINI_KEY)
        full_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=full_prompt,
        )
        return getattr(response, "text", None)
    except Exception as exc:
        logger.warning("Gemini invocation failed: %s", exc)
        return None


def _invoke_purdue(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if not _PURDUE_KEY:
        return None
    try:
        url = "https://genai.rcac.purdue.edu/api/chat/completions"
        headers = {
            "Authorization": f"Bearer {_PURDUE_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "llama4:latest",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # Some APIs return list of dicts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content) if content else None
    except Exception as exc:
        logger.warning("Purdue GenAI invocation failed: %s", exc)
        return None
