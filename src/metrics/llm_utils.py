import hashlib
import logging
import re
import threading
from collections import OrderedDict
from typing import Optional

from src.aws.sagemaker_llm import SageMakerLLMService, get_sagemaker_service

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


def reduce_readme_for_llm(readme_text: str, max_chars: int = 4000) -> str:
    """
    Trim a README down to the most signal-rich sections before sending to SageMaker.
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


def cached_sagemaker_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    cache_scope: str,
    max_tokens: int = 384,
    service: Optional[SageMakerLLMService] = None,
) -> Optional[str]:
    """
    Invoke SageMaker chat endpoint with a tiny in-memory cache so repeated prompts
    (especially from concurrent /rate calls) do not hammer the endpoint.
    """
    sm_service = service or get_sagemaker_service()
    if not sm_service:
        logger.debug("SageMaker service unavailable for cache scope %s", cache_scope)
        return None

    cache_allowed = bool(getattr(sm_service, "enable_llm_cache", False))
    digest = hashlib.sha256(f"{cache_scope}:{system_prompt}:{user_prompt}".encode("utf-8")).hexdigest()
    if cache_allowed:
        cached = _cache_get(digest)
        if cached is not None:
            logger.debug("LLM cache hit: scope=%s key=%s", cache_scope, digest[:8])
            return cached

    logger.info(
        "CW_SAGEMAKER_CACHE_MISS: scope=%s prompt_chars=%d tokens=%d",
        cache_scope,
        len(user_prompt),
        max_tokens,
    )
    response = sm_service.invoke_chat_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=0.15,
    )
    if response:
        if cache_allowed:
            _cache_set(digest, response)
    return response


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
