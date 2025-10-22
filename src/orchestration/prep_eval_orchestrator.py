# src/orchestration/prep_eval_orchestrator.py
from __future__ import annotations
import asyncio
import inspect
from typing import Dict, List

from src.api.prep_eval_context import prepare_eval_context
from src.models.model_types import EvalContext


async def _call_prep(url: str) -> EvalContext:
    # Support either async or sync prep functions
    if inspect.iscoroutinefunction(prepare_eval_context):
        return await prepare_eval_context(url)  # type: ignore[misc, no-any-return]
    return await asyncio.to_thread(prepare_eval_context, url)  # type: ignore[misc, no-any-return]


async def prep_eval_many(urls: List[str], limit: int = 4) -> Dict[str, EvalContext]:
    """
    Build EvalContext for many URLs concurrently (bounded by `limit`).
    Returns: { url -> EvalContext }
    """
    sem = asyncio.Semaphore(limit)
    out: Dict[str, EvalContext] = {}

    async def one(u: str) -> None:
        async with sem:
            out[u] = await _call_prep(u)

    await asyncio.gather(*(one(u) for u in urls))
    return out
