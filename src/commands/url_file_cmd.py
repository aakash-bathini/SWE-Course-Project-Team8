# src/commands/url_file_cmd.py
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List
import os
import logging

from src.orchestration.logging_util import setup_logging_util
from src.orchestration.prep_eval_orchestrator import prep_eval_many
from src.orchestration.metric_orchestrator import orchestrate
from src.models.types import EvalContext, OrchestrationReport


# 1) Parse URLs from an ASCII file → List[str]
def parse_urls_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        print(f"Error: URL file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return [s for s in _read_lines_ascii(p)]
    except UnicodeError:
        print("Error: URL file must be ASCII-encoded.", file=sys.stderr)
        sys.exit(1)


def _read_lines_ascii(p: Path) -> Iterable[str]:
    with p.open("r", encoding="ascii", errors="strict") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


# 2) Prep the contexts (concurrently) → Dict[url, EvalContext]
async def prep_contexts(urls: List[str]) -> Dict[str, EvalContext]:
    return await prep_eval_many(urls, limit=8)


async def setup_logging() -> None:
    log_file = os.path.join(os.path.dirname(__file__), "temp.log")
    setup_logging_util(level=2, log_file=log_file, also_stderr=False)
   
# 3) Run metrics one URL at a time (sequential for simplicity)
#    → Dict[url, OrchestrationReport]
async def run_metrics_on_contexts(
    urls: List[str],
    ctx_map: Dict[str, EvalContext],
    limit: int = 4,
) -> Dict[str, OrchestrationReport]:
    reports: Dict[str, OrchestrationReport] = {}
    for u in urls:
        ctx = ctx_map.get(u)
        if ctx is None:
            continue  # skip missing/failed preps
        report = await orchestrate(ctx, limit=limit)
        reports[u] = report
    return reports


# 4) Print results in NDJSON (one line per URL)
def print_ndjson(urls: List[str], reports: Dict[str, OrchestrationReport]) -> None:
    for u in urls:
        rep = reports.get(u)
        if not rep:
            continue
        payload = {
            "URL": u,
            "total_latency_ms": rep.total_latency_ms,
            "results": {
                name: {
                    "value": r.value,
                    "latency_ms": r.latency_ms,
                    **({"error": r.error} if r.error else {}),
                }
                for name, r in rep.results.items()
            },
        }
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))


# 5) Final entrypoint that calls the 4 above
def run_eval(url_file: str) -> None:
    urls = parse_urls_from_file(url_file)
    asyncio.run(setup_logging())
    ctx_map = asyncio.run(prep_contexts(urls))
    reports = asyncio.run(run_metrics_on_contexts(urls, ctx_map, limit=4))
    print_ndjson(urls, reports)
    sys.exit(0)