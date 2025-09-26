from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List
import os
import logging
import re
from urllib.parse import urlparse

from src.orchestration.logging_util import setup_logging_util
from src.orchestration.prep_eval_orchestrator import prep_eval_many
from src.orchestration.metric_orchestrator import orchestrate
from src.models.types import EvalContext, OrchestrationReport


def normalize_url(u: str) -> str:
    """
    Normalize Hugging Face and GitHub URLs to their base project URL.
    - Removes `/tree/<branch>` or `/blob/<branch>` parts for Hugging Face/GitHub
    - Leaves dataset and repo URLs intact
    """
    parsed = urlparse(u)
    if "huggingface.co" in parsed.netloc:
        parts = parsed.path.strip("/").split("/")
        # Hugging Face model or dataset structure: /org/name[/tree/branch/...]
        if len(parts) >= 2 and parts[2:] and parts[2] in ("tree", "blob"):
            return f"{parsed.scheme}://{parsed.netloc}/{parts[0]}/{parts[1]}"
    if "github.com" in parsed.netloc:
        parts = parsed.path.strip("/").split("/")
        # GitHub repo base: /org/repo[/tree/branch/...]
        if len(parts) >= 2 and parts[2:] and parts[2] in ("tree", "blob"):
            return f"{parsed.scheme}://{parsed.netloc}/{parts[0]}/{parts[1]}"
    return u


# 1) Parse URLs from an ASCII file → List[str]
def parse_urls_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        #print(f"Error: URL file not found: {path}", file=sys.stderr)
        logging.error(f"URL file not found: {path}")
        sys.exit(1)
    try:
        # normalize each URL after cleaning it
        return [normalize_url(s) for s in _read_lines_ascii(p)]
    except UnicodeError:
        #print("Error: URL file must be ASCII-encoded.", file=sys.stderr)
        logging.error("URL file must be ASCII-encoded.")
        sys.exit(1)


def _read_lines_ascii(p: Path) -> Iterable[str]:
    """
    Reads each line from the file, cleans it up, and ensures it's a valid URL.
    - Splits on commas and spaces if multiple URLs are in one line
    - Removes leading/trailing commas and spaces
    - Skips empty strings
    - Auto-adds 'https://' if missing
    """
    with p.open("r", encoding="ascii", errors="strict") as f:
        for line in f:
            # Split on commas and whitespace
            parts = [part.strip() for part in re.split(r"[,\s]+", line) if part.strip()]
            for s in parts:
                if not s:
                    continue
                if not s.startswith("http://") and not s.startswith("https://"):
                    s = "https://" + s
                yield s


# 2) Prep the contexts (concurrently) → Dict[url, EvalContext]
async def prep_contexts(urls: List[str]) -> Dict[str, EvalContext]:
    return await prep_eval_many(urls, limit=8)


async def setup_logging() -> None:
    setup_logging_util(False)


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
def print_ndjson(
    urls: List[str], 
    ctx_map: Dict[str, EvalContext], 
    reports: Dict[str, OrchestrationReport]
) -> None:
    for u in urls:
        rep = reports.get(u)
        if not rep:
            continue

        ctx = ctx_map.get(u)  # has .category set by prepare_eval_context
        category = ctx.category if ctx else None

        # derive a short name from the URL (last path component)
        path = urlparse(u).path.strip("/")
        name = path.split("/")[-1] if path else u

        out = {"name": name}
        if category is not None:
            out["category"] = category  # "MODEL" | "DATASET" | "CODE"

        for label, r in rep.results.items():
            out[label] = r.value
            out[f"{label}_latency"] = r.latency_ms
            if getattr(r, "error", None):
                out[f"{label}_error"] = r.error

        print(json.dumps(out, separators=(",", ":"), ensure_ascii=True))


# 5) Final entrypoint that calls the 4 above
def run_eval(url_file: str) -> None:
    urls = parse_urls_from_file(url_file)
    asyncio.run(setup_logging())
    ctx_map = asyncio.run(prep_contexts(urls))
    reports = asyncio.run(run_metrics_on_contexts(urls, ctx_map, limit=4))
    print_ndjson(urls, ctx_map, reports)
    sys.exit(0)
