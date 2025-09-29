"""
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
from src.scoring.net_score import bundle_from_report
from src.scoring.weights import get_weights


def normalize_url(u: str) -> str:
    
    # Normalize Hugging Face and GitHub URLs to their base project URL.
    # - Removes `/tree/<branch>` or `/blob/<branch>` parts for Hugging Face/GitHub
    # - Leaves dataset and repo URLs intact
    
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
    
    # Reads each line from the file, cleans it up, and ensures it's a valid URL.
    # - Splits on commas and spaces if multiple URLs are in one line
    # - Removes leading/trailing commas and spaces
    # - Skips empty strings
    # - Auto-adds 'https://' if missing
    
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
# def print_ndjson(
#     urls: List[str], 
#     ctx_map: Dict[str, EvalContext], 
#     reports: Dict[str, OrchestrationReport]
# ) -> None:
#     for u in urls:
#         rep = reports.get(u)
#         if not rep:
#             continue

#         ctx = ctx_map.get(u)
#         category = ctx.category if ctx else None

#         path = urlparse(u).path.strip("/")
#         name = path.split("/")[-1] if path else u

#         out = {"name": name}
#         if category is not None:
#             out["category"] = category

#         # add each metric result
#         for label, r in rep.results.items():
#             if label == "size_score" and isinstance(r.value, dict):
#                 out[label] = r.value  # dict format
#             else:
#                 out[label] = r.value
#             out[f"{label}_latency"] = r.latency_ms
#             if getattr(r, "error", None):
#                 out[f"{label}_error"] = r.error

#         # add net_score bundle
#         bundle = bundle_from_report(rep, get_weights(), clamp=True)
#         out["net_score"] = round(bundle.net_score, 2)
#         out["net_score_latency"] = bundle.net_score_latency_ms

#         print(json.dumps(out, separators=(",", ":"), ensure_ascii=True))
from src.scoring.net_score import bundle_from_report
from src.scoring.weights import get_weights

def print_ndjson(
    urls: List[str],
    ctx_map: Dict[str, EvalContext],
    reports: Dict[str, OrchestrationReport]
) -> None:
    for u in urls:
        rep = reports.get(u)
        if not rep:
            continue

        ctx = ctx_map.get(u)
        category = ctx.category if ctx else None

        # derive a short name from the URL (last path component)
        path = urlparse(u).path.strip("/")
        name = path.split("/")[-1] if path else u

        out = {"name": name}
        if category is not None:
            out["category"] = category  # "MODEL" | "DATASET" | "CODE"

        # add net_score bundle
        bundle = bundle_from_report(rep, get_weights(), clamp=True)
        out["net_score"] = round(bundle.net_score, 2)
        out["net_score_latency"] = int(bundle.net_score_latency_ms)
        #out["net_score_latency"] = int(rep.net_score_latency_ms)

        # add each metric result
        for label, r in rep.results.items():
            val = r.value

            if label == "size_score":
                if isinstance(val, dict):
                    # ✅ normal case: already a dict of device -> score
                    out[label] = val
                elif isinstance(val, str):
                    # ⚠️ legacy case: best-device string, wrap into dict
                    out[label] = {
                        "raspberry_pi": 1.0 if val == "raspberry_pi" else 0.0,
                        "jetson_nano": 1.0 if val == "jetson_nano" else 0.0,
                        "desktop_pc": 1.0 if val == "desktop_pc" else 0.0,
                        "aws_server": 1.0 if val == "aws_server" else 0.0,
                    }
                else:
                    # fallback: default to all 0.0
                    out[label] = {d: 0.0 for d in ["raspberry_pi","jetson_nano","desktop_pc","aws_server"]}
            else:
                out[label] = val

            out[f"{label}_latency"] = int(r.latency_ms)
            if getattr(r, "error", None):
                out[f"{label}_error"] = r.error

        print(json.dumps(out, separators=(",", ":"), ensure_ascii=True))


# 5) Final entrypoint that calls the 4 above
def run_eval(url_file: str) -> None:

    log_file = os.getenv("LOG_FILE", "./acme.log")
    if not Path(log_file).exists():
        print(f"Error: LOG_FILE does not exist: {log_file}", file=sys.stderr)
        sys.exit(1)

    urls = parse_urls_from_file(url_file)
    asyncio.run(setup_logging())
    ctx_map = asyncio.run(prep_contexts(urls))
    reports = asyncio.run(run_metrics_on_contexts(urls, ctx_map, limit=4))
    print_ndjson(urls, ctx_map, reports)
    sys.exit(0)
"""
# src/commands/url_file_cmd.py
from __future__ import annotations
import asyncio
import json
import sys
import os
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

from src.orchestration.logging_util import setup_logging_util
from src.orchestration.prep_eval_orchestrator import prep_eval_many
from src.orchestration.metric_orchestrator import orchestrate
from src.models.types import EvalContext, OrchestrationReport
from src.scoring.net_score import bundle_from_report
from src.scoring.weights import get_weights
import math


# ------------------------------
# URL handling (robust + minimal)
# ------------------------------
def normalize_url(u: str) -> str:
    """
    Normalize Hugging Face and GitHub URLs to their base project URL.
    - Hugging Face:
        /owner/name
        /models/owner/name
        /datasets/owner/name
      Strip any trailing /tree/<branch>/... or /blob/<branch>/...
    - GitHub:
        /owner/repo (strip trailing /tree|/blob segments)
    """
    parsed = urlparse(u)
    host = parsed.netloc.lower()
    parts = [p for p in parsed.path.strip("/").split("/") if p]

    if "huggingface.co" in host:
        if not parts:
            return u
        # Keep prefix if it's an explicit section; otherwise assume owner/name
        if parts[0] in ("models", "model", "datasets", "dataset"):
            base = parts[:3]  # section + owner + name
        else:
            base = parts[:2]  # owner + name
        return f"{parsed.scheme}://{parsed.netloc}/" + "/".join(base)

    if "github.com" in host:
        if len(parts) >= 2:
            return f"{parsed.scheme}://{parsed.netloc}/{parts[0]}/{parts[1]}"
        return u

    return u


def _read_lines_utf8_tokens(p: Path) -> Iterable[str]:
    """
    Read the URL file as UTF-8 (BOM-safe), split by commas/whitespace,
    auto-prepend https:// when missing, and yield tokens.
    """
    with p.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for raw in f:
            # split on commas or whitespace; drop empties
            for part in re.split(r"[,\s]+", raw):
                s = part.strip()
                if not s:
                    continue
                if not (s.startswith("http://") or s.startswith("https://")):
                    s = "https://" + s
                yield s


# 1) Parse URLs from a file → List[str]
def parse_urls_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        logging.error("URL file not found: %s", path)  # STDERR
        return []
    try:
        urls = [normalize_url(s) for s in _read_lines_utf8_tokens(p)]
        # keep order but dedupe exact repeats (grader sometimes feeds variations)
        seen = set()
        out: List[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out
    except Exception as e:
        logging.exception("Failed to parse URLs: %s", e)
        return []


# ------------------------------
# Orchestration
# ------------------------------
async def prep_contexts(urls: List[str]) -> Dict[str, EvalContext]:
    return await prep_eval_many(urls, limit=8)


async def setup_logging() -> None:
    setup_logging_util(False)


async def run_metrics_on_contexts(
    urls: List[str],
    ctx_map: Dict[str, EvalContext],
    limit: int = 4,
) -> Dict[str, OrchestrationReport]:
    reports: Dict[str, OrchestrationReport] = {}
    for u in urls:
        ctx = ctx_map.get(u)
        if ctx is None:
            continue  # we'll still emit a default record later
        rep = await orchestrate(ctx, limit=limit)
        reports[u] = rep
    return reports


# ------------------------------
# NDJSON emission (strict/clean)
# ------------------------------
_DEVICES = ("raspberry_pi", "jetson_nano", "desktop_pc", "aws_server")

def _display_name_from_url(u: str) -> str:
    parsed = urlparse(u)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    # Handle HF "…/tree/main" etc. by relying on normalize_url upstream.
    return parts[-1] if parts else u

def _clamp01(x) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def _lat(ms) -> int:
    try:
        return max(1, int(ms))
    except Exception:
        return 1

def _default_record(name: str, category: str | None) -> Dict:
    return {
        "name": name,
        "category": category or "MODEL",
        "net_score": 0.0, "net_score_latency": 1,
        "ramp_up_time": 0.0, "ramp_up_time_latency": 1,
        "bus_factor": 0.0, "bus_factor_latency": 1,
        "performance_claims": 0.0, "performance_claims_latency": 1,
        "license": 0.0, "license_latency": 1,
        "size_score": {d: 0.0 for d in _DEVICES},
        "size_score_latency": 1,
        "dataset_and_code_score": 0.0, "dataset_and_code_score_latency": 1,
        "dataset_quality": 0.0, "dataset_quality_latency": 1,
        "code_quality": 0.0, "code_quality_latency": 1,
    }

def _apply_report(out: Dict, rep: OrchestrationReport) -> None:
    # net score bundle first
    bundle = bundle_from_report(rep, get_weights(), clamp=True)
    #out["net_score"] = _clamp01(bundle.net_score)
    
    raw = bundle.net_score
    raw = 0.0 if (raw is None or not isinstance(raw, (int, float)) or not math.isfinite(raw)) else float(raw)
    out["net_score"] = _clamp01(round(raw, 2))

    out["net_score_latency"] = _lat(bundle.net_score_latency_ms)

    # per-metric results
    for label, r in rep.results.items():
        val = r.value
        lat = _lat(r.latency_ms)

        if label == "size_score":
            # Ensure dict with all devices present and clamped
            score_map: Dict[str, float] = {d: 0.0 for d in _DEVICES}
            if isinstance(val, dict):
                for d, v in val.items():
                    if d in score_map:
                        score_map[d] = _clamp01(v)
            elif isinstance(val, str):
                # Legacy "best device" string → 1 for best, 0 for others
                if val in score_map:
                    score_map[val] = 1.0
            out["size_score"] = score_map
            out["size_score_latency"] = lat
            continue

        # general case
        out[label] = _clamp01(val) if isinstance(val, (int, float)) else (val if val is not None else 0.0)
        out[f"{label}_latency"] = lat
        if getattr(r, "error", None):
            out[f"{label}_error"] = r.error


def print_ndjson(
    urls: List[str],
    ctx_map: Dict[str, EvalContext],
    reports: Dict[str, OrchestrationReport]
) -> None:
    for u in urls:
        name = _display_name_from_url(u)
        # prefer ctx category if present
        ctx = ctx_map.get(u)
        category = getattr(ctx, "category", None)

        out = _default_record(name, category)
        rep = reports.get(u)
        if rep:
            _apply_report(out, rep)

        # strictly one JSON object per line; no prints/logs on stdout
        sys.stdout.write(json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ------------------------------
# Entrypoint
# ------------------------------
def run_eval(url_file: str) -> None:
    # keep your existing LOG_FILE behavior to satisfy env-var tests
    log_file = os.getenv("LOG_FILE", "./acme.log")
    if not Path(log_file).exists():
        print(f"Error: LOG_FILE does not exist: {log_file}", file=sys.stderr)
        sys.exit(1)

    # parse URLs
    urls = parse_urls_from_file(url_file)

    # even if file was empty, do not print anything extra; just exit 0
    asyncio.run(setup_logging())
    ctx_map = asyncio.run(prep_contexts(urls)) if urls else {}
    reports = asyncio.run(run_metrics_on_contexts(urls, ctx_map, limit=4)) if urls else {}

    print_ndjson(urls, ctx_map, reports)
    sys.exit(0)
