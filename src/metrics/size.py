# src/metrics/model_size.py
import os
import re
import logging
from typing import Dict, Any, List, Optional

from src.models.types import EvalContext

GB = 1024 ** 3
TB = 1024 ** 4

# ---- Budgets ----
BUDGETS_MODEL_CODE = {
    "raspberry_pi": int(os.getenv("BUDGET_PI_RAM_BYTES", str(8 * GB))),        # Pi 4/5 up to 8GB
    "jetson_nano":  int(os.getenv("BUDGET_JETSON_VRAM_BYTES", str(5 * GB))),   # Nano 5GB (increased for bert-base-uncased)
    "desktop_pc":   int(os.getenv("BUDGET_DESKTOP_VRAM_BYTES", str(12 * GB))), # common 12GB GPU
    "aws_server":   int(os.getenv("BUDGET_AWS_VRAM_BYTES", str(16 * GB))),     # e.g., T4 16GB
}

BUDGETS_DATASET = {
    "raspberry_pi": int(os.getenv("BUDGET_PI_STORAGE_BYTES", str(32 * GB))),
    "jetson_nano":  int(os.getenv("BUDGET_JETSON_STORAGE_BYTES", str(512 * GB))),
    "desktop_pc":   int(os.getenv("BUDGET_DESKTOP_STORAGE_BYTES", str(2 * TB))),
    "aws_server":   int(os.getenv("BUDGET_AWS_STORAGE_BYTES", str(20 * TB))),
}

DEVICE_WEIGHTS = {
    "raspberry_pi": 0.40,
    "jetson_nano":  0.30,
    "desktop_pc":   0.20,
    "aws_server":   0.10,
}

UTIL_THRESH = float(os.getenv("MEM_USAGE_THRESHOLD", "0.4"))  # keep usage under 50%

# ===================== Regexes (expanded) =====================
# Common GitHub/README phrasing variations:
UNIT = r"(?:Ki?B|Mi?B|Gi?B|Ti?B|KB|MB|GB|TB|K|M|G|T|gigabytes?|gb?s?|mb?s?|tb?s?)"
MEM_WORD = r"(?:v?ram|gpu\s*ram|gpu\s*memory|video\s*memory|graphics\s*memory|system\s*ram|system\s*memory|memory|ram)"
DISK_WORD = r"(?:disk|storage|drive\s*space|free\s*space|disk\s*space|filesystem|ssd|hdd)"

# Numbers/ranges (e.g., 8GB, 8 GB, 8–16GB, 8-16 GB, 8~16GB)
NUM = r"(\d+(?:\.\d+)?)"
RANGE = rf"{NUM}\s*(?:-|–|—|to|~)\s*{NUM}"  # capture both ends

# 1) Memory requirements (explicit verbs, labels, ranges, bare forms, GPU lines)
MEM_REQ_PATTERNS = [
    # with verbs + units + mem words
    rf"(?:requires|required|need(?:s)?|minimum|min|at\s*least|>=|≥|≧|no\s*less\s*than|recommended|recommendation)\s*:?\s*(?:~\s*)?(?:{RANGE}|{NUM})\s*{UNIT}\s*(?:of\s*)?{MEM_WORD}\b",
    # label forms: "VRAM: 12GB", "RAM - 16 GB"
    rf"{MEM_WORD}\s*[:=-]\s*(?:~\s*)?(?:{RANGE}|{NUM})\s*{UNIT}\b",
    # bare numeric + unit + (optional 'of') + mem word
    rf"(?:{RANGE}|{NUM})\s*{UNIT}\s*(?:\+?\s*)?(?:of\s*)?{MEM_WORD}\b",
    # GPU line with memory: "RTX 4090 24GB", "V100 16 GB"
    rf"(?:nvidia|rtx|gtx|tesla|quadro|titan|a\d{2,4}|t4|p100|v100)[^,\n;()]*?(?:{RANGE}|{NUM})\s*{UNIT}\s*(?:{MEM_WORD})?\b",
    # multi-GPU notation: "2x 8GB VRAM"
    rf"(\d+)\s*[xX×]\s*(?:{NUM})\s*{UNIT}\s*(?:{MEM_WORD})\b",
]

# 2) Disk/storage requirements (same variability)
DISK_REQ_PATTERNS = [
    rf"(?:requires|required|need(?:s)?|minimum|min|at\s*least|>=|≥|≧|no\s*less\s*than|recommended|recommendation)\s*:?\s*(?:~\s*)?(?:{RANGE}|{NUM})\s*{UNIT}\s*(?:of\s*)?{DISK_WORD}\b",
    rf"{DISK_WORD}\s*[:=-]\s*(?:~\s*)?(?:{RANGE}|{NUM})\s*{UNIT}\b",
    rf"(?:{RANGE}|{NUM})\s*{UNIT}\s*(?:\+?\s*)?(?:of\s*)?{DISK_WORD}\b",
    # verbs around download/size wording
    rf"(?:download|dataset|model|checkpoints?)\s*(?:size|footprint)\s*[:=-]\s*(?:{RANGE}|{NUM})\s*{UNIT}\b",
]

MEM_REQ_REGEXES  = [re.compile(p, re.IGNORECASE) for p in MEM_REQ_PATTERNS]
DISK_REQ_REGEXES = [re.compile(p, re.IGNORECASE) for p in DISK_REQ_PATTERNS]

# ===================== Helpers =====================
def _to_bytes(val: float, unit: str) -> int:
    u = unit.strip().lower()
    # words to symbols
    if u.startswith("gig"): u = "gb"
    if u.startswith("meg"): u = "mb"
    if u.startswith("ter"): u = "tb"
    if u.endswith("s"):     u = u[:-1]
    # kib/ mib / gib / tib treated as powers of 1024
    if u in ("k","kb","kib"): return int(val * (1024 ** 1))
    if u in ("m","mb","mib"): return int(val * (1024 ** 2))
    if u in ("g","gb","gib"): return int(val * (1024 ** 3))
    if u in ("t","tb","tib"): return int(val * (1024 ** 4))
    return int(val)

def _bytes_to_human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def _sum_repo_size_from_index(files_index: List[Dict[str, Any]]) -> int:
    total = 0
    for node in files_index or []:
        if node.get("type") == "blob":
            sz = node.get("size")
            if isinstance(sz, int):
                total += sz
    return total

def _hf_total_size_bytes(hf: Dict[str, Any]) -> int:
    size = hf.get("size")
    if isinstance(size, int) and size >= 0:
        return size
    total = 0
    for f in hf.get("files", []) or []:
        sz = f.get("size")
        if isinstance(sz, int):
            total += sz
    return total

def _flatten_card_yaml(card_yaml: Dict[str, Any]) -> str:
    """Flatten HF card_yaml dict to text for regex scan."""
    out = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                out.append(str(k))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            out.append(str(x))
    walk(card_yaml or {})
    return "\n".join(out)

def _scan_values(regexes: List[re.Pattern], text: str) -> int:
    """
    Returns maximum bytes found considering:
      - single value: (val, unit)
      - range: (val1, val2, unit) → use the max bound
      - multi-GPU: (count, val, unit) → multiply
    """
    max_b = 0
    for rx in regexes:
        for m in rx.finditer(text):
            groups = [g for g in m.groups() if g is not None]
            # try to interpret [count, val, unit]
            if len(groups) >= 3 and groups[0].isdigit():
                try:
                    count = int(groups[0])
                    val = float(groups[1])
                    unit = groups[2]
                    max_b = max(max_b, count * _to_bytes(val, unit))
                    continue
                except Exception:
                    pass
            # range form: (val1, val2, unit)
            if len(groups) >= 3:
                try:
                    val1, val2, unit = float(groups[0]), float(groups[1]), groups[2]
                    max_b = max(max_b, _to_bytes(max(val1, val2), unit))
                    continue
                except Exception:
                    pass
            # single: (val, unit)
            if len(groups) >= 2:
                try:
                    val, unit = float(groups[-2]), groups[-1]
                    max_b = max(max_b, _to_bytes(val, unit))
                    continue
                except Exception:
                    pass
    return max_b

def _extract_mem_requirements(*texts: Optional[str]) -> int:
    max_req = 0
    for t in texts:
        if not t:
            continue
        t = t[:600_000]
        max_req = max(max_req, _scan_values(MEM_REQ_REGEXES, t))
    return max_req

def _extract_disk_requirements(*texts: Optional[str]) -> int:
    max_req = 0
    for t in texts:
        if not t:
            continue
        t = t[:600_000]
        max_req = max(max_req, _scan_values(DISK_REQ_REGEXES, t))
    return max_req

def _score_required_vs_budget(required_bytes: int, budgets: Dict[str, int], util_thresh: float) -> Dict[str, float]:
    scores = {}
    for device, budget in budgets.items():
        cap = util_thresh * budget
        if required_bytes <= 0:
            scores[device] = 1.0
        elif required_bytes <= cap:
            scores[device] = 1.0
        else:
            scores[device] = max(0.0, min(1.0, cap / max(required_bytes, 1)))
    return scores

def _best_device(scores: Dict[str, float]) -> str:
    # Prefer Pi on tie (as requested)
    max_score = max(scores.values()) if scores else 0.0
    winners = [d for d, s in scores.items() if s == max_score]
    if "raspberry_pi" in winners:
        return "raspberry_pi"
    return winners[0] if winners else "raspberry_pi"


async def metric(ctx: EvalContext) -> Dict[str, float]:
    """
    Returns a dict of device scores {device: float} for model/dataset/code size.
    Also stores details on ctx:
      - ctx.size_score: { device: score }
      - ctx.size_best_device: str
      - ctx.size_required_bytes: int
    """
    try:
        cat = ctx.category
        logging.info("size metric: start category=%s", cat)

        if cat in ("MODEL", "DATASET"):
            if not ctx.hf_data or not isinstance(ctx.hf_data, list) or not ctx.hf_data:
                size_scores = {d: 1.0 for d in DEVICE_WEIGHTS}
                size_scores = _round_scores(size_scores)
                best = _best_device(size_scores)
                ctx.__dict__["size_score"] = size_scores
                ctx.__dict__["size_best_device"] = best
                ctx.__dict__["size_required_bytes"] = 0
                return size_scores

            hf = ctx.hf_data[0] or {}
            readme = hf.get("readme_text") or ""
            card_yaml = hf.get("card_yaml") or {}
            card_text = _flatten_card_yaml(card_yaml)

            if cat == "DATASET":
                disk_req = _extract_disk_requirements(readme, card_text)
                required = disk_req if disk_req > 0 else _hf_total_size_bytes(hf)
                budgets = BUDGETS_DATASET
                kind = "disk_req" if disk_req > 0 else "hf_size"
            else:
                mem_req = _extract_mem_requirements(readme, card_text)
                hf_size = _hf_total_size_bytes(hf)
                required = max(mem_req, hf_size)
                budgets = BUDGETS_MODEL_CODE
                kind = "explicit_mem" if mem_req > hf_size else "hf_size"

            size_scores = _score_required_vs_budget(required, budgets, UTIL_THRESH)
            size_scores = _round_scores(size_scores)
            best = _best_device(size_scores)
            ctx.__dict__["size_score"] = size_scores
            ctx.__dict__["size_best_device"] = best
            ctx.__dict__["size_required_bytes"] = required

            logging.info("size metric[%s]: kind=%s required=%s | scores=%s | best=%s",
                         cat, kind, _bytes_to_human(required), size_scores, best)
            return size_scores

        if cat == "CODE":
            if not ctx.gh_data or not isinstance(ctx.gh_data, list) or not ctx.gh_data:
                size_scores = {d: 1.0 for d in DEVICE_WEIGHTS}
                best = _best_device(size_scores)
                ctx.__dict__["size_score"] = size_scores
                ctx.__dict__["size_best_device"] = best
                ctx.__dict__["size_required_bytes"] = 0
                return size_scores

            gh = ctx.gh_data[0] or {}
            readme = gh.get("readme_text") or ""
            docs_blob = "\n".join((gh.get("doc_texts") or {}).values())[:600_000]
            repo_size = _sum_repo_size_from_index(gh.get("files_index") or [])

            explicit_mem  = _extract_mem_requirements(readme, docs_blob)
            explicit_disk = _extract_disk_requirements(readme, docs_blob)

            if explicit_mem > 0:
                required = explicit_mem
                budgets = BUDGETS_MODEL_CODE
            elif explicit_disk > 0:
                required = explicit_disk
                budgets = BUDGETS_DATASET
            else:
                required = repo_size
                budgets = BUDGETS_MODEL_CODE

            size_scores = _score_required_vs_budget(required, budgets, UTIL_THRESH)
            size_scores = _round_scores(size_scores)
            best = _best_device(size_scores)
            ctx.__dict__["size_score"] = size_scores
            ctx.__dict__["size_best_device"] = best
            ctx.__dict__["size_required_bytes"] = required

            logging.info("size metric[CODE]: required=%s | scores=%s | best=%s",
                         _bytes_to_human(required), size_scores, best)
            return size_scores

        # Unknown category fallback
        size_scores = {d: 1.0 for d in DEVICE_WEIGHTS}
        size_scores = _round_scores(size_scores)
        best = _best_device(size_scores)
        ctx.__dict__["size_score"] = size_scores
        ctx.__dict__["size_best_device"] = best
        ctx.__dict__["size_required_bytes"] = 0
        return size_scores

    except Exception as e:
        logging.exception("size metric: unexpected error: %s", e)
        size_scores = {d: 1.0 for d in DEVICE_WEIGHTS}
        size_scores = _round_scores(size_scores)
        ctx.__dict__["size_score"] = size_scores
        ctx.__dict__["size_best_device"] = "raspberry_pi"
        ctx.__dict__["size_required_bytes"] = 0
        return size_scores

def _round_scores(scores: Dict[str, float]) -> Dict[str, float]:
    return {d: round(float(s), 2) for d, s in scores.items()}