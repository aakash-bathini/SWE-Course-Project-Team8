# src/api/huggingface.py

from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from huggingface_hub import HfApi, ModelCard
from datetime import datetime, date

# hf_cache metadata
_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
_CACHE_PATH = os.path.join(_PROJECT_ROOT, ".cache", "hf_meta.json")
_CACHE_TTL_S = int(os.environ.get("HF_META_CACHE_TTL_S", "3600"))  # default 1 hour refresh

# hf_cache utils
"""
<project_root>/.cache/hf_meta.json:
key = "repo_type:repo_id"
cache[key] = {"payload": data, "fetched_at": time}
"""


# helper for consistent timestamps
def _now() -> float:
    return time.time()


def _project_root() -> str:
    return os.environ.get("PROJECT_ROOT", os.getcwd())


def _cache_path() -> str:
    return os.path.join(_project_root(), ".cache", "hf_meta.json")


# load cache, create if nonexistent
def _load_cache() -> Dict[str, Any]:
    path = _cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _json_default(o: Any) -> str:
    # catches datetime/date or any other odd type and renders as string
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return str(o)


# save updated cache data
def _save_cache(cache: Dict[str, Any]) -> None:
    path = _cache_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, default=_json_default)


# within TTL range?
def _is_fresh(entry: Dict[str, Any]) -> bool:
    return (_now() - entry.get("fetched_at", 0)) <= _CACHE_TTL_S


# url parsing
def parse_hf_url(url: str) -> Tuple[str, str]:
    """
    returns (repo_type, repo_id).
    accepts:
      - https://huggingface.co/<owner>/<name>
      - https://huggingface.co/models/<owner>/<name>
      - https://huggingface.co/datasets/<owner>/<name>
    defaults to 'model' type when ambiguous
    """
    p = urlparse(url)
    if "huggingface.co" not in p.netloc:
        raise ValueError(f"api/huggingface error: Not a Hugging Face URL: {url}")
    parts = [x for x in p.path.split("/") if x]
    if not parts:
        raise ValueError(f"api/huggingface error: Malformed HF URL: {url}")
    if parts[0] in ("models", "model"):
        return "model", "/".join(parts[1:])
    if parts[0] in ("datasets", "dataset"):
        return "dataset", "/".join(parts[1:])
    # if just owner/name, assume is model
    return "model", "/".join(parts)


# parse for github links
_GH_RE = re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^\s)]+)?")


def _extract_github_links(readme_text: str | None, card_yaml: Dict[str, Any]) -> List[str]:
    links = set()

    # common card.yaml fields for github
    for key in ("repository", "code", "github", "source", "repo"):
        val = card_yaml.get(key)
        if isinstance(val, str) and "github.com" in val:
            links.add(val.strip())

    # search through nested structs for github
    for key in ("papers", "links", "resources"):
        val = card_yaml.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and "github.com" in item:
                    links.add(item.strip())
        elif isinstance(val, dict):
            for item in val.values():
                if isinstance(item, str) and "github.com" in item:
                    links.add(item.strip())

    # search readme for github
    if readme_text:
        for m in _GH_RE.finditer(readme_text):
            links.add(m.group(0))

    # clean trailing punctuation in links
    cleaned = [u.rstrip(").,]}>") for u in links]
    return sorted(set(cleaned))


# main function
def scrape_hf_url(url: str) -> Tuple[Dict[str, Any], str]:
    """
    Fetch HF repo metadata + other values, plus relevant
    GitHub links for metrics
    Caches to <project>/.cache/hf_meta.json with TTL
    Returns a flat dict ready for metrics
    """
    repo_type, repo_id = parse_hf_url(url)
    cache = _load_cache()
    key = f"{repo_type}:{repo_id}"
    stored_hf_data = cache.get(key)
    if stored_hf_data and _is_fresh(stored_hf_data):
        return stored_hf_data["payload"], repo_type

    api = HfApi()
    info: Any = (
        api.model_info(repo_id, files_metadata=True)
        if repo_type == "model"
        else api.dataset_info(repo_id, files_metadata=True)
    )

    # license: prefer cardData.license, fallback to repo license
    card_yaml = getattr(info, "cardData", {}) or {}
    lic = card_yaml.get("license") or getattr(info, "license", None)

    siblings = getattr(info, "siblings", None) or []
    files = [{"path": s.rfilename, "size": getattr(s, "size", None)} for s in siblings]
    size = sum(int(f["size"]) for f in files if f["size"] is not None)

    tags = getattr(info, "tags", []) or []
    datasets = [
        t.split("dataset:", 1)[-1] for t in tags if isinstance(t, str) and t.startswith("dataset:")
    ]

    # README
    readme_text: str | None = None
    try:
        readme_text = ModelCard.load(repo_id).text
    except Exception:
        readme_text = None

    gh_links = _extract_github_links(readme_text, card_yaml)

    # normalize lastModified to ISO string for JSON safety
    lm = getattr(info, "lastModified", None)
    if isinstance(lm, (datetime, date)):
        lm = lm.isoformat()

    data = {
        "url": f"https://huggingface.co/{repo_id}",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "license": lic,
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "last_modified": lm,
        "tags": tags,
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "card_yaml": card_yaml,
        "readme_text": readme_text,
        "files": files,
        "size": size,
        "datasets": datasets,
        "github_links": gh_links,
        "_source": {
            "fetched_at": _now(),
            "last_modified": lm,
        },
    }

    cache[key] = {"payload": data, "fetched_at": data["_source"]["fetched_at"]}
    _save_cache(cache)
    return data, repo_type
