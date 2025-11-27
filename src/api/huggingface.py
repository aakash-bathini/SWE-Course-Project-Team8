# src/api/huggingface.py

from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple
import errno
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


def _preferred_cache_dir() -> str:
    """Return a writable cache directory.

    Preference order (if set): HF_HOME, TRANSFORMERS_CACHE, HUGGINGFACE_HUB_CACHE,
    XDG_CACHE_HOME. If none are set, try `<project_root>/.cache`, but if the
    filesystem is read-only (e.g., AWS Lambda `/var/task`), fall back to `/tmp/.cache`.
    """
    for env_var in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        value = os.environ.get(env_var)
        if value:
            return os.path.join(value)

    # Default to project .cache
    default_dir = os.path.join(_project_root(), ".cache")
    try:
        os.makedirs(default_dir, exist_ok=True)
        # Test writability by creating and removing a temp file
        test_path = os.path.join(default_dir, ".writable")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return default_dir
    except OSError as e:
        if e.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            tmp_dir = os.path.join("/tmp", ".cache")
            os.makedirs(tmp_dir, exist_ok=True)
            return tmp_dir
        raise


def _cache_path() -> str:
    return os.path.join(_preferred_cache_dir(), "hf_meta.json")


# load cache, create if nonexistent
def _load_cache() -> Dict[str, Any]:
    path = _cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "r") as f:
            data: Dict[str, Any] = json.load(f)
            return data
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, default=_json_default)
    except OSError as e:
        # If write fails due to read-only FS, retry in /tmp
        if e.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            alt_dir = os.path.join("/tmp", ".cache")
            os.makedirs(alt_dir, exist_ok=True)
            alt_path = os.path.join(alt_dir, "hf_meta.json")
            with open(alt_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, default=_json_default)
        else:
            raise


# within TTL range?
def _is_fresh(entry: Dict[str, Any]) -> bool:
    fetched_at = entry.get("fetched_at", 0)
    if not isinstance(fetched_at, (int, float)):
        return False
    return (_now() - fetched_at) <= _CACHE_TTL_S


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
        if isinstance(val, str):
            try:
                host = urlparse(val.strip()).hostname or ""
            except Exception:
                host = ""
            if host.lower() == "github.com":
                links.add(val.strip())

    # search through nested structs for github
    for key in ("papers", "links", "resources"):
        val = card_yaml.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    try:
                        host = urlparse(item.strip()).hostname or ""
                    except Exception:
                        host = ""
                    if host.lower() == "github.com":
                        links.add(item.strip())
        elif isinstance(val, dict):
            for item in val.values():
                if isinstance(item, str):
                    try:
                        host = urlparse(item.strip()).hostname or ""
                    except Exception:
                        host = ""
                    if host.lower() == "github.com":
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
    datasets = [t.split("dataset:", 1)[-1] for t in tags if isinstance(t, str) and t.startswith("dataset:")]

    # README - Fetch with multiple fallback methods
    readme_text: str = ""  # Default to empty string instead of None

    # Method 1: Try ModelCard.load() (preferred, most reliable)
    try:
        card = ModelCard.load(repo_id)
        if card and hasattr(card, "text") and card.text:
            readme_text = card.text
            print(f"HF_README: ✓ Successfully loaded README for {repo_id} via ModelCard.load()")
    except Exception as e:
        print(f"HF_README: Method 1 (ModelCard.load) failed for {repo_id}: {type(e).__name__}: {e}")

    # Method 2: Try getting README from cardData in info object
    if not readme_text and card_yaml:
        try:
            # Some models have the text embedded in cardData
            if isinstance(card_yaml, dict):
                card_text = card_yaml.get("text") or card_yaml.get("content") or card_yaml.get("readme")
                if card_text and isinstance(card_text, str):
                    readme_text = card_text
                    print(f"HF_README: ✓ Extracted README from cardData for {repo_id}")
        except Exception as e:
            print(f"HF_README: Method 2 (cardData extraction) failed for {repo_id}: {e}")

    # Method 3: Try direct HTTP request to README.md file
    if not readme_text:
        try:
            import requests

            readme_url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
            response = requests.get(readme_url, timeout=10)
            if response.status_code == 200 and response.text:
                readme_text = response.text
                print(f"HF_README: ✓ Fetched README via HTTP from {readme_url}")
            else:
                print(f"HF_README: Method 3 (HTTP) returned status {response.status_code} for {repo_id}")
        except Exception as e:
            print(f"HF_README: Method 3 (HTTP request) failed for {repo_id}: {e}")

    # Method 4: Try HfApi to get README file content directly
    if not readme_text:
        try:
            readme_content = api.hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type=repo_type,
            )
            if readme_content and os.path.exists(readme_content):
                with open(readme_content, "r", encoding="utf-8") as f:
                    readme_text = f.read()
                print(f"HF_README: ✓ Downloaded README file for {repo_id} via hf_hub_download")
        except Exception as e:
            print(f"HF_README: Method 4 (hf_hub_download) failed for {repo_id}: {e}")

    # Final check and logging
    if readme_text:
        print(f"HF_README: SUCCESS - README length: {len(readme_text)} chars for {repo_id}")
    else:
        print(f"HF_README: WARNING - No README text found for {repo_id} after all fallback methods")
        readme_text = ""  # Ensure it's empty string, not None

    gh_links = _extract_github_links(readme_text if readme_text else None, card_yaml)

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

    fetched_time: float = data["_source"]["fetched_at"]  # type: ignore[assignment, index, call-overload]
    cache[key] = {"payload": data, "fetched_at": fetched_time}
    _save_cache(cache)
    return data, repo_type
