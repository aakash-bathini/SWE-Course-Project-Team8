# src/api/github.py
from __future__ import annotations
import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, quote

import requests

# gh_cache metadata
_CACHE_TTL_S = int(os.environ.get("GH_META_CACHE_TTL_S", "3600"))  # 1h default

# gh_cache utils
"""
<project_root>/.cache/gh_meta.json:
key = "owner/repo"
cache[key] = {"payload": data, "fetched_at": time}
"""


# helper for consistent timestamps
def _now() -> float:
    return time.time()


def _project_root() -> str:
    return os.environ.get("PROJECT_ROOT", os.getcwd())


def _cache_path() -> str:
    return os.path.join(_project_root(), ".cache", "gh_meta.json")


# load cache, create if nonexistent
def _load_cache() -> Dict[str, Any]:
    path = _cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# save updated cache data
def _save_cache(cache: Dict[str, Any]) -> None:
    path = _cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


# within TTL range?
def _is_fresh(entry: Dict[str, Any]) -> bool:
    return (_now() - entry.get("fetched_at", 0)) <= _CACHE_TTL_S


# url parsing
def parse_github_url(url: str) -> Tuple[str, str]:
    p = urlparse(url)
    if "github.com" not in p.netloc.lower():
        raise ValueError(f"api/github error: Not a GitHub URL: {url}")
    parts = [x for x in p.path.split("/") if x]
    if len(parts) < 2:
        raise ValueError(f"api/github error: Malformed GitHub URL: {url}")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


# http helpers
def _gh_headers() -> Dict[str, str]:
    # request header
    hdr = {"Accept": "application/vnd.github+json", "User-Agent": "swe-proj-bot"}
    # optionally include token
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    return hdr


def _get_json(url: str, timeout: float = 12.0) -> Dict[str, Any]:
    r = requests.get(url, headers=_gh_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()  # type: ignore[no-any-return]


# selectively harvest content
# file budget for code-quality signals
_MAX_DOC_FILES = int(os.environ.get("GH_MAX_DOC_FILES", "25"))
_MAX_TOTAL_BYTES = int(os.environ.get("GH_MAX_TOTAL_BYTES", str(1_000_000)))  # ~1MB

_DOC_PATTERNS = [
    r"^README(\.[a-zA-Z0-9]+)?$",
    r"^LICENSE(\.[a-zA-Z0-9]+)?$",
    r"^COPYING(\.[a-zA-Z0-9]+)?$",
    r"^CHANGES(\.[a-zA-Z0-9]+)?$",
    r"^CHANGELOG(\.[a-zA-Z0-9]+)?$",
    r"^CONTRIBUTING(\.[a-zA-Z0-9]+)?$",
    r"^SECURITY(\.[a-zA-Z0-9]+)?$",
    r"^CODE_OF_CONDUCT(\.[a-zA-Z0-9]+)?$",
    r"^pyproject\.toml$",
    r"^setup\.py$",
    r"^requirements(\.[a-zA-Z0-9]+)?\.txt$",
    r"^environment\.yml$",
    r"^Makefile$",
    r"^Pipfile$",
    r"^Pipfile\.lock$",
    r"^docs/.*\.md$",
    r"^doc/.*\.md$",
    r"^.*\.md$",  # catch more markdowns, later filtered by budget
    r"^.*\.rst$",
    r"^scripts/.*\.sh$",
]

_DOC_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in _DOC_PATTERNS]


# true if any matching files found
def _should_fetch(path: str) -> bool:
    return any(rx.match(path) for rx in _DOC_REGEXES)


# get reasonable amount of data from file
def _fetch_bitesized_file(owner: str, repo: str, path: str, ref: str, size: int) -> str | None:
    if size is not None and size > _MAX_TOTAL_BYTES:
        return None
    # contents API returns base64 for files
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{quote(path)}?ref={quote(ref)}"
    try:
        j = _get_json(url)
    except Exception:
        return None
    if isinstance(j, dict) and j.get("type") == "file":
        content = j.get("content") or ""
        if j.get("encoding") == "base64":
            try:
                return base64.b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                return None
        # sometimes raw text is sent
        if isinstance(content, str):
            return content
    return None


# main function
def scrape_github_url(url: str) -> Dict[str, Any]:
    """
    Fetch GitHub repo metadata + README + contributors + tree index,
    and selectively download small text/doc files for metrics
    Caches to <project_root>/.cache/gh_meta.json with TTL
    Returns flat dict ready for metrics
    """
    owner, repo = parse_github_url(url)
    cache = _load_cache()
    key = f"{owner}/{repo}"
    stored = cache.get(key)
    if stored and _is_fresh(stored):
        return stored["payload"]

    base = f"https://api.github.com/repos/{owner}/{repo}"

    # repo metadata
    repo_json = _get_json(base)
    default_branch = repo_json.get("default_branch") or "HEAD"

    # README (base64)
    readme_text = None
    try:
        readme_json = _get_json(f"{base}/readme")
        if isinstance(readme_json, dict) and "content" in readme_json:
            enc = readme_json.get("encoding", "base64")
            blob = readme_json.get("content") or ""
            if enc == "base64":
                readme_text = base64.b64decode(blob).decode("utf-8", errors="replace")
                # print("README: " + readme_text + "\n")
            else:
                readme_text = str(blob)
                # print("README: " + readme_text + "\n")
    except Exception:
        readme_text = None

    # contributors (first 100)
    contributors: Dict[str, int] = {}
    try:
        contrib_json = _get_json(f"{base}/contributors?per_page=100&anon=1")
        if isinstance(contrib_json, list):
            for it in contrib_json:
                login = it.get("login") or f"anon:{it.get('name', '?')}"
                contributors[login] = int(it.get("contributions", 0))
    except Exception:
        contributors = {}

    # tree index for small-text file selection
    # use the "git trees" API to fetch a recursive listing without blob contents
    tree: List[Dict[str, Any]] = []
    try:
        tree_json = _get_json(f"{base}/git/trees/{quote(default_branch)}?recursive=1")
        if isinstance(tree_json, dict) and isinstance(tree_json.get("tree"), list):
            # keep only fields we care about (path, type, size)
            for node in tree_json["tree"]:
                path = node.get("path")
                typ = node.get("type")
                size = node.get("size")  # may be absent for 'tree' entries
                if isinstance(path, str) and isinstance(typ, str):
                    tree.append({"path": path, "type": typ, "size": size})
    except Exception:
        tree = []

    # selectively fetch small doc/code-quality files (budgeted)
    doc_texts: Dict[str, str] = {}
    bytes_left = _MAX_TOTAL_BYTES
    fetched = 0
    for node in tree:
        if fetched >= _MAX_DOC_FILES:
            break
        if node["type"] != "blob":
            continue
        path = node["path"]
        size = int(node["size"] or 0)
        if not _should_fetch(path):
            continue
        if size > bytes_left:
            continue
        text = _fetch_bitesized_file(owner, repo, path, default_branch, size)
        if text:
            doc_texts[path] = text
            fetched += 1
            bytes_left -= size

    # build profile
    data = {
        "url": repo_json.get("html_url") or url,
        "repo_id": f"{owner}/{repo}",
        "repo_type": "code",
        # descriptive
        "name": repo_json.get("name"),
        "full_name": repo_json.get("full_name"),
        "description": repo_json.get("description"),
        "homepage": repo_json.get("homepage"),
        "default_branch": default_branch,
        "topics": repo_json.get("topics", []) or [],
        "language": repo_json.get("language"),
        # status
        "archived": bool(repo_json.get("archived", False)),
        "disabled": bool(repo_json.get("disabled", False)),
        "fork": bool(repo_json.get("fork", False)),
        # timestamp strings from API
        "created_at": repo_json.get("created_at"),
        "updated_at": repo_json.get("updated_at"),
        "pushed_at": repo_json.get("pushed_at"),
        # popularity / activity
        "stars": int(repo_json.get("stargazers_count", 0) or 0),
        "forks": int(repo_json.get("forks_count", 0) or 0),
        "open_issues": int(repo_json.get("open_issues_count", 0) or 0),
        "watchers": int(repo_json.get("subscribers_count") or repo_json.get("watchers_count") or 0),
        # licensing
        "license_spdx": (repo_json.get("license") or {}).get("spdx_id"),
        # content
        "readme_text": readme_text,
        "doc_texts": doc_texts,  # path -> text
        "files_index": tree,  # repo-wide listing
        # people
        "contributors": contributors,  # login -> contributions
        "_source": {
            "fetched_at": _now(),
            "api_base": base,
            "limits": {
                "max_files": _MAX_DOC_FILES,
                "max_total_bytes": _MAX_TOTAL_BYTES,
            },
        },
    }

    cache[key] = {"payload": data, "fetched_at": data["_source"]["fetched_at"]}
    _save_cache(cache)
    return data
