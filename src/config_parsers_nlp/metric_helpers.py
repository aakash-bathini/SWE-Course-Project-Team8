from src.models.model_types import EvalContext


# helpers for README checks
def _has_any(text: str, keywords: list[str]) -> float:
    t = (text or "").lower()
    return 1.0 if any(k in t for k in keywords) else 0.0


# normalize path parts for matching
def _norm_parts(p: str) -> list[str]:
    return p.replace("\\", "/").lstrip("./").lower().split("/")


def collect_paths(ctx: EvalContext) -> set[str]:
    paths: set[str] = set()
    hf = (ctx.hf_data or [{}])[0]
    gh_list = ctx.gh_data or []

    # HF files: list of dicts like {"path": "...", "size": ...}
    for f in hf.get("files") or []:
        p = f.get("path") if isinstance(f, dict) else f
        if isinstance(p, str) and p:
            paths.add(p)

    # GH tree (your scraper appears to store full tree entries)
    for gh in gh_list:
        for entry in gh.get("files_index") or []:  # <-- adjust key to whatever you use
            p = entry.get("path") if isinstance(entry, dict) else entry
            if isinstance(p, str) and p:
                paths.add(p)

        # GH doc_texts keys are also paths
        for p in (gh.get("doc_texts") or {}).keys():
            if isinstance(p, str) and p:
                paths.add(p)

    return paths
