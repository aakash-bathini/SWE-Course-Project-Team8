import logging, math
from src.models.types import EvalContext

# helpers for README checks
def _has_any(text: str, keywords: list[str]) -> float:
    t = (text or "").lower()
    return 1.0 if any(k in t for k in keywords) else 0.0
# normalize path parts for matching
def _norm_parts(p: str) -> list[str]:
    return p.replace("\\", "/").lstrip("./").lower().split("/")

def collect_paths(ctx: EvalContext) -> set[str]:
    paths: set[str] = set()
    hf = (ctx.hf_data or [{}])[0] if ctx.hf_data else {}
    gh_list = ctx.gh_data or []

    # HF files: list of dicts like {"path": "...", "size": ...}
    for f in (hf.get("files") or []):
        p = f.get("path") if isinstance(f, dict) else f
        if isinstance(p, str) and p:
            paths.add(p)

    # GH tree (your scraper appears to store full tree entries)
    for gh in gh_list:
        for entry in (gh.get("files_index") or []):   # <-- adjust key to whatever you use
            p = entry.get("path") if isinstance(entry, dict) else entry
            if isinstance(p, str) and p:
                paths.add(p)

        # GH doc_texts keys are also paths
        for p in (gh.get("doc_texts") or {}).keys():
            if isinstance(p, str) and p:
                paths.add(p)

    return paths

async def metric(ctx: EvalContext) -> float:
    hf = (ctx.hf_data or [{}])[0] if ctx.hf_data else {}
    gh_list = ctx.gh_data or []

    readme_parts = []
    if hf.get("readme_text"):
        readme_parts.append(hf["readme_text"])
    if gh_list and gh_list[0].get("readme_text"):
        readme_parts.append(gh_list[0]["readme_text"])
    readme_text = "\n\n".join(readme_parts).lower()

    # README subscore

    weights = {"install":0.25,"usage":0.35,"desc":0.15,"io":0.15,"links":0.10}
    install_s = _has_any(readme_text, ["pip install","conda install","requirements.txt","install"])
    usage_s   = 1.0 if ("```" in (readme_text or "")) or _has_any(readme_text, ["usage","example","from transformers","pipeline("]) else 0.0
    desc_s    = 1.0 if len((readme_text or "").split()) >= 80 and _has_any(readme_text, ["overview","summary","description"]) else 0.0
    io_s      = _has_any(readme_text, ["inputs","outputs","tokeniz","schema","feature","split"])
    links_s   = _has_any(readme_text, ["docs","documentation","getting started","read the docs","wiki"])
    readme_score = sum([
        weights["install"]*install_s,
        weights["usage"]*usage_s,
        weights["desc"]*desc_s,
        weights["io"]*io_s,
        weights["links"]*links_s,
    ])

    readme_score = max(0.0, min(1.0, readme_score))
    logging.info(f"Repo README subscore: install={install_s}, usage={usage_s}, desc={desc_s}, io={io_s}, links={links_s} => {readme_score:.3f}")

    # examples subscore

    
    paths = collect_paths(ctx)
    logging.debug("TOTAL PATHS: %d", len(paths))
    for p in list(paths)[:25]:
        logging.debug("PATH: %r", p)

    has_nb  = any(p.endswith(".ipynb") for p in paths) or any("notebook" in _norm_parts(p) or "tutorial" in _norm_parts(p) for p in paths)
    has_ex  = any("examples" in _norm_parts(p) for p in paths)
    py_examples = sum(p.endswith(".py") and ("examples" in _norm_parts(p) or "demo" in _norm_parts(p) or "inference" in _norm_parts(p) or "train" in _norm_parts(p)) for p in paths)


    # examples scoring logic
    if not has_nb and not has_ex and py_examples == 0: # no examples at all
        examples_score = 0.0
    elif has_nb or py_examples > 0: # some examples
        examples_score = 0.4
    elif (py_examples >= 2) or (has_nb and (py_examples >= 1)): # decent examples
        examples_score = 0.7
    else: # good examples
        examples_score = 0.4

    if has_ex and (py_examples + (1 if has_nb else 0)) >= 3: # very good examples
        examples_score = min(1.0, examples_score + 0.3)

    logging.info(f"Repo examples subscore: has_nb={has_nb}, has_ex={has_ex}, py_examples={py_examples} => {examples_score:.3f}")

    # install subscore

    manifest_paths = [p.lower() for p in paths]
    has_reqs = any(p.endswith("requirements.txt") for p in manifest_paths)
    has_env = any(p.endswith("environment.yml") or p.endswith("environment.yaml") for p in manifest_paths)
    has_setup = any(p.endswith("setup.py") or p.endswith("setup.cfg") for p in manifest_paths)
    has_pyproj = any(p.endswith("pyproject.toml") for p in manifest_paths)
    has_conda = any(p.endswith("conda.yaml") or p.endswith("conda.yml") for p in manifest_paths)
    has_docker = any("dockerfile" in p or p.endswith(".docker") for p in manifest_paths)
    has_make = any(p.endswith("makefile") for p in manifest_paths)
    has_pipfile = any(p.endswith("pipfile") or p.endswith("pipfile.lock") for p in manifest_paths)
    has_manifest = has_reqs or has_env or has_setup or has_pyproj or has_conda or has_docker or has_make or has_pipfile

    manifest_score = 0.0
    if has_manifest:
        manifest_score = 0.5
        if sum([has_reqs, has_env, has_setup, has_pyproj, has_conda, has_docker, has_make, has_pipfile]) >= 2:
            manifest_score = 0.75
        if sum([has_reqs, has_env, has_setup, has_pyproj, has_conda, has_docker, has_make, has_pipfile]) >= 3:
            manifest_score = 1.0
    logging.info(f"Repo manifest subscore: has_reqs={has_reqs}, has_env={has_env}, has_setup={has_setup}, has_pyproj={has_pyproj}, has_conda={has_conda}, has_docker={has_docker}, has_make={has_make}, has_pipfile={has_pipfile} => {manifest_score:.3f}")

    total_score = 0.5 * readme_score + 0.3 * examples_score + 0.2 * manifest_score
    total_score = max(0.0, min(1.0, total_score))
    return round(total_score, 2)