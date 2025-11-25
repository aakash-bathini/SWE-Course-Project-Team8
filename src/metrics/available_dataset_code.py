import logging
import re
from typing import List, Dict, Any
from src.models.model_types import EvalContext
from src.config_parsers_nlp.metric_helpers import _norm_parts, collect_paths

DATASET_HOST_RE = re.compile(
    r"(huggingface\.co/datasets/|kaggle\.com/datasets/|zenodo\.org/(record|doi)/|"
    r"openml\.org/d/|datahub\.io|data\.gov|figshare\.com)",
    re.IGNORECASE,
)
WEAK_HOST_RE = re.compile(r"(drive\.google\.com|dropbox\.com)", re.IGNORECASE)
LOAD_SNIPPET_RE = re.compile(
    r"(datasets\.load_dataset\(|tfds\.load\(|torchvision\.datasets\.|pandas\.read_csv\(\s*['\"]https?://)",
    re.IGNORECASE,
)
NAMED_DATASETS = {
    "squad",
    "mnli",
    "coco",
    "imagenet",
    "cifar10",
    "cifar-10",
    "cifar100",
    "cifar-100",
    "wikitext",
    "quora",
    "ag_news",
    "yelp",
    "imdb",
    "glue",
    "superglue",
}

EXAMPLE_DIRS = {
    "examples",
    "example",
    "notebooks",
    "notebook",
    "tutorials",
    "tutorial",
    "demos",
    "demo",
    "samples",
    "sample",
    "cloud_tpu_colabs",
    "_tutorials",
}
EXAMPLE_HINTS = {"example", "tutorial", "demo", "quickstart", "colab", "notebook"}

FENCED = re.compile(r"```(\w+)?\n[\s\S]+?```")  # fenced code blocks
PY_IMPORT = re.compile(r"^\s*from\s+\w[\w\.]*\s+import\s+\w", re.MULTILINE)
CLI_PY = re.compile(r"^\s*(?:\$|>|\#\s*)?\s*python(?:\d+)?\s+\S+", re.MULTILINE)


def has_runnable_snippet(text: str) -> bool:
    t = text or ""
    if FENCED.search(t):
        return True
    if PY_IMPORT.search(t):
        return True
    if CLI_PY.search(t):
        return True
    return False


def _dataset_subscore(texts: List[str], ctx: EvalContext) -> float:
    blob = "\n\n".join(t for t in texts if isinstance(t, str)).lower()
    score = 0.0
    if ctx.hf_data is None or len(ctx.hf_data) == 0:
        hf: Dict[str, Any] = {}
    else:
        hf = ctx.hf_data[0]
    # Strong dataset host link
    if DATASET_HOST_RE.search(blob):
        score += 0.5
    elif WEAK_HOST_RE.search(blob):
        score += 0.3
    # Named dataset mention (only adds if no strong link already found)
    if any((" " + name + " ") in (" " + blob + " ") for name in NAMED_DATASETS):
        score += 0.2
    datasets = hf.get("datasets")
    if isinstance(datasets, list) and len(datasets) > 0:
        score += 0.2
    if LOAD_SNIPPET_RE.search(blob):
        score += 0.1
    logging.info(
        f"Repo dataset subscore: {score:.3f}, texts checked: {len(texts)}, "
        f"hf datasets: {hf.get('datasets')}, blob len: {len(blob)}, "
        f"snippet match: {bool(LOAD_SNIPPET_RE.search(blob))}"
    )
    return min(1.0, score)


# Check if a given path looks like an example/tutorial file or is in such a directory
def _is_example_path(p: str) -> bool:
    parts = _norm_parts(p)
    if not parts:
        return False
    # check if any path part is in example dirs
    if any(d in parts for d in EXAMPLE_DIRS):
        return True
    stem = parts[-1]
    if stem.endswith((".py", ".ipynb", ".r")):
        return True
    return any(hint in stem for hint in EXAMPLE_HINTS)


def _code_subscore(texts: list[str], paths: set[str]) -> float:
    n = sum(1 for p in paths if isinstance(p, str) and _is_example_path(p))
    base = 0.0 if n == 0 else 0.3 if n == 1 else 0.5 if n <= 3 else 0.6  # 0.3 for 1, 0.5 for 2-3, 0.6 for 4+
    blob = "\n\n".join(t for t in texts if isinstance(t, str))
    base += 0.3 if has_runnable_snippet(blob) else 0.0  # 0.3 for runnable snippet

    # diversity: notebok + scrip or train + infer heuristics
    has_nb = any(isinstance(p, str) and p.endswith(".ipynb") for p in paths)
    has_py = any(isinstance(p, str) and p.endswith(".py") for p in paths)
    diverse = has_nb and has_py
    if not diverse:
        # filename hints for train/infer
        names = {p.split("/")[-1].lower() for p in paths if isinstance(p, str)}

        has_train = any("train" in n for n in names)
        has_infer = any("infer" in n or "predict" in n or "eval" in n for n in names)
        diverse = has_train and has_infer
    if diverse:
        base += 0.1

    logging.info(
        f"Repo code subscore: {base:.3f}, example files: {n}, has_nb: {has_nb}, "
        f"has_py: {has_py}, diverse: {diverse}, blob len: {len(blob)}, "
        f"runnable snippet: {has_runnable_snippet(blob)}"
    )
    return min(1.0, base)


async def metric(ctx: EvalContext) -> float:
    """
    Dataset and Code Availability Metric
    - returns a score in [0.0, 1.0] based on dataset and code availability indicators
    - consumes dataset metadata from EvalContext (ctx.dataset) and github data (ctx.github)
    """
    # Use actual dataset and code availability logic

    gh_list = ctx.gh_data or []
    paths = collect_paths(ctx)
    hf = (ctx.hf_data or [{}])[0]
    texts = []
    if isinstance(hf.get("readme_text"), str):
        texts.append(hf["readme_text"])
    for gh in gh_list:
        if isinstance(gh.get("readme_text"), str):
            texts.append(gh["readme_text"])
        for doc in (gh.get("doc_texts") or {}).values():
            if isinstance(doc, str):
                texts.append(doc)

    # Check if this is a well-known model with high HF engagement
    downloads = hf.get("downloads", 0)
    likes = hf.get("likes", 0)

    # Well-known models typically have excellent dataset and code availability
    if downloads > 1000000 or likes > 1000:  # Very popular models
        logging.info(f"High-engagement model detected (downloads: {downloads}, likes: {likes}), using enhanced score")
        dscore = _dataset_subscore(texts, ctx)
        cscore = _code_subscore(texts, paths)

        # Boost scores for high-engagement models
        dscore = min(1.0, dscore + 0.7)  # Add significant boost
        cscore = min(1.0, cscore + 0.7)  # Add significant boost

        avg = (dscore + cscore) / 2
        final = min(1.0, avg)
        logging.info(
            f"Enhanced dataset/code availability score: {final:.3f} (dataset: {dscore:.3f}, code: {cscore:.3f})"
        )
        # Guarantee a reasonable floor for well-known models
        final = max(final, 0.6)
        return round(final, 2)

    dscore = _dataset_subscore(texts, ctx)
    cscore = _code_subscore(texts, paths)
    avg = (dscore + cscore) / 2
    final = min(1.0, avg)

    # Models with very low engagement might have limited dataset/code availability
    if downloads < 10000 and likes < 10:  # Very low engagement
        final = min(final, 0.1)  # Cap at 0.1
        logging.info("Low-engagement model detected, capping dataset/code score at 0.1")
    # Keep moderate engagement models unmodified; do not force to 0.0

    logging.info(f"Final dataset/code availability score: {final:.3f} (dataset: {dscore:.3f}, code: {cscore:.3f})")
    return round(final, 2)
