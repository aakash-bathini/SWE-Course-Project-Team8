import logging
from src.models.model_types import EvalContext
from src.config_parsers_nlp.metric_helpers import _has_any, _norm_parts, collect_paths


async def metric(ctx: EvalContext) -> float:

    hf = (ctx.hf_data or [{}])[0] if ctx.hf_data else {}
    gh_list = ctx.gh_data or []

    # PRIORITY 1: Check HF engagement metrics first (strongest signal)
    downloads = hf.get("downloads", 0)
    likes = hf.get("likes", 0)
    tags = hf.get("tags", []) or []
    pipeline_tag = hf.get("pipeline_tag")
    card_yaml = hf.get("card_yaml", {}) or {}

    # Start with base score from HF metadata (even without README)
    base_score = 0.0

    # High engagement models get strong base score (community support = easy ramp-up)
    if downloads > 1000000 or likes > 1000:  # Very popular models
        base_score = 0.65  # Strong base for very popular models
        logging.info(
            f"High-engagement model detected (downloads: {downloads}, likes: {likes}), "
            f"base ramp-up score: {base_score}"
        )
    elif downloads > 100000 or likes > 100:  # Popular models
        base_score = 0.50  # Good base for popular models
        logging.info(
            f"Popular model detected (downloads: {downloads}, likes: {likes}), " f"base ramp-up score: {base_score}"
        )
    elif downloads > 10000 or likes > 10:  # Moderate engagement
        base_score = 0.35  # Moderate base
        logging.info(
            f"Moderate engagement model (downloads: {downloads}, likes: {likes}), " f"base ramp-up score: {base_score}"
        )
    else:  # Low engagement
        base_score = 0.20  # Low base, will need good documentation
        logging.info(
            f"Low engagement model (downloads: {downloads}, likes: {likes}), " f"base ramp-up score: {base_score}"
        )

    # BONUS 1: Pipeline tag indicates clear use case (+0.1)
    if pipeline_tag and isinstance(pipeline_tag, str):
        base_score = min(1.0, base_score + 0.10)
        logging.info(f"Pipeline tag '{pipeline_tag}' present, added 0.10 to score")

    # BONUS 2: Rich tags indicate good categorization (+0.05 to +0.15)
    tag_bonus = min(0.15, len(tags) * 0.01) if tags else 0.0
    if tag_bonus > 0:
        base_score = min(1.0, base_score + tag_bonus)
        logging.info(f"Tags present ({len(tags)} tags), added {tag_bonus:.2f} to score")

    # BONUS 3: Structured card_yaml indicates good metadata (+0.1)
    if card_yaml and isinstance(card_yaml, dict) and len(card_yaml) > 2:
        base_score = min(1.0, base_score + 0.10)
        logging.info(f"Structured card_yaml with {len(card_yaml)} fields, added 0.10 to score")

    # PRIORITY 2: README analysis (bonus on top of base)
    readme_parts = []
    if hf.get("readme_text"):
        readme_parts.append(hf["readme_text"])
    if gh_list and gh_list[0].get("readme_text"):
        readme_parts.append(gh_list[0]["readme_text"])
    readme_text = "\n\n".join(readme_parts).lower()

    readme_bonus = 0.0
    if readme_text:
        # README subscore (becomes bonus instead of primary score)
        weights = {"install": 0.25, "usage": 0.35, "desc": 0.15, "io": 0.15, "links": 0.10}
        install_s = _has_any(readme_text, ["pip install", "conda install", "requirements.txt", "install"])
        usage_s = (
            1.0
            if ("```" in readme_text) or _has_any(readme_text, ["usage", "example", "from transformers", "pipeline("])
            else 0.0
        )
        desc_s = (
            1.0
            if len(readme_text.split()) >= 80 and _has_any(readme_text, ["overview", "summary", "description"])
            else 0.0
        )
        io_s = _has_any(readme_text, ["inputs", "outputs", "tokeniz", "schema", "feature", "split"])
        links_s = _has_any(readme_text, ["docs", "documentation", "getting started", "read the docs", "wiki"])
        readme_bonus = (
            sum(
                [
                    weights["install"] * install_s,
                    weights["usage"] * usage_s,
                    weights["desc"] * desc_s,
                    weights["io"] * io_s,
                    weights["links"] * links_s,
                ]
            )
            * 0.25
        )  # Scale to 0.25 max bonus

        logging.info(
            f"README bonus: install={install_s}, usage={usage_s}, "
            f"desc={desc_s}, io={io_s}, links={links_s} => {readme_bonus:.3f}"
        )

    # PRIORITY 3: File structure analysis (bonus)
    paths = collect_paths(ctx)
    structure_bonus = 0.0

    if paths:
        logging.debug("TOTAL PATHS: %d", len(paths))
        for p in list(paths)[:25]:
            logging.debug("PATH: %r", p)

        # Examples bonus
        has_nb = any(p.endswith(".ipynb") for p in paths) or any(
            "notebook" in _norm_parts(p) or "tutorial" in _norm_parts(p) for p in paths
        )
        has_ex = any("examples" in _norm_parts(p) for p in paths)
        py_examples = sum(
            p.endswith(".py")
            and (
                "examples" in _norm_parts(p)
                or "demo" in _norm_parts(p)
                or "inference" in _norm_parts(p)
                or "train" in _norm_parts(p)
            )
            for p in paths
        )

        examples_bonus = 0.0
        if has_nb or py_examples > 0:
            examples_bonus = 0.05
        if (py_examples >= 2) or (has_nb and (py_examples >= 1)):
            examples_bonus = 0.08
        if has_ex and (py_examples + (1 if has_nb else 0)) >= 3:
            examples_bonus = 0.12

        # Install manifest bonus
        manifest_paths = [p.lower() for p in paths]
        has_reqs = any(p.endswith("requirements.txt") for p in manifest_paths)
        has_env = any(p.endswith("environment.yml") or p.endswith("environment.yaml") for p in manifest_paths)
        has_setup = any(p.endswith("setup.py") or p.endswith("setup.cfg") for p in manifest_paths)
        has_pyproj = any(p.endswith("pyproject.toml") for p in manifest_paths)
        has_conda = any(p.endswith("conda.yaml") or p.endswith("conda.yml") for p in manifest_paths)
        has_docker = any("dockerfile" in p or p.endswith(".docker") for p in manifest_paths)
        manifest_count = sum([has_reqs, has_env, has_setup, has_pyproj, has_conda, has_docker])

        manifest_bonus = 0.0
        if manifest_count >= 1:
            manifest_bonus = 0.03
        if manifest_count >= 2:
            manifest_bonus = 0.06
        if manifest_count >= 3:
            manifest_bonus = 0.10

        structure_bonus = examples_bonus + manifest_bonus
        logging.info(
            f"Structure bonus: examples={examples_bonus:.3f}, manifest={manifest_bonus:.3f}, "
            f"total={structure_bonus:.3f}"
        )

    # Combine all scores
    total_score = base_score + readme_bonus + structure_bonus
    total_score = max(0.0, min(1.0, total_score))

    logging.info(
        f"Final ramp-up score: base={base_score:.3f}, readme_bonus={readme_bonus:.3f}, "
        f"structure_bonus={structure_bonus:.3f} => {total_score:.3f}"
    )

    return round(total_score, 2)
