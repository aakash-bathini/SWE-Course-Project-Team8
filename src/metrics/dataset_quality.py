import logging
import math
from typing import List, Any
from src.models.model_types import EvalContext


async def metric(ctx: EvalContext) -> float:
    """
    Dataset Quality Metric
    - returns a score in [0.0, 1.0] based on dataset quality indicators
    - consumes dataset metadata from EvalContext (ctx.dataset)
    """
    hf_list: List[Any] = ctx.hf_data or []  # list of huggingface profiles
    if not hf_list:
        logging.debug("dataset_quality: no huggingface data available")
        return 0.0  # no huggingface data to check
    hf_profile_raw = hf_list[0]  # first profile
    if not isinstance(hf_profile_raw, dict):
        logging.debug("dataset_quality: invalid huggingface data type %s", type(hf_profile_raw).__name__)
        return 0.0
    hf_profile = hf_profile_raw
    repo_type = hf_profile.get("repo_type")

    # compute community score from likes and downloads (used for both datasets and models)
    likes = hf_profile.get("likes") or 0
    downloads = hf_profile.get("downloads") or 0
    likes_norm = math.log1p(likes) / math.log1p(1e5)  # normalize likes, cap at 100k
    downloads_norm = math.log1p(downloads) / math.log1p(1e8)  # normalize downloads, cap at 100M
    community_score = 0.5 * likes_norm + 0.5 * downloads_norm
    community_score = min(community_score, 1.0)

    # If the HF item is a dataset, evaluate its card contents.
    if repo_type == "dataset":
        doc = hf_profile.get("card_yaml") or ""
        checks = {
            "description": ["dataset", "summary", "description"],
            "intended_use": ["intended use", "task", "purpose"],
            "schema": ["schema", "features", "columns", "field"],
            "splits": ["train", "test", "split", "dev", "validation"],
            "processing": ["preprocess", "cleaning", "filtering"],
            "usage": ["example", "usage", "load", "how to"],
            "evaluation": ["evaluation", "benchmark", "metric", "accuracy"],
            "ethics": ["ethical", "bias", "limitation", "fairness", "privacy"],
            "citation": ["citation", "doi", "bibtex"],
        }
        doc_score_int = sum(1 for terms in checks.values() if any(term in doc for term in terms))
        doc_score = min(doc_score_int / len(checks), 1.0)
        dataset_score = 0.3 * community_score + 0.7 * doc_score
        return round(max(0.0, min(1.0, dataset_score)), 2)

    # For models, approximate dataset quality by how clearly datasets are referenced.
    # Many models list training/eval datasets in tags and README.

    # Generic heuristic for all models
    datasets = hf_profile.get("datasets") or []
    ds_presence = min(1.0, len(datasets) / 5.0)
    readme = (hf_profile.get("readme_text") or "").lower()
    mention_terms = ["dataset", "trained on", "evaluated on", "benchmark", "corpus", "data set"]
    mentions = sum(1 for t in mention_terms if t in readme)
    mention_score = min(1.0, mentions / 6.0)
    # Heavier weight on documentation/mentions than community for this metric.
    model_ds_score = 0.6 * max(ds_presence, mention_score) + 0.4 * community_score

    # Check if this is a well-known model with high HF engagement
    downloads = hf_profile.get("downloads", 0)
    likes = hf_profile.get("likes", 0)

    # Well-known models typically have excellent dataset quality
    if downloads > 1000000 or likes > 1000:  # Very popular models
        logging.info(
            f"High-engagement model detected (downloads: {downloads}, likes: {likes}), boosting dataset quality score"
        )
        model_ds_score = min(1.0, model_ds_score + 0.4)  # Add substantial boost for high-signal metadata.
    # Do not penalize moderate engagement by forcing to 0.0; keep computed score
    elif downloads < 10000 and likes < 10:  # Very low engagement models
        # Use a higher minimum floor for stability under sparse metadata.
        model_ds_score = max(0.35, model_ds_score * 0.85)
        logging.info("Very low engagement model detected, applying minimum floor to dataset quality score")

    # Conservative floor for model dataset_quality to reduce under-scoring when metadata is sparse.
    model_ds_score = max(0.35, model_ds_score)

    return round(max(0.0, min(1.0, model_ds_score)), 2)
