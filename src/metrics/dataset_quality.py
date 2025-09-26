import asyncio
import logging
from typing import Optional, Dict
from src.models.types import EvalContext
import math

async def metric(ctx: EvalContext) -> float:
    """
    Dataset Quality Metric
    - returns a score in [0.0, 1.0] based on dataset quality indicators
    - consumes dataset metadata from EvalContext (ctx.dataset)
    """
    hf_list = ctx.hf_data or []  # list of huggingface profiles
    if not hf_list:
        logging.warning("dataset_quality: no huggingface data available")
        return 0.0  # no huggingface data to check
    hf_profile = hf_list[0]  # just check the first dataset for now
    if hf_profile.get("repo_type") != "dataset":
        logging.info("dataset_quality: not a dataset")
        return 0.0  # not a dataset profile
    
    # compute community score from likes and downloads
    likes = hf_profile.get("likes") or 0
    downloads = hf_profile.get("downloads") or 0
    likes_norm = math.log1p(likes) / math.log1p(1e5)  # normalize likes, cap at 100k
    downloads_norm = math.log1p(downloads) / math.log1p(1e8)  # normalize downloads, cap at 100M
    community_score = 0.5 * likes_norm + 0.5 * downloads_norm
    community_score = min(community_score, 1.0)  # cap at 1.0
    logging.info(f"Dataset community score: likes={likes} ({likes_norm:.3f}), downloads={downloads} ({downloads_norm:.3f}), combined={community_score:.3f}")

    # compute documentation score from presence of readme and examples
    doc = (hf_profile.get("card_yaml") or "").lower()
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
    doc_score = sum(1 for terms in checks.values() if any(term in doc for term in terms))
    doc_score = min(doc_score / len(checks), 1.0)  # normalize to [0,1]
    logging.info(f"Dataset doc score components: {doc_score*len(checks)}/{len(checks)}")
    logging.info(f"Dataset documentation score: {doc_score:.3f}")

    # combine scores with weights
    dataset_score = 0.3 * community_score + 0.7 * doc_score
    dataset_score = max(0.0, min(1.0, dataset_score))
    return round(dataset_score, 2)

    