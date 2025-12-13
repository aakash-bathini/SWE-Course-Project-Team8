"""
LLM-based Relationship Analysis for Artifacts
Uses LLM to analyze READMEs and extract relationships between artifacts (datasets, code repos)
Per rubric: "LLMs are used to analyze the README or to analyze the relationship between artifacts"
"""

import re
import json
import logging
from typing import Dict, Any, Optional

from src.metrics.llm_utils import (
    reduce_readme_for_llm,
    cached_llm_chat,
    extract_json_from_llm,
)

MAX_INPUT_CHARS = 7000

logger = logging.getLogger(__name__)


async def analyze_artifact_relationships(
    readme_text: Optional[str], hf_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Use LLM to analyze README and extract relationships to datasets and code repositories.

    Returns a dict with:
    - linked_datasets: List of dataset names/URLs mentioned in README
    - linked_code_repos: List of code repository URLs mentioned in README
    - relationship_confidence: Confidence score [0.0, 1.0] for the relationships found

    Heuristically infer links for auto-linking.
    """
    if not readme_text or not readme_text.strip():
        logger.info("No README text provided for relationship analysis")
        return {
            "linked_datasets": [],
            "linked_code_repos": [],
            "relationship_confidence": 0.0,
        }

    # Truncate README if too long
    truncated_readme = readme_text[:MAX_INPUT_CHARS] if len(readme_text) > MAX_INPUT_CHARS else readme_text
    readme_content = reduce_readme_for_llm(truncated_readme)

    prompt = f"""
    You are analyzing the README of an ML model to identify relationships with datasets and code repositories.
    Your job is to extract explicit mentions of datasets and code repositories that are related to this model.

    Step 1: Identify dataset relationships. Look for:
    - Dataset names mentioned in training/evaluation context (e.g., "trained on BookCorpus", "evaluated on SQuAD")
    - Dataset URLs (huggingface.co/datasets/, kaggle.com/datasets/, etc.)
    - Dataset references in model cards or evaluation sections

    Step 2: Identify code repository relationships. Look for:
    - GitHub repository URLs
    - Code repository links in model cards
    - Training/inference code references

    Step 3: For each relationship found, provide:
    - name_or_url: The exact name or URL mentioned
    - relationship_type: "dataset" or "code_repo"
    - confidence: Score [0.0, 1.0] indicating how confident you are this is a real relationship
    - context: Brief excerpt from README showing where it was mentioned

    Return ONLY valid JSON (no extra commentary, no markdown).
    Use this schema:

    {{
    "relationships": [
        {{
        "name_or_url": "bookcorpus",
        "relationship_type": "dataset",
        "confidence": 0.9,
        "context": "trained on BookCorpus dataset"
        }},
        {{
        "name_or_url": "https://github.com/huggingface/transformers",
        "relationship_type": "code_repo",
        "confidence": 0.8,
        "context": "code available at https://github.com/huggingface/transformers"
        }}
    ],
    "summary": {{
        "total_relationships": 0,
        "datasets_found": 0,
        "code_repos_found": 0,
        "overall_confidence": 0.0
    }}
    }}

    README Content:
    ---
    {readme_content}
    ---
    """

    analysis_json = None
    cache_scope = f"relationships:{(hf_data or {}).get('repo_id') or (hf_data or {}).get('model_id') or 'unknown'}"
    for attempt in range(1, 3):
        try:
            system_prompt = (
                "You are an engineer analyzing README files to find relationships "
                "between ML artifacts (models, datasets, code repositories)."
            )
            raw = cached_llm_chat(
                system_prompt=system_prompt,
                user_prompt=prompt,
                cache_scope=cache_scope,
                max_tokens=384,
                temperature=0.15,
                top_p=0.9,
            )
            if not raw:
                continue
            cleaned = extract_json_from_llm(raw)
            if not cleaned:
                logger.debug("Relationship analysis: LLM response lacked JSON payload (attempt %d)", attempt)
                continue
            analysis_json = json.loads(cleaned)
            logger.info("Relationship analysis JSON parse succeeded on attempt %d", attempt)
            break
        except Exception as e:
            logger.debug(f"Relationship analysis attempt {attempt} failed: {e}")
            continue  # Try again

    # Fallback: Use heuristic extraction if LLM fails
    if not analysis_json:
        logger.warning("LLM relationship analysis failed, using heuristic fallback")
        return _heuristic_relationship_extraction(readme_text, hf_data)

    # Extract relationships from LLM response
    relationships = analysis_json.get("relationships", [])
    linked_datasets = []
    linked_code_repos = []
    confidence_scores = []

    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        rel_type = rel.get("relationship_type", "")
        name_or_url = rel.get("name_or_url", "")
        confidence = rel.get("confidence", 0.5)

        if rel_type == "dataset" and name_or_url:
            linked_datasets.append(name_or_url)
            confidence_scores.append(confidence)
        elif rel_type == "code_repo" and name_or_url:
            linked_code_repos.append(name_or_url)
            confidence_scores.append(confidence)

    # Calculate overall confidence
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    if not linked_datasets and not linked_code_repos:
        overall_confidence = 0.0

    logger.info(
        f"LLM relationship analysis found {len(linked_datasets)} datasets, "
        f"{len(linked_code_repos)} code repos, confidence={overall_confidence:.2f}"
    )

    return {
        "linked_datasets": linked_datasets,
        "linked_code_repos": linked_code_repos,
        "relationship_confidence": round(overall_confidence, 2),
    }


def _heuristic_relationship_extraction(readme_text: str, hf_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Heuristic fallback for relationship extraction when LLM is unavailable.
    Uses regex patterns to find dataset and code repository mentions.
    """
    linked_datasets = []
    linked_code_repos = []

    # Extract GitHub URLs
    github_pattern = re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^\s)]+)?")
    github_matches = github_pattern.findall(readme_text)
    linked_code_repos.extend(github_matches)

    # Extract dataset URLs
    dataset_url_pattern = re.compile(
        r"https?://(?:www\.)?(?:huggingface\.co/datasets/|kaggle\.com/datasets/|zenodo\.org/(?:record|doi)/)[^\s)]+",
        re.IGNORECASE,
    )
    dataset_url_matches = dataset_url_pattern.findall(readme_text)
    linked_datasets.extend(dataset_url_matches)

    # Extract common dataset names (case-insensitive)
    common_datasets = [
        "bookcorpus",
        "squad",
        "imagenet",
        "cifar",
        "mnist",
        "wikitext",
        "glue",
        "superglue",
    ]
    readme_lower = readme_text.lower()
    for dataset_name in common_datasets:
        if dataset_name in readme_lower:
            linked_datasets.append(dataset_name)

    # Remove duplicates
    linked_datasets = list(set(linked_datasets))
    linked_code_repos = list(set(linked_code_repos))

    logger.info(
        f"Heuristic relationship extraction found {len(linked_datasets)} datasets, "
        f"{len(linked_code_repos)} code repos"
    )

    return {
        "linked_datasets": linked_datasets,
        "linked_code_repos": linked_code_repos,
        "relationship_confidence": 0.5 if (linked_datasets or linked_code_repos) else 0.0,
    }
