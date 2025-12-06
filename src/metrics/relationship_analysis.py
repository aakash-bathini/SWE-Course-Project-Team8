"""
LLM-based Relationship Analysis for Artifacts
Uses LLM to analyze READMEs and extract relationships between artifacts (datasets, code repos)
Per rubric: "LLMs are used to analyze the README or to analyze the relationship between artifacts"
"""

import re
import os
import json
import logging
import requests
from typing import Dict, Any, Optional

MAX_INPUT_CHARS = 7000
api_key = os.getenv("GEMINI_API_KEY")
purdue_api_key = os.getenv("GEN_AI_STUDIO_API_KEY")

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

    Per JD's Q&A: "guide it to match the autograder tests" for auto-linking.
    """
    if not readme_text or not readme_text.strip():
        logger.info("No README text provided for relationship analysis")
        return {
            "linked_datasets": [],
            "linked_code_repos": [],
            "relationship_confidence": 0.0,
        }

    # Truncate README if too long
    readme_content = readme_text[:MAX_INPUT_CHARS] if len(readme_text) > MAX_INPUT_CHARS else readme_text

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
    for attempt in range(1, 4):  # up to 3 attempts
        try:
            # PRIORITY 1: Try AWS SageMaker (per rubric requirement)
            from src.aws.sagemaker_llm import get_sagemaker_service

            sagemaker_service = get_sagemaker_service()
            if sagemaker_service:
                system_prompt = (
                    "You are an engineer analyzing README files to find relationships "
                    "between ML artifacts (models, datasets, code repositories)."
                )
                logger.info(f"Relationship analysis attempt {attempt} with AWS SageMaker")
                raw = sagemaker_service.invoke_chat_model(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_tokens=1024,
                    temperature=0.1,
                )
                if raw:
                    # Clean and parse JSON response
                    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
                    analysis_json = json.loads(cleaned)
                    logger.info(f"Relationship analysis JSON parse succeeded on attempt {attempt} with SageMaker")
                    break  # Success, stop retrying

            # FALLBACK 2: Try Gemini API
            if api_key:
                from google import genai

                client = genai.Client()
                logger.info(f"Relationship analysis attempt {attempt} with Gemini (fallback)")
                response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                raw = response.text
            # FALLBACK 3: Try Purdue GenAI API
            else:
                url = "https://genai.rcac.purdue.edu/api/chat/completions"
                headers = {
                    "Authorization": f"Bearer {purdue_api_key}",
                    "Content-Type": "application/json",
                }
                body = {
                    "model": "llama4:latest",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an engineer analyzing README files to find relationships "
                                "between ML artifacts (models, datasets, code repositories)."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "max_tokens": 1024,
                }
                logger.info(f"Relationship analysis attempt {attempt} with Purdue GenAI (fallback)")
                purdue_response = requests.post(url, headers=headers, json=body, timeout=30)
                purdue_data = purdue_response.json()
                raw_content = purdue_data["choices"][0]["message"]["content"]
                raw = str(raw_content) if raw_content is not None else None

            # Clean and parse JSON response
            if raw is not None:
                cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
            else:
                cleaned = ""
            analysis_json = json.loads(cleaned)
            logger.info(f"Relationship analysis JSON parse succeeded on attempt {attempt}")
            break  # Success, stop retrying

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
