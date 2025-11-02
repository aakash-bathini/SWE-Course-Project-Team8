"""
Treescore Metric - Phase 2
Calculates average net score of parent models in lineage graph

Scoring:
- 0.0 to 1.0: Average of parent model scores
- 0.0 if no parents exist
"""

import logging
from typing import Set
from src.models.model_types import EvalContext

logger = logging.getLogger(__name__)


async def metric(context: EvalContext) -> float:
    """
    Calculate treescore as average of parent model scores
    """
    try:
        # Get parent models from lineage
        parent_urls = _extract_parent_models(context)

        if not parent_urls:
            logger.info("No parent models found in lineage")
            return 0.0

        # Calculate scores for parent models
        parent_scores = []

        for parent_url in parent_urls[:5]:  # Limit to 5 parents for performance
            try:
                # Create context for parent model
                parent_context = EvalContext(
                    url=parent_url, category=context.category, hf_data=[], gh_data=[]
                )

                # Calculate parent score (simplified - use basic metrics)
                parent_score = await _calculate_parent_score(parent_context)
                parent_scores.append(parent_score)

            except Exception as e:
                logger.warning(f"Error calculating score for parent {parent_url}: {e}")
                continue

        if not parent_scores:
            return 0.0

        # Return average of parent scores
        avg_score = sum(parent_scores) / len(parent_scores)
        return round(avg_score, 2)

    except Exception as e:
        logger.error(f"Treescore metric error: {e}")
        return 0.0


def _extract_parent_models(context: EvalContext) -> list[str]:
    """
    Extract parent model URLs from HuggingFace data
    """
    try:
        parent_urls = []

        if not context.hf_data or len(context.hf_data) == 0:
            return parent_urls

        hf_info = context.hf_data[0]

        # Check card_yaml for base_model field
        card_yaml = hf_info.get("card_yaml", {})

        # Handle base_model (can be string or list)
        base_model = card_yaml.get("base_model")
        if base_model:
            if isinstance(base_model, str):
                parent_urls.append(_normalize_model_url(base_model))
            elif isinstance(base_model, list):
                parent_urls.extend(_normalize_model_url(m) for m in base_model)

        # Check for model-index with base_model references
        model_index = card_yaml.get("model-index", [])
        if isinstance(model_index, list):
            for entry in model_index:
                if isinstance(entry, dict):
                    results = entry.get("results", [])
                    if isinstance(results, list):
                        for result in results:
                            if isinstance(result, dict):
                                dataset_name = result.get("dataset", {})
                                if isinstance(dataset_name, dict):
                                    name = dataset_name.get("name")
                                    if name and "/" in name:
                                        parent_urls.append(_normalize_model_url(name))

        # Check tags for fine-tuned indicators
        tags = hf_info.get("tags", [])
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("base_model:"):
                model_name = tag.replace("base_model:", "").strip()
                parent_urls.append(_normalize_model_url(model_name))

        # Remove duplicates while preserving order
        seen: Set[str] = set()
        unique_parents = []
        for url in parent_urls:
            if url not in seen:
                seen.add(url)
                unique_parents.append(url)

        return unique_parents

    except Exception as e:
        logger.error(f"Error extracting parent models: {e}")
        return []


def _normalize_model_url(model_identifier: str) -> str:
    """
    Convert model identifier to full HuggingFace URL
    """
    if model_identifier.startswith("http"):
        return model_identifier

    # Assume it's a HuggingFace model ID (owner/model)
    return f"https://huggingface.co/{model_identifier}"


async def _calculate_parent_score(context: EvalContext) -> float:
    """
    Calculate simplified score for parent model
    Uses a subset of metrics to avoid recursion and performance issues
    """
    try:
        # Import metrics here to avoid circular imports
        from src.metrics import license_check, size

        scores = []

        # Calculate license score
        try:
            license_score = await license_check.metric(context)
            scores.append(license_score)
        except Exception:
            pass

        # Calculate size score (use average of device scores)
        try:
            size_scores = await size.metric(context)
            if isinstance(size_scores, dict) and size_scores:
                avg_size = sum(size_scores.values()) / len(size_scores)
                scores.append(avg_size)
        except Exception:
            pass

        # Return average of available scores, or 0.5 as default
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.5  # Neutral score if we can't calculate

    except Exception as e:
        logger.error(f"Error calculating parent score: {e}")
        return 0.5
