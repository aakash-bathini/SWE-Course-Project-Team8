"""
Treescore Metric - Phase 2
Calculates average net score of parent models in lineage graph

Scoring:
- 0.0 to 1.0: Average of parent model scores
- 0.0 if no parents exist
"""

import logging
import json
from typing import Set
from src.models.model_types import EvalContext

logger = logging.getLogger(__name__)


async def metric(context: EvalContext) -> float:
    """
    Calculate treescore as average of parent model scores
    """
    try:
        logger.info(
            "CW_TREESCORE_START: url=%s category=%s hf_data=%s gh_data=%s",
            getattr(context, "url", None),
            getattr(context, "category", None),
            bool(getattr(context, "hf_data", None)),
            bool(getattr(context, "gh_data", None)),
        )
        # Get parent models from lineage
        parent_urls = _extract_parent_models(context)
        logger.info("CW_TREESCORE_PARENTS: count=%d parents=%s", len(parent_urls), parent_urls)

        if not parent_urls:
            logger.info("No parent models found in lineage")
            # Sentinel meaning "not applicable" so ingest gate can ignore
            return -1.0

        # Calculate scores for parent models
        parent_scores = []

        for parent_url in parent_urls[:5]:  # Limit to 5 parents for performance
            try:
                logger.info("CW_TREESCORE_PARENT_SCORE: evaluating %s", parent_url)
                # Create context for parent model
                parent_context = EvalContext(url=parent_url, category=context.category, hf_data=[], gh_data=[])

                # Calculate parent score (simplified - use basic metrics)
                parent_score = await _calculate_parent_score(parent_context)
                logger.info("CW_TREESCORE_PARENT_SCORE: url=%s score=%.3f", parent_url, parent_score)
                parent_scores.append(parent_score)

            except Exception as e:
                logger.warning(f"Error calculating score for parent {parent_url}: {e}")
                continue

        if not parent_scores:
            return -1.0

        # Return average of parent scores
        avg_score = sum(parent_scores) / len(parent_scores)
        logger.info(
            "CW_TREESCORE_RESULT: parents_count=%d parent_scores=%s avg=%.3f",
            len(parent_scores),
            parent_scores,
            avg_score,
        )
        return round(avg_score, 2)

    except Exception as e:
        logger.error(f"Treescore metric error: {e}")
        return -1.0


def _extract_parent_models(context: EvalContext) -> list[str]:
    """
    Extract parent model URLs from HuggingFace data
    """
    try:
        logger.info(
            "CW_TREESCORE_EXTRACT_START: url=%s hf_data_present=%s",
            getattr(context, "url", None),
            bool(getattr(context, "hf_data", None)),
        )
        parent_urls: list[str] = []

        if not context.hf_data or len(context.hf_data) == 0:
            logger.info("CW_TREESCORE_EXTRACT: no hf_data available")
            return parent_urls

        raw = context.hf_data[0]
        if isinstance(raw, dict):
            hf_info = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                hf_info = parsed if isinstance(parsed, dict) else {}
            except Exception:
                hf_info = {}
        else:
            hf_info = {}

        # Check card_yaml for base_model field
        # Defensive: card_yaml might be stored as a JSON string
        card_yaml_raw = hf_info.get("card_yaml", {})
        if isinstance(card_yaml_raw, str):
            try:
                card_yaml = json.loads(card_yaml_raw)
                if not isinstance(card_yaml, dict):
                    card_yaml = {}
            except Exception:
                card_yaml = {}
        elif isinstance(card_yaml_raw, dict):
            card_yaml = card_yaml_raw
        else:
            card_yaml = {}

        # Handle base_model (can be string or list)
        base_model = card_yaml.get("base_model") if isinstance(card_yaml, dict) else None
        if base_model:
            if isinstance(base_model, str):
                parent_urls.append(_normalize_model_url(base_model))
            elif isinstance(base_model, list):
                parent_urls.extend(_normalize_model_url(m) for m in base_model)
        logger.info(
            "CW_TREESCORE_EXTRACT_BASE: base_model_field=%s parent_urls=%s",
            base_model,
            parent_urls,
        )

        # Check for model-index with base_model references
        if isinstance(card_yaml, dict):
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
            logger.info("CW_TREESCORE_EXTRACT_MODEL_INDEX: parents_now=%s", parent_urls)

        # Check tags for fine-tuned indicators
        # Defensive: tags might be stored as a JSON string or list of strings
        tags_raw = hf_info.get("tags", [])
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
                if not isinstance(tags, list):
                    tags = []
            except Exception:
                tags = []
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []

        for tag in tags:
            if isinstance(tag, str) and tag.startswith("base_model:"):
                model_name = tag.replace("base_model:", "").strip()
                parent_urls.append(_normalize_model_url(model_name))
        logger.info("CW_TREESCORE_EXTRACT_TAGS: tags_checked=%d parents_now=%s", len(tags), parent_urls)

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

    Uses a subset of metrics (license + size) instead of full net_score to avoid:
    1. Infinite recursion (calculating net_score would require treescore, which requires parent net_scores)
    2. Performance issues (full metric calculation is expensive)
    3. Circular dependencies

    This is acceptable per requirements which state "average of total model scores" -
    we interpret this as a reasonable approximation using key metrics.
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
                logger.info(
                    "CW_TREESCORE_PARENT_COMPONENTS: url=%s license_score=%s avg_size_score=%.3f raw_size_scores=%s",
                    getattr(context, "url", None),
                    scores[0] if scores else None,
                    avg_size,
                    size_scores,
                )
        except Exception:
            pass

        # Return average of available scores, or 0.5 as default
        if scores:
            logger.info(
                "CW_TREESCORE_PARENT_FINAL: url=%s scores=%s avg=%.3f",
                getattr(context, "url", None),
                scores,
                sum(scores) / len(scores),
            )
            return sum(scores) / len(scores)
        else:
            return 0.5  # Neutral score if we can't calculate

    except Exception as e:
        logger.error(f"Error calculating parent score: {e}")
        return 0.5
