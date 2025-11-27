"""
Reproducibility Metric - Phase 2
Evaluates reproducibility based on available artifacts and documentation

Scoring:
- 0.0: No reproducibility artifacts
- 0.3-0.5: Some config/training info available
- 0.6-0.8: Good documentation and config files
- 0.9-1.0: Excellent reproducibility (all artifacts + code)
"""

import logging
import re
from typing import Optional, Dict, Any, List
from src.models.model_types import EvalContext

logger = logging.getLogger(__name__)


async def metric(context: EvalContext) -> float:
    """
    Calculate reproducibility score based on available artifacts and metadata

    Checks:
    1. HF config files (config.json, model files, tokenizer files)
    2. Training parameters in card_yaml
    3. Code examples in README
    4. Tags indicating training framework/method
    5. GitHub repo with training scripts
    """
    try:
        hf = (context.hf_data or [{}])[0] if context.hf_data else {}
        gh = (context.gh_data or [{}])[0] if context.gh_data else {}

        score = 0.0

        # PRIORITY 1: HF config files (strong signal of reproducibility)
        files = hf.get("files", [])
        config_score = _check_config_files(files)
        score += config_score
        logger.info(f"Config files score: {config_score:.2f}")

        # PRIORITY 2: Training parameters in card_yaml
        card_yaml = hf.get("card_yaml", {}) or {}
        training_score = _check_training_params(card_yaml)
        score += training_score
        logger.info(f"Training params score: {training_score:.2f}")

        # PRIORITY 3: Tags indicating reproducibility
        tags = hf.get("tags", []) or []
        tag_score = _check_reproducibility_tags(tags)
        score += tag_score
        logger.info(f"Tags score: {tag_score:.2f}")

        # PRIORITY 4: Code examples in README
        readme_text = hf.get("readme_text") or gh.get("readme_text") or ""
        code_score = _check_code_examples(readme_text)
        score += code_score
        logger.info(f"Code examples score: {code_score:.2f}")

        # PRIORITY 5: GitHub repo bonus (training scripts)
        if gh:
            gh_score = _check_github_reproducibility(gh)
            score += gh_score
            logger.info(f"GitHub score: {gh_score:.2f}")

        # Cap at 1.0
        score = min(1.0, score)

        # Apply engagement-based adjustments
        downloads = hf.get("downloads", 0)
        likes = hf.get("likes", 0)

        if downloads > 1000000 or likes > 1000:
            # Very popular models likely have good reproducibility
            score = min(1.0, score + 0.15)
            logger.info(f"High-engagement boost applied")

        logger.info(f"Final reproducibility score: {score:.2f}")
        return round(score, 2)

    except Exception as e:
        logger.error(f"Reproducibility metric error: {e}")
        return 0.0


def _check_config_files(files: List[Dict[str, Any]]) -> float:
    """
    Check for presence of config files that enable reproducibility
    """
    if not files:
        return 0.0

    score = 0.0
    file_paths = [f.get("path", "").lower() for f in files if isinstance(f, dict)]

    # Essential config files
    if any("config.json" in p for p in file_paths):
        score += 0.25
    if any("tokenizer" in p and ".json" in p for p in file_paths):
        score += 0.10
    if any("model" in p and (".safetensors" in p or ".bin" in p or ".pt" in p) for p in file_paths):
        score += 0.15
    if any("pytorch_model" in p or "tf_model" in p or "flax_model" in p for p in file_paths):
        score += 0.10

    return min(0.30, score)


def _check_training_params(card_yaml: Dict[str, Any]) -> float:
    """
    Check card_yaml for training parameters and hyperparameters
    """
    if not card_yaml:
        return 0.0

    score = 0.0

    # Check for training-related fields
    if "training" in card_yaml or "train" in card_yaml:
        score += 0.15
    if "hyperparameters" in card_yaml or "training_hyperparameters" in card_yaml:
        score += 0.10
    if "model-index" in card_yaml:
        # model-index often contains evaluation results indicating reproducibility
        score += 0.10
    if "base_model" in card_yaml:
        # Fine-tuned models with base_model info are more reproducible
        score += 0.05

    return min(0.20, score)


def _check_reproducibility_tags(tags: List[str]) -> float:
    """
    Check tags for reproducibility indicators
    """
    if not tags:
        return 0.0

    score = 0.0
    tags_lower = [t.lower() for t in tags if isinstance(t, str)]

    # Framework tags indicate code is available
    frameworks = ["pytorch", "tensorflow", "jax", "flax", "keras"]
    if any(fw in t for t in tags_lower for fw in frameworks):
        score += 0.10

    # Training method tags
    methods = ["fine-tuned", "trained", "pretrained"]
    if any(m in t for t in tags_lower for m in methods):
        score += 0.05

    return min(0.15, score)


def _check_code_examples(readme_text: str) -> float:
    """
    Check README for code examples that aid reproducibility
    """
    if not readme_text:
        return 0.0

    score = 0.0

    # Find Python code blocks
    code_pattern = r"```(?:python|py)\s*\n(.*?)```"
    matches = re.findall(code_pattern, readme_text, re.DOTALL | re.IGNORECASE)

    if matches:
        # Count substantial code blocks
        substantial = [m for m in matches if len(m.strip()) > 30]
        if len(substantial) >= 3:
            score = 0.20
        elif len(substantial) >= 1:
            score = 0.15
        else:
            score = 0.10

    # Check for specific keywords indicating reproducibility
    text_lower = readme_text.lower()
    if "reproduce" in text_lower or "replicat" in text_lower:
        score += 0.05

    return min(0.25, score)


def _check_github_reproducibility(gh_data: Dict[str, Any]) -> float:
    """
    Check GitHub repo for training scripts and reproducibility artifacts
    """
    if not gh_data:
        return 0.0

    score = 0.0

    # Check for README
    if gh_data.get("readme_text"):
        score += 0.05

    # Check for training-related files (if we have file paths)
    doc_texts = gh_data.get("doc_texts", {}) or {}
    file_names = [k.lower() for k in doc_texts.keys()]

    if any("train" in f for f in file_names):
        score += 0.05
    if any("requirements" in f or "environment" in f for f in file_names):
        score += 0.05

    return min(0.10, score)
