import asyncio
import logging
from typing import Optional, Dict
from src.models.model_types import EvalContext
from src.config_parsers_nlp.readme_parser import extract_license_evidence
import src.config_parsers_nlp.spdx as spdx

_PREFERRED_LICENSE_NAMES = (
    "LICENSE",
    "LICENSE.TXT",
    "LICENSE.MD",
    "LICENCE",
    "LICENCE.TXT",
    "LICENCE.MD",
    "COPYING",
    "COPYRIGHT",
    "UNLICENSE",
)


def _select_license_text(doc_texts: dict[str, str]) -> Optional[str]:
    """Choose the best license text from a dict of path -> text."""
    if not doc_texts:
        return None  # nothing available in repo
    upper_map = {path.upper(): text for path, text in doc_texts.items()}  # case insensitive match
    # exact filename match
    for name in _PREFERRED_LICENSE_NAMES:
        for path_upper, text in upper_map.items():
            if path_upper.endswith(f"/{name}") or path_upper == name:
                return text
    # loose basename match
    for name in _PREFERRED_LICENSE_NAMES:
        for path_upper, text in upper_map.items():
            basename = path_upper.rsplit("/", 1)[-1]
            if basename.startswith(name):
                return text
    return None  # no good match


async def metric(ctx: EvalContext) -> float:
    """
    License check metric
    -returns a score in [0.0, 1.0] based on license presence and compliance
    -consumes Github data from EvalContext (ctx.github) and HF data
    
    Priority order:
    1. HF license field (most reliable for HF models)
    2. GitHub license_spdx (reliable for GitHub repos)
    3. README/License file parsing (fallback)
    """
    gh = getattr(ctx, "gh_data", None) or []  # list of github profiles
    hf = (ctx.hf_data or [{}])[0] if ctx.hf_data else {}

    # PRIORITY 1: Check HF license field first (most common case for HF models)
    hf_license = hf.get("license")
    if hf_license and isinstance(hf_license, str) and hf_license.strip():
        # HF provides license, classify it directly
        score, rationale = spdx.classify_license(hf_license.strip())
        logging.info(f"license_check: Using HF license field '{hf_license}' => {rationale} (score: {score})")
        return float(score)
    
    # PRIORITY 2: Check GitHub license_spdx
    gh_spdx: Optional[str] = None
    if gh and gh[0]:
        gh_profile = gh[0]
        gh_spdx = gh_profile.get("license_spdx")
        if gh_spdx and isinstance(gh_spdx, str) and gh_spdx.strip():
            score, rationale = spdx.classify_license(gh_spdx.strip())
            logging.info(f"license_check: Using GitHub license_spdx '{gh_spdx}' => {rationale} (score: {score})")
            return float(score)
    
    # PRIORITY 3: Fall back to README/License file parsing (most expensive)
    readme_text: Optional[str] = None
    doc_texts: Dict[str, str] = {}
    
    if gh and gh[0]:
        gh_profile = gh[0]
        readme_text = gh_profile.get("readme_text")
        doc_texts = gh_profile.get("doc_texts") or {}
    else:
        # Use HF README if available
        readme_text = hf.get("readme_text")
        doc_texts = {}

    def compute() -> float:
        license_text = _select_license_text(doc_texts)
        source, spdx_ids, spdx_exprs, hints = extract_license_evidence(readme_text, license_text)
        
        if spdx_ids:
            score, rationale = spdx.classify_license(spdx_ids[0])
            logging.info(f"license_check: Parsed from {source}, spdx_ids={spdx_ids}, hints={hints} => {rationale}")
            return float(score)
        
        # No license found anywhere
        logging.info("license_check: No license found in HF metadata, GitHub, README, or license files")
        return 0.0

    return await asyncio.to_thread(compute)
