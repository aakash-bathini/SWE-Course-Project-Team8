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
    """
    # Use actual license detection logic

    gh = getattr(ctx, "gh_data", None) or []  # list of github profiles
    hf = (ctx.hf_data or [{}])[0] if ctx.hf_data else {}

    # Try GitHub data first
    if gh and gh[0]:
        gh_profile = gh[0]
        readme_text: Optional[str] = gh_profile.get("readme_text")
        doc_texts: Dict[str, str] = gh_profile.get("doc_texts") or {}
        gh_spdx: Optional[str] = gh_profile.get("license_spdx")
    else:
        # Fall back to HF data
        readme_text: Optional[str] = hf.get("readme_text")
        doc_texts: Dict[str, str] = {}
        gh_spdx: Optional[str] = hf.get("license")

    def compute() -> float:
        license_text = _select_license_text(doc_texts)
        source, spdx_ids, spdx_exprs, hints = extract_license_evidence(readme_text, license_text)

        if gh_spdx and not spdx_ids:
            spdx_ids = [gh_spdx]  # use github's detected license if nothing else found
        score, rationale = (
            spdx.classify_license(spdx_ids[0]) if spdx_ids else (0.0, "No license found")
        )
        try:
            logging.info(
                f"license_check: source={source}, spdx_ids={spdx_ids}, hints={hints} => {rationale}"
            )
            return float(score)
        except Exception:
            return 0.0

    return await asyncio.to_thread(compute)
