# src/api/prep_eval_context.py
from __future__ import annotations
#from urllib.parse import urlparse

from src.models.types import EvalContext, Category
#from src.api.huggingface import scrape_hf_url
#from src.api.github import scrape_github_url

def prepare_eval_context(url: str | None = None) -> EvalContext:
    """
    Builds EvalContext for HuggingFace and GitHub URLs.
        HuggingFace: returns (profile, type), category = MODEL or DATASET 
        and scrapes associated GitHub repos if present.
        
        GitHub: returns just the GitHub profile, category = CODE.
        
        Both hf_data and gh_data are always lists of dictionaries for consistency
        Potentially add github -> parse for hf link functionality 
    """
    if not url:
        raise ValueError("URL is required")
    
    #temp test for pipeline
    print(f"PREP URL: {url}")
    return EvalContext(url=url)

    # host = urlparse(url).netloc.lower()

    # # huggingface link: model or dataset
    # if "huggingface.co" in host:
    #     hf_profile, hf_type = scrape_hf_url(url)
    #     cat: Category = "MODEL" if hf_type == "model" else "DATASET"

    #     hf_data = [hf_profile]  # always a list

    #     gh_data: list[dict] = []
    #     gh_links = hf_profile.get("github_links") or []
    #     seen = set()  # avoid duplicate repos
    #     for gh_url in gh_links:
    #         try:
    #             gh_profile = scrape_github_url(gh_url)  # now returns dict only
    #             repoid = gh_profile.get("repo_id")
    #             if repoid and repoid not in seen:
    #                 seen.add(repoid)
    #                 gh_data.append(gh_profile)
    #         except Exception:
    #             continue  # keep harvesting others even if one fails

    #     return EvalContext(url=url, category=cat, hf_data=hf_data, gh_data=gh_data)

    # # github link: code
    # if "github.com" in host:
    #     gh_profile = scrape_github_url(url)
    #     gh_data = [gh_profile]  # always a list
    #     return EvalContext(url=url, category="CODE", hf_data=[], gh_data=gh_data)

    # raise ValueError(f"prep_eval_context error: unsupported URL host: {host}")
