"""
Reviewedness Metric - Phase 2
Calculates fraction of code introduced through reviewed pull requests

Scoring:
- -1.0: No GitHub repository linked
- 0.0 to 1.0: Fraction of code committed via reviewed PRs
"""

import logging
import json
import requests
import os
from typing import Optional
from urllib.parse import urlparse
from src.models.model_types import EvalContext

logger = logging.getLogger(__name__)


async def metric(context: EvalContext) -> float:
    """
    Calculate reviewedness score based on GitHub PR review statistics
    """
    try:
        # Get GitHub repository URL
        github_url = _extract_github_url(context)

        if not github_url:
            logger.info("No GitHub repository found")
            return -1.0

        # Parse owner and repo from URL
        owner, repo = _parse_github_url(github_url)

        if not owner or not repo:
            logger.warning(f"Could not parse GitHub URL: {github_url}")
            return -1.0

        # Calculate review statistics
        review_fraction = _calculate_review_fraction(owner, repo)

        return review_fraction

    except Exception as e:
        logger.error(f"Reviewedness metric error: {e}")
        return -1.0


def _extract_github_url(context: EvalContext) -> Optional[str]:
    """
    Extract GitHub repository URL from context
    """
    try:
        # Check URL directly
        if context.url and "github.com" in context.url.lower():
            return context.url

        # Check HF data for GitHub links
        if context.hf_data and len(context.hf_data) > 0:
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

            # Check github_links field
            github_links = hf_info.get("github_links", [])
            if github_links and len(github_links) > 0:
                return github_links[0]

            # Check card_yaml for repository field
            # Defensive: card_yaml might be stored as a JSON string
            card_yaml_raw = hf_info.get("card_yaml", {})
            if isinstance(card_yaml_raw, str):
                try:
                    import json
                    card_yaml = json.loads(card_yaml_raw)
                    if not isinstance(card_yaml, dict):
                        card_yaml = {}
                except Exception:
                    card_yaml = {}
            elif isinstance(card_yaml_raw, dict):
                card_yaml = card_yaml_raw
            else:
                card_yaml = {}
            repo_url = card_yaml.get("repository") or card_yaml.get("repo") if isinstance(card_yaml, dict) else None
            if repo_url and urlparse(repo_url).hostname == "github.com":
                return repo_url

        return None

    except Exception as e:
        logger.error(f"Error extracting GitHub URL: {e}")
        return None


def _parse_github_url(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse owner and repo name from GitHub URL
    Returns: (owner, repo)
    """
    try:
        # Remove trailing slashes and .git
        url = url.rstrip("/").replace(".git", "")

        # Extract path from URL
        parts = url.split("github.com/")
        if len(parts) < 2:
            return None, None

        path_parts = parts[1].split("/")
        if len(path_parts) < 2:
            return None, None

        owner = path_parts[0]
        repo = path_parts[1]

        return owner, repo

    except Exception as e:
        logger.error(f"Error parsing GitHub URL: {e}")
        return None, None


def _calculate_review_fraction(owner: str, repo: str) -> float:
    """
    Calculate fraction of commits that came through reviewed PRs
    """
    try:
        # Get GitHub token if available (optional)
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        # Get repository statistics
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"

        # Get commit count (sample recent commits)
        commits_url = f"{repo_url}/commits?per_page=100"
        commits_response = requests.get(commits_url, headers=headers, timeout=10)

        if commits_response.status_code != 200:
            logger.warning(f"GitHub API error: {commits_response.status_code}")
            return -1.0

        commits = commits_response.json()
        total_commits = len(commits)

        if total_commits == 0:
            return 0.0

        # Count commits associated with reviewed PRs
        reviewed_commits = 0

        for commit in commits[:50]:  # Sample first 50 for performance
            sha = commit.get("sha")
            if not sha:
                continue

            # Check if commit is associated with a PR
            pr_url = f"{repo_url}/commits/{sha}/pulls"
            pr_response = requests.get(pr_url, headers=headers, timeout=5)

            if pr_response.status_code == 200:
                prs = pr_response.json()

                # Check if any associated PR had reviews
                for pr in prs:
                    pr_number = pr.get("number")
                    if pr_number:
                        reviews_url = f"{repo_url}/pulls/{pr_number}/reviews"
                        reviews_response = requests.get(reviews_url, headers=headers, timeout=5)

                        if reviews_response.status_code == 200:
                            reviews = reviews_response.json()
                            if len(reviews) > 0:
                                reviewed_commits += 1
                                break

        # Calculate fraction based on sample
        sample_size = min(50, total_commits)
        review_fraction = reviewed_commits / sample_size if sample_size > 0 else 0.0

        logger.info(
            f"Reviewedness for {owner}/{repo}: {review_fraction:.2f} " f"({reviewed_commits}/{sample_size} commits)"
        )

        return round(review_fraction, 2)

    except requests.RequestException as e:
        logger.error(f"GitHub API request error: {e}")
        return -1.0
    except Exception as e:
        logger.error(f"Error calculating review fraction: {e}")
        return -1.0
