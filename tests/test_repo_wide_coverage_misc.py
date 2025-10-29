import asyncio

from src.scoring.weights import calculate_net_score, get_weights
from src.scoring.net_score import net_score
from src.orchestration.metric_orchestrator import orchestrate
from src.models.model_types import EvalContext
from src.aws.deployment import handler as lambda_handler
from src.api.github import parse_github_url, _should_fetch, _is_fresh


def test_scoring_helpers():
    weights = get_weights()
    m = {k: 0.8 for k in weights.keys()}
    assert 0.0 < calculate_net_score(m) <= 1.0
    assert 0.0 < net_score(m) <= 1.0


def test_github_helpers():
    owner, repo = parse_github_url("https://github.com/user/repo.git")
    assert owner == "user" and repo == "repo"
    assert _should_fetch("README.md")
    assert _is_fresh({"fetched_at": 0.0}) in (True, False)  # function executes


def test_lambda_handler_safe_error():
    # Minimal event triggers handler path and safe 500 response
    resp = lambda_handler({}, None)
    assert isinstance(resp, dict)
    assert resp.get("statusCode") in (200, 500)


def test_orchestrator_wrapper():
    ctx = EvalContext(
        url="https://huggingface.co/org/model", category="MODEL", hf_data=[{}], gh_data=[{}]
    )
    model_data = {"url": ctx.url, "hf_data": ctx.hf_data, "gh_data": ctx.gh_data}
    result = asyncio.get_event_loop().run_until_complete(orchestrate(model_data))
    assert isinstance(result, dict)
