import pytest
from urllib.parse import urlparse

from src.models.model_types import EvalContext
from src.metrics import size as size_metric
from src.metrics import (
    available_dataset_code,
    bus_factor_metric,
    dataset_quality,
    code_quality_metric,
    performance_metric,
)
from src.metrics.phase2_adapter import calculate_phase2_metrics
from src.metrics.license_check import metric as license_metric
from src.api.huggingface import parse_hf_url, _extract_github_links
from src.config_parsers_nlp.metric_helpers import collect_paths


def _sample_ctx() -> EvalContext:
    readme = (
        "This model achieves 92% accuracy on GLUE and MNLI.\n"
        "Evaluation on SQuAD and CoLA with F1 and BLEU metrics.\n"
        "| Metric | Score |\n| Acc | 0.92 |\n"
        "Requires 8GB VRAM and 20 GB disk."
    )
    card_yaml = {
        "license": "MIT",
        "repository": "https://github.com/org/repo",
        "links": ["https://github.com/org/repo/issues"],
    }
    hf = {
        "repo_type": "model",
        "readme_text": readme,
        "card_yaml": card_yaml,
        "files": [{"path": "examples/train.py", "size": 1234}, {"path": "README.md", "size": 321}],
        "size": 3_500_000_000,  # ~3.5GB
        "datasets": ["squad", "mnli"],
        "downloads": 250_000,
        "likes": 250,
    }
    gh = {
        "readme_text": "Install, Usage, Example, Test, License contents here.",
        "doc_texts": {"LICENSE": "MIT License\nPermission is hereby granted..."},
        "files_index": [
            {"path": "examples/notebook.ipynb"},
            {"path": "tutorials/quickstart.md"},
        ],
        "contributors": {"alice": 50, "bob": 40, "carl": 10},
        "maintainability_score": 0.4,
    }
    return EvalContext(
        url="https://huggingface.co/org/model", category="MODEL", hf_data=[hf], gh_data=[gh]
    )


@pytest.mark.asyncio
async def test_metrics_cover_common_paths():
    ctx = _sample_ctx()

    # Size metric across model path
    s = await size_metric.metric(ctx)
    assert isinstance(s, dict) and s

    # Dataset/code availability
    ds_code = await available_dataset_code.metric(ctx)
    assert 0.0 <= ds_code <= 1.0

    # Bus factor
    bf = await bus_factor_metric.metric(ctx)
    assert 0.0 <= bf <= 1.0

    # Dataset quality
    dq = await dataset_quality.metric(ctx)
    assert 0.0 <= dq <= 1.0

    # Code quality (heuristic path)
    cq = await code_quality_metric.metric(ctx)
    assert 0.0 <= cq <= 1.0

    # Performance claims (heuristic path)
    pf = await performance_metric.metric(ctx)
    assert 0.0 <= pf <= 1.0

    # License check
    lic = await license_metric(ctx)
    assert 0.0 <= lic <= 1.0


def test_phase2_adapter_runs_metrics():
    ctx = _sample_ctx()
    model_data = {"url": ctx.url, "hf_data": ctx.hf_data, "gh_data": ctx.gh_data}
    import asyncio

    metrics, latencies = asyncio.get_event_loop().run_until_complete(
        calculate_phase2_metrics(model_data)
    )
    assert isinstance(metrics, dict)


def test_helper_and_hf_parse():
    ctx = _sample_ctx()
    paths = collect_paths(ctx)
    assert any(p.endswith(".ipynb") for p in paths)
    # parse multiple HF URL variants
    assert parse_hf_url("https://huggingface.co/org/model")[0] == "model"
    assert parse_hf_url("https://huggingface.co/datasets/user/data")[0] == "dataset"
    # extract github links
    links = _extract_github_links(
        "Check https://github.com/user/repo for code.",
        {"repository": "https://github.com/user/repo"},
    )
    assert any(urlparse(u).hostname == "github.com" for u in links)


@pytest.mark.asyncio
async def test_additional_size_branches():
    # dataset branch
    ctx = _sample_ctx()
    ctx.category = "DATASET"
    scores_ds = await size_metric.metric(ctx)
    assert isinstance(scores_ds, dict)
    # code branch
    ctx2 = _sample_ctx()
    ctx2.category = "CODE"
    gh = ctx2.gh_data[0]
    gh["doc_texts"]["USAGE.md"] = "Requires 2 GB memory and 10GB storage."
    scores_code = await size_metric.metric(ctx2)
    assert isinstance(scores_code, dict)


def test_readme_parser_helpers():
    from src.config_parsers_nlp.readme_parser import (
        extract_license_block,
        find_spdx_ids,
        find_license_hints,
    )

    md = """
    # License
    SPDX-License-Identifier: Apache-2.0
    This project is licensed under the MIT license.
    """
    block = extract_license_block(md)
    ids = find_spdx_ids(block or md)
    hints = find_license_hints(block or md)
    assert ids and hints
