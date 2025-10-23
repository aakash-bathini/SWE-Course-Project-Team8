import os
import sys
import io
import json
import types
import pytest
import requests
from pathlib import Path
from typing import Any
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory

import src.metrics.size as model_size
from src.api.prep_eval_context import prepare_eval_context
from src.commands import url_file_cmd
from src.models.model_types import OrchestrationReport, MetricRun, EvalContext
from src.scoring.net_score import bundle_from_report, subscores_from_results
from src.orchestration import logging_util
import src.api.huggingface as hf
import src.api.github as gh

# -------------------------------------------------------------------
# LOGGING_UTIL.PY
# -------------------------------------------------------------------


def test_logfile_missing_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    log_file = tmp_path / "doesnotexist.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    with pytest.raises(SystemExit) as e:
        url_file_cmd.run_eval("nonexistent.txt")
    assert e.value.code == 1


def test_logging_levels(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    log_file = tmp_path / "log.txt"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "1")
    lvl = logging_util.setup_logging_util(False)
    assert lvl == 1
    assert log_file.exists()


def test_logging_level0(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    log_file = tmp_path / "log0.txt"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "0")
    lvl = logging_util.setup_logging_util(False)
    assert lvl == 0
    assert log_file.exists()
    assert log_file.read_text() == ""


# -------------------------------------------------------------------
# URL_FILE_CMD.PY
# -------------------------------------------------------------------


def test_parse_urls_utf8_tokens(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text("huggingface.co/google-bert/bert-base-uncased\n")
    urls = url_file_cmd.parse_urls_from_file(str(url_file))
    assert urls[0].startswith("https://")


def test_parse_urls_missing_file_returns_empty() -> None:
    urls = url_file_cmd.parse_urls_from_file("nope.txt")
    assert urls == []


def test_parse_urls_non_utf8_file(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    # write with utf-16 so that utf-8 decoder produces garbage
    url_file.write_text("☃", encoding="utf-16")
    urls = url_file_cmd.parse_urls_from_file(str(url_file))
    # We don’t care what garbage it produces, only that it’s a list of strings and not a crash
    assert isinstance(urls, list)
    assert all(isinstance(u, str) for u in urls)


def test_normalize_url_variants() -> None:
    u1 = "https://huggingface.co/org/model/tree/main"
    u2 = "https://github.com/org/repo/blob/main/file.py"
    assert url_file_cmd.normalize_url(u1) == "https://huggingface.co/org/model"
    assert url_file_cmd.normalize_url(u2) == "https://github.com/org/repo"


def test_normalize_url_passthrough() -> None:
    url = "https://example.com/foo"
    assert url_file_cmd.normalize_url(url) == url


# def test_read_lines_ascii_multiple(tmp_path: Path):
#     f = tmp_path / "urls.txt"
#     f.write_text("hf.co/a/b\ngithub.com/c/d\n", encoding="ascii")
#     urls = list(url_file_cmd._read_lines_ascii(f))
#     assert all(isinstance(u, str) for u in urls)

# def test_read_lines_ascii_blank_lines(tmp_path: Path):
#     f = tmp_path / "urls.txt"
#     f.write_text("\n\n", encoding="ascii")
#     urls = list(url_file_cmd._read_lines_ascii(f))
#     assert urls == []


def test_print_ndjson(capsys: Any) -> None:
    rep = OrchestrationReport(
        results={"bus_factor": MetricRun("bus_factor", 0.5, 12)},
        total_latency_ms=12,
    )
    url = "https://huggingface.co/foo/bar"
    ctx = EvalContext(url="test://url", category="MODEL")
    url_file_cmd.print_ndjson([url], {url: ctx}, {url: rep})
    captured = capsys.readouterr().out
    out_json = json.loads(captured.strip())
    assert out_json["name"] == "bar"
    assert "net_score" in out_json


# -------------------------------------------------------------------
# PREP_EVAL_CONTEXT.PY
# -------------------------------------------------------------------


def test_prepare_eval_context_hf(monkeypatch: MonkeyPatch) -> None:
    from src.api import prep_eval_context

    monkeypatch.setattr(
        prep_eval_context,
        "scrape_hf_url",
        lambda u: ({"readme_text": "hello", "github_links": []}, "model"),
    )
    ctx = prepare_eval_context("https://huggingface.co/foo/bar")
    assert ctx.category == "MODEL"


def test_prepare_eval_context_github(monkeypatch: MonkeyPatch) -> None:
    from src.api import prep_eval_context

    monkeypatch.setattr(prep_eval_context, "scrape_github_url", lambda u: {"repo_id": "org/repo"})
    ctx = prepare_eval_context("https://github.com/org/repo")
    assert ctx.category == "CODE"


def test_prepare_eval_context_bad(caplog: Any) -> None:
    caplog.set_level("DEBUG")
    ctx = prepare_eval_context("https://google.com")
    assert ctx.category is None
    assert "unsupported URL host" in caplog.text


def test_prepare_eval_context_none() -> None:
    with pytest.raises(ValueError):
        prepare_eval_context(None)


# -------------------------------------------------------------------
# NET_SCORE.PY
# -------------------------------------------------------------------


def test_net_score_empty_report() -> None:
    rep = OrchestrationReport(results={}, total_latency_ms=0)
    bundle = bundle_from_report(rep, {}, clamp=True)
    assert 0.0 <= bundle.net_score <= 1.0


def test_net_score_with_size_dict() -> None:
    rep = OrchestrationReport(
        results={
            "size_score": MetricRun(
                "size_score", value={"raspberry_pi": 0.5, "jetson_nano": 0.7}, latency_ms=10
            )
        },
        total_latency_ms=10,
    )
    subs = subscores_from_results(rep.results)
    assert "size_score" in subs
    bundle = bundle_from_report(rep, {"size_score": 1.0}, clamp=True)
    assert 0.0 <= bundle.net_score <= 1.0


# -------------------------------------------------------------------
# METRICS: SIZE
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_size_metric_empty_ctx() -> None:
    from src.metrics import size

    ctx = EvalContext(url="test://url", category="MODEL", hf_data=[])
    scores = await size.metric(ctx)
    assert isinstance(scores, dict)


@pytest.mark.asyncio
async def test_size_metric_with_readme_regex() -> None:
    from src.metrics import size

    ctx = EvalContext(
        url="test://url",
        category="MODEL",
        hf_data=[{"readme_text": "Requires 16GB VRAM", "card_yaml": {}, "files": []}],
        gh_data=[],
    )
    scores = await size.metric(ctx)
    assert any(v <= 1.0 for v in scores.values())


def test_sum_repo_size_from_index() -> None:
    from typing import Dict, Any
    from src.metrics.size import _sum_repo_size_from_index

    files: list[Dict[str, Any]] = [{"type": "blob", "size": 123}, {"type": "tree"}]
    assert _sum_repo_size_from_index(files) == 123


# -------------------------------------------------------------------
# DATASET_QUALITY.PY
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dataset_quality_no_data() -> None:
    from src.metrics import dataset_quality

    ctx = EvalContext(url="test://url", hf_data=[])
    score = await dataset_quality.metric(ctx)
    assert score == 0.0


@pytest.mark.asyncio
async def test_dataset_quality_with_scores() -> None:
    from src.metrics import dataset_quality

    ctx = EvalContext(
        url="test://url",
        hf_data=[
            {
                "downloads": 5000,
                "likes": 100,
                "readme_text": "dataset for testing",
                "repo_type": "dataset",
            }
        ],
    )
    score = await dataset_quality.metric(ctx)
    assert 0.0 <= score <= 1.0


# -------------------------------------------------------------------
# PERFORMANCE_METRIC.PY
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_performance_metric_no_keys(monkeypatch: MonkeyPatch) -> None:
    from src.metrics import performance_metric

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEN_AI_STUDIO_API_KEY", raising=False)
    ctx = EvalContext(url="test://url", gh_data=[{}], hf_data=[])
    score = await performance_metric.metric(ctx)
    assert score == 0.0


# -------------------------------------------------------------------
# CODE_QUALITY.PY
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_quality_metric_empty() -> None:
    from src.metrics import code_quality_metric

    ctx = EvalContext(url="test://url", gh_data=[], hf_data=[])
    score = await code_quality_metric.metric(ctx)
    assert score == 0.1  # Low-engagement models are capped at 0.1


# -------------------------------------------------------------------
# LICENSE_CHECK.PY
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_license_check_metric(monkeypatch: MonkeyPatch) -> None:
    import src.metrics.license_check as lc
    from src.config_parsers_nlp import spdx

    monkeypatch.setattr(lc, "extract_license_evidence", lambda *a, **k: ("src", ["MIT"], [], []))
    monkeypatch.setattr(spdx, "classify_license", lambda l: (1.0, "ok"))
    ctx = EvalContext(url="test://url", gh_data=[{"doc_texts": {"LICENSE": "MIT"}}], hf_data=[])
    score = await lc.metric(ctx)
    assert 0.0 <= score <= 1.0


# -------------------------------------------------------------------
# BUS_FACTOR.PY
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_factor_metric() -> None:
    import src.metrics.bus_factor_metric as bf

    ctx = EvalContext(url="test://url", gh_data=[{"contributors": {"a": 10, "b": 5}}], hf_data=[])
    score = await bf.metric(ctx)
    assert 0.0 <= score <= 1.0


# -------------------------------------------------------------------
# API: GITHUB
# -------------------------------------------------------------------


def test_parse_github_url_valid_and_invalid() -> None:
    assert gh.parse_github_url("https://github.com/user/repo") == ("user", "repo")
    with pytest.raises(ValueError):
        gh.parse_github_url("https://example.com/notgithub")
    with pytest.raises(ValueError):
        gh.parse_github_url("https://github.com/user")


def test_should_fetch_patterns() -> None:
    assert gh._should_fetch("README.md")
    assert not gh._should_fetch("binary.bin")


def test_is_fresh_and_cache(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    import src.api.github as gh

    monkeypatch.setenv("GH_META_CACHE_TTL_S", "3600")  # ensure positive TTL
    monkeypatch.setattr(gh, "_CACHE_TTL_S", 3600)
    entry = {"fetched_at": gh._now()}
    assert gh._is_fresh(entry)


# -------------------------------------------------------------------
# API: HUGGINGFACE
# -------------------------------------------------------------------


def test_parse_hf_url_valid_and_invalid() -> None:
    assert hf.parse_hf_url("https://huggingface.co/user/model") == ("model", "user/model")
    assert hf.parse_hf_url("https://huggingface.co/datasets/user/data") == ("dataset", "user/data")
    with pytest.raises(ValueError):
        hf.parse_hf_url("https://example.com/not-hf")
    with pytest.raises(ValueError):
        hf.parse_hf_url("https://huggingface.co/")


def test_extract_github_links_from_card_and_readme() -> None:
    links = hf._extract_github_links(
        "see https://github.com/u/r", {"repository": "https://github.com/u/r2"}
    )
    assert any("github.com" in l for l in links)


# -------------------------------------------------------------------
# README_PARSER.PY
# -------------------------------------------------------------------

import src.config_parsers_nlp.readme_parser as rp


def test_strip_markdown_noise_removes_code_and_comments() -> None:
    md = "```python\nprint(123)\n```\n<!--comment-->\n[id]: http://link\n`inline`"
    out = rp._strip_markdown_noise(md)
    assert "```" not in out
    assert "<!--" not in out
    assert "`inline`" not in out


def test_extract_section_found_and_not_found() -> None:
    md = "# License\nMIT\n# Other\n"
    section = rp.extract_section(md, rp.LICENSE_HX)
    assert section is not None and "MIT" in section
    assert rp.extract_section("no headings", rp.LICENSE_HX) is None


# def test_find_spdx_ids_exprs_hints():
#     text = "SPDX-License-Identifier: MIT\nApache-2.0 OR GPL-3.0-only\nThis project uses the MIT license"
#     ids = rp.find_spdx_ids(text)
#     exprs = rp.find_spdx_expressions(text)
#     hints = rp.find_license_hints(text)
#     assert "MIT" in ids
#     assert any("OR" in e for e in exprs)
#     assert any("mit" in h for h in hints)

# -------------------------------------------------------------------
# SPDX.PY
# -------------------------------------------------------------------

import src.config_parsers_nlp.spdx as spdx


def test_normalize_and_classify_license() -> None:
    assert spdx.normalize_license("MIT License") == "MIT"
    score, msg = spdx.classify_license("MIT")
    assert score == 1.0
    assert spdx.classify_license("GPL-3.0-only")[0] == 0.0
    assert spdx.classify_license("gpl")[0] == 0.3


# -------------------------------------------------------------------
# ORCHESTRATION.LOGGING_UTIL
# -------------------------------------------------------------------

import src.orchestration.logging_util as logutil


def test_setup_logging_util_with_env(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    log_file = tmp_path / "log.txt"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "2")
    lvl = logutil.setup_logging_util(also_stderr=True)
    assert lvl == 2


# -------------------------------------------------------------------
# METRIC_ORCHESTRATOR.PY
# -------------------------------------------------------------------

import src.orchestration.metric_orchestrator as mo
from src.models.model_types import EvalContext


@pytest.mark.asyncio
async def test_run_one_success_and_failure() -> None:
    async def good(ctx: EvalContext) -> float:
        return 0.5

    async def bad(ctx: EvalContext) -> float:
        raise RuntimeError("fail")

    ctx = EvalContext(url="u")
    r1 = await mo._run_one(("good", good), ctx)
    assert r1.value == 0.5
    r2 = await mo._run_one(("bad", bad), ctx)
    assert r2.error


# -------------------------------------------------------------------
# API: GITHUB
# -------------------------------------------------------------------


def test_github_rate_limit(monkeypatch: MonkeyPatch) -> None:
    import src.api.github as gh
    import requests as requests_lib

    class FakeResp:
        status_code = 403
        text = "rate limit"

        def raise_for_status(self) -> None:
            raise requests.exceptions.HTTPError("403")

    monkeypatch.setattr(requests_lib, "get", lambda *a, **k: FakeResp())
    monkeypatch.setattr(gh, "requests", requests_lib)
    with pytest.raises(requests.exceptions.HTTPError):
        gh._get_json("https://api.github.com/repos/foo/bar")


def test_github_get_json_success(monkeypatch: MonkeyPatch) -> None:
    import src.api.github as gh
    import requests as requests_lib
    from typing import Dict, Any

    class FakeResp:
        def json(self) -> Dict[str, Any]:
            return {"ok": True}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(requests_lib, "get", lambda *a, **k: FakeResp())
    monkeypatch.setattr(gh, "requests", requests_lib)
    out = gh._get_json("https://api.github.com/repos/foo/bar")
    assert out == {"ok": True}


def test_github_cache_roundtrip(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    import src.api.github as gh

    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    (tmp_path / ".cache").mkdir(parents=True, exist_ok=True)
    data = {"foo": "bar"}
    gh._save_cache(data)
    out = gh._load_cache()
    assert isinstance(out, dict)


# -------------------------------------------------------------------
# API: HUGGINGFACE
# -------------------------------------------------------------------


def test_parse_hf_url_variants() -> None:
    from src.api.huggingface import parse_hf_url

    assert parse_hf_url("https://huggingface.co/models/org/name")[0] == "model"
    assert parse_hf_url("https://huggingface.co/datasets/org/data")[0] == "dataset"
    assert parse_hf_url("https://huggingface.co/org/name")[0] == "model"
    with pytest.raises(ValueError):
        parse_hf_url("https://not-hf.com/foo")


def test_extract_github_links() -> None:
    from src.api.huggingface import _extract_github_links

    readme = "Check https://github.com/org/repo"
    card_yaml = {"repository": "https://github.com/org/other"}
    links = _extract_github_links(readme, card_yaml)
    assert any("github.com" in l for l in links)


def test_is_fresh_and_cache_helpers(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    import src.api.huggingface as hf

    monkeypatch.setenv("HF_META_CACHE_TTL_S", "3600")  # ensure positive TTL
    monkeypatch.setattr(hf, "_CACHE_TTL_S", 3600)
    entry = {"fetched_at": hf._now()}
    assert hf._is_fresh(entry)


# -------------------------------------------------------------------
# METRICS: SIZE – deeper coverage
# -------------------------------------------------------------------

import types
import pytest
import src.metrics.size as size


@pytest.mark.asyncio
async def test_size_metric_mem_requirements(monkeypatch: MonkeyPatch) -> None:
    # README mentions GB → should parse memory requirement
    ctx = EvalContext(
        url="test://url",
        category="MODEL",
        hf_data=[{"readme_text": "Needs 24GB GPU", "card_yaml": {}, "files": []}],
        gh_data=[],
    )
    scores = await size.metric(ctx)
    assert any(0.0 <= v <= 1.0 for v in scores.values())


@pytest.mark.asyncio
async def test_size_metric_small_large_values(monkeypatch: MonkeyPatch) -> None:
    # Small requirement (MB) and huge requirement (TB) to hit edges
    ctx1 = EvalContext(
        url="test://url",
        category="MODEL",
        hf_data=[{"readme_text": "Requires 512MB memory", "card_yaml": {}, "files": []}],
        gh_data=[],
    )
    scores1 = await size.metric(ctx1)
    assert all(0.0 <= v <= 1.0 for v in scores1.values())

    ctx2 = EvalContext(
        url="test://url",
        category="MODEL",
        hf_data=[{"readme_text": "Needs 1000TB GPU", "card_yaml": {}, "files": []}],
        gh_data=[],
    )
    scores2 = await size.metric(ctx2)
    assert all(0.0 <= v <= 1.0 for v in scores2.values())


def test_to_bytes_and_human() -> None:
    assert size._to_bytes(1, "GB") == 1024**3
    assert size._to_bytes(1, "GiB") == 1024**3
    assert size._to_bytes(1, "MB") == 1024**2
    assert size._to_bytes(1, "TB") == 1024**4
    assert size._to_bytes(5, "foo") == 5
    assert size._bytes_to_human(1023) == "1023.0B"
    assert size._bytes_to_human(1024) == "1.0KB"
    assert size._bytes_to_human(5 * 1024**3) == "5.0GB"


def test_flatten_card_yaml_and_sum_size() -> None:
    from typing import Dict, Any

    data = {"a": {"b": [1, 2, {"c": 3}]}}
    flat = size._flatten_card_yaml(data)
    assert "a" in flat and "b" in flat and "3" in flat

    files: list[Dict[str, Any]] = [
        {"type": "blob", "size": 100},
        {"type": "tree"},
        {"type": "blob", "size": 50},
    ]
    assert size._sum_repo_size_from_index(files) == 150


def test_hf_total_size_bytes_prefers_size() -> None:
    hf = {"size": 12345, "files": [{"size": 99}]}
    assert size._hf_total_size_bytes(hf) == 12345

    hf2 = {"files": [{"size": 10}, {"size": 20}]}
    assert size._hf_total_size_bytes(hf2) == 30


@pytest.mark.asyncio
async def test_metric_dataset_disk_and_model_fallback() -> None:
    ctx = EvalContext(
        url="test://url",
        category="DATASET",
        hf_data=[{"readme_text": "dataset size 50GB", "card_yaml": {}, "files": []}],
        gh_data=[],
    )
    scores = await size.metric(ctx)
    assert isinstance(scores, dict)

    ctx2 = EvalContext(
        url="test://url",
        category="MODEL",
        hf_data=[{"readme_text": "", "card_yaml": {}, "files": [], "size": 1024}],
        gh_data=[],
    )
    scores2 = await size.metric(ctx2)
    assert isinstance(scores2, dict)


@pytest.mark.asyncio
async def test_metric_code_paths_and_unknown() -> None:
    from src.models.model_types import Category

    cat: Category = "CODE"
    ctx = EvalContext(
        url="test://url",
        category=cat,
        hf_data=[],
        gh_data=[
            {
                "readme_text": "Requires 2GB RAM",
                "doc_texts": {},
                "files_index": [{"type": "blob", "size": 100}],
            }
        ],
    )
    scores = await size.metric(ctx)
    assert isinstance(scores, dict)

    ctx2 = EvalContext(url="test://url", category=None, hf_data=[], gh_data=[])
    scores2 = await size.metric(ctx2)
    assert isinstance(scores2, dict)


import src.metrics.registry as registry


def test_get_all_metrics_returns_list() -> None:
    metrics = registry.get_all_metrics()
    assert isinstance(metrics, list)
    assert any(name == "size_score" for name, fn in metrics)
    assert all(callable(fn) for _, fn in metrics)


# -------------------------------------------------------------------
# AVAILABLE_DATASET_CODE.PY
# -------------------------------------------------------------------
import src.metrics.available_dataset_code as adc


@pytest.mark.asyncio
async def test_dataset_and_code_metric_with_links_and_examples(tmp_path: Path) -> None:
    ctx = EvalContext(
        url="test://url",
        hf_data=[{"readme_text": "Uses SQuAD dataset", "datasets": ["squad"]}],
        gh_data=[
            {
                "readme_text": "see https://huggingface.co/datasets/foo/bar",
                "doc_texts": {"file.py": "import datasets\n"},
            }
        ],
    )
    score = await adc.metric(ctx)
    assert 0.0 <= score <= 1.0


def test_has_runnable_snippet_and_is_example_path() -> None:
    assert adc.has_runnable_snippet("```python\nprint(1)\n```")
    assert adc._is_example_path("examples/test.py")
    assert adc._is_example_path("notebook/tutorial.ipynb")
    assert not adc._is_example_path("README.md")


# -------------------------------------------------------------------
# RAMP_UP_TIME.PY
# -------------------------------------------------------------------
import src.metrics.ramp_up_time as rut


@pytest.mark.asyncio
async def test_ramp_up_time_metric_with_readme_and_examples() -> None:
    ctx = EvalContext(
        url="test://url",
        hf_data=[
            {
                "readme_text": "# Overview\nThis is a summary with usage example\n```python\nprint(1)\n```"
            }
        ],
        gh_data=[
            {
                "readme_text": "inputs outputs",
                "files_index": [{"path": "examples/demo.py", "type": "blob"}],
            }
        ],
    )
    score = await rut.metric(ctx)
    assert 0.0 <= score <= 1.0


# -------------------------------------------------------------------
# WEIGHTS.PY
# -------------------------------------------------------------------
import src.scoring.weights as weights


def test_get_weights_normalizes_sum_to_one() -> None:
    w = weights.get_weights()
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-6
    assert all(0.0 < v < 1.0 for v in w.values())


# -------------------------------------------------------------------
# CLI.PY
# -------------------------------------------------------------------
import src.commands.cli as cli


def test_cli_main_usage(capsys: Any) -> None:
    with pytest.raises(SystemExit):
        cli.main([])
    out = capsys.readouterr()
    assert "Usage" in out.err


def test_cli_eval_and_test(monkeypatch: MonkeyPatch) -> None:
    from typing import Dict, Any

    called: Dict[str, Any] = {}
    monkeypatch.setattr(cli, "run_tests", lambda: called.setdefault("test", True))
    monkeypatch.setattr(cli, "run_eval", lambda f: called.setdefault("eval", f))
    cli.main(["test"])
    cli.main(["eval", "foo.txt"])
    assert "test" in called and "eval" in called


# -------------------------------------------------------------------
# METRIC_HELPERS.PY
# -------------------------------------------------------------------
import src.config_parsers_nlp.metric_helpers as mh


def test_metric_helpers_norm_and_collect_paths() -> None:
    ctx = EvalContext(
        url="test://url",
        hf_data=[{"files": [{"path": "a/b/c.py", "size": 1}]}],
        gh_data=[
            {
                "files_index": [{"path": "x/y/z.py", "type": "blob"}],
                "doc_texts": {"README.md": "hi"},
            }
        ],
    )
    paths = mh.collect_paths(ctx)
    assert any("c.py" in p or "z.py" in p for p in paths)
    assert mh._has_any("This has install", ["install"])
    assert "a" in mh._norm_parts("a/b/c.py")

    # -------------------------------------------------------------------


# MORE COVERAGE: registry, url_file_cmd, license_check, performance_metric
# -------------------------------------------------------------------

import src.metrics.registry as registry
import src.metrics.license_check as license_check
import src.metrics.performance_metric as perf
import src.orchestration.prep_eval_orchestrator as prep_orch
import src.orchestration.metric_orchestrator as metric_orch

# ----------------- registry.py -----------------


def test_registry_get_all_metrics_includes_expected() -> None:
    metrics = registry.get_all_metrics()
    names = [name for name, _ in metrics]
    # sanity check: includes all expected metrics
    for expected in ["ramp_up_time", "bus_factor", "performance_claims", "license", "size_score"]:
        assert expected in names


# ----------------- url_file_cmd.py -----------------
import src.commands.url_file_cmd as url_file_cmd


def test_clamp01_and_lat_helpers() -> None:
    assert url_file_cmd._clamp01(1.5) == 1.0
    assert url_file_cmd._clamp01(-2) == 0.0
    assert url_file_cmd._clamp01("bad") == 0.0
    assert url_file_cmd._lat("oops") == 1


def test_display_name_and_default_record() -> None:
    u = "https://huggingface.co/org/model"
    assert url_file_cmd._display_name_from_url(u) == "model"
    record = url_file_cmd._default_record("foo", None)
    assert "net_score" in record


def test_apply_report_with_size_dict_and_string() -> None:
    from src.models.model_types import MetricRun, OrchestrationReport

    rep = OrchestrationReport(
        results={
            "size_score": MetricRun("size_score", {"raspberry_pi": 2.0}, 10),
        },
        total_latency_ms=10,
    )
    out = url_file_cmd._default_record("foo", "MODEL")
    url_file_cmd._apply_report(out, rep)
    assert out["size_score"]["raspberry_pi"] == 1.0  # clamped

    rep2 = OrchestrationReport(
        results={
            "size_score": MetricRun("size_score", "raspberry_pi", 10),
        },
        total_latency_ms=10,
    )
    out2 = url_file_cmd._default_record("bar", "MODEL")
    url_file_cmd._apply_report(out2, rep2)
    assert out2["size_score"]["raspberry_pi"] == 1.0


# ----------------- license_check.py -----------------


@pytest.mark.asyncio
async def test_license_check_metric_fallback(monkeypatch: MonkeyPatch) -> None:
    # case: no github data
    ctx = EvalContext(url="test://url", gh_data=[], hf_data=[])
    score = await license_check.metric(ctx)
    assert score == 0.0

    # case: gh profile present but no license
    ctx2 = EvalContext(url="test://url", gh_data=[{}], hf_data=[])
    score2 = await license_check.metric(ctx2)
    assert score2 == 0.0


# ----------------- performance_metric.py -----------------


@pytest.mark.asyncio
async def test_performance_metric_handles_missing_readme(monkeypatch: MonkeyPatch) -> None:
    from typing import Any

    monkeypatch.setenv("GEMINI_API_KEY", "dummy")  # force Gemini branch
    monkeypatch.setattr(perf, "api_key", "dummy")

    # Make client throw exception
    class DummyClient:
        class models:
            @staticmethod
            def generate_content(*a: Any, **k: Any) -> Any:
                raise RuntimeError("fail")

    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=DummyClient))
    ctx = EvalContext(url="test://url", hf_data=[], gh_data=[])
    score = await perf.metric(ctx)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_performance_metric_json_parse(monkeypatch: MonkeyPatch) -> None:
    from typing import Dict, Any
    import requests as requests_lib

    monkeypatch.setattr(perf, "api_key", None)
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "dummy")

    # Fake requests.post returning JSON
    class DummyResp:
        def json(self) -> Dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"summary":{"total_claims":1,"overall_evidence_quality":0.5,"overall_specificity":0.5},"claims_found":[]}'
                        }
                    }
                ]
            }

    monkeypatch.setattr(requests_lib, "post", lambda *a, **k: DummyResp())
    monkeypatch.setattr(perf, "requests", requests_lib)
    ctx = EvalContext(
        url="test://url", hf_data=[{"readme_text": "achieves 90% accuracy"}], gh_data=[]
    )
    score = await perf.metric(ctx)
    assert 0.0 <= score <= 1.0


# ----------------- prep_eval_orchestrator.py -----------------


@pytest.mark.asyncio
async def test_prep_eval_many_runs(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(prep_orch, "prepare_eval_context", lambda url: EvalContext(url=url))
    urls = ["https://huggingface.co/org/model", "https://github.com/org/repo"]
    result = await prep_orch.prep_eval_many(urls, limit=2)
    assert set(result.keys()) == set(urls)


# ----------------- metric_orchestrator.py -----------------


@pytest.mark.asyncio
async def test_orchestrate_with_dummy_metrics(monkeypatch: MonkeyPatch) -> None:
    import src.orchestration.metric_orchestrator as mo

    async def fake_metric(ctx: EvalContext) -> float:
        return 0.7

    # Patch inside metric_orchestrator, not registry
    monkeypatch.setattr(mo, "get_all_metrics", lambda: [("dummy", fake_metric)])

    ctx = EvalContext(url="u", category="MODEL", hf_data=[], gh_data=[])
    rep = await mo.orchestrate(ctx, limit=1)

    assert "dummy" in rep.results
    assert rep.results["dummy"].value == 0.7


# -------------------------------------------------------------------
# MODEL_SIZE.PY
# -------------------------------------------------------------------
import pytest
import types
import src.metrics.size as ms


def test_to_bytes_and_bytes_to_human_all_units() -> None:
    assert ms._to_bytes(1, "K") == 1024
    assert ms._to_bytes(1, "KB") == 1024
    assert ms._to_bytes(1, "KiB") == 1024
    assert ms._to_bytes(1, "MB") == 1024**2
    assert ms._to_bytes(1, "MiB") == 1024**2
    assert ms._to_bytes(1, "GB") == 1024**3
    assert ms._to_bytes(1, "gigabytes") == 1024**3
    assert ms._to_bytes(1, "TB") == 1024**4
    assert ms._to_bytes(5, "foo") == 5

    # human-readable
    assert ms._bytes_to_human(512) == "512.0B"
    assert ms._bytes_to_human(2048).endswith("KB")
    assert ms._bytes_to_human(5 * 1024**2).endswith("MB")
    assert ms._bytes_to_human(7 * 1024**3).endswith("GB")
    assert ms._bytes_to_human(5 * 1024**4).endswith("TB")


def test_sum_repo_size_and_hf_total_size() -> None:
    from typing import Any, Dict, List

    files: List[Dict[str, Any]] = [
        {"type": "blob", "size": 100},
        {"type": "tree"},
        {"type": "blob", "size": 50},
    ]
    assert ms._sum_repo_size_from_index(files) == 150

    hf = {"size": 123}
    assert ms._hf_total_size_bytes(hf) == 123
    hf2 = {"files": [{"size": 10}, {"size": 20}]}
    assert ms._hf_total_size_bytes(hf2) == 30
    hf3: Dict[str, Any] = {}
    assert ms._hf_total_size_bytes(hf3) == 0


def test_flatten_card_yaml_nested() -> None:
    card = {"a": {"b": [1, {"c": 2}]}, "d": "val"}
    flat = ms._flatten_card_yaml(card)
    assert "a" in flat and "1" in flat and "2" in flat and "val" in flat


# def test_scan_values_single_range_multi():
#     text1 = "requires 16 GB ram"
#     text2 = "needs 8-16GB memory"
#     text3 = "2x 8GB vram"
#     single = ms._scan_values(ms.MEM_REQ_REGEXES, text1)
#     rng = ms._scan_values(ms.MEM_REQ_REGEXES, text2)
#     multi = ms._scan_values(ms.MEM_REQ_REGEXES, text3)
#     assert single >= 16*1024**3
#     assert rng >= 16*1024**3
#     assert multi >= 16*1024**3  # 2x8GB

# def test_extract_mem_and_disk_reqs():
#     text = "recommended: 32 GB ram"
#     assert ms._extract_mem_requirements(text) >= 32*1024**3
#     text2 = "dataset size: 10 GB disk"
#     assert ms._extract_disk_requirements(text2) >= 10*1024**3


def test_score_required_vs_budget_and_best_device() -> None:
    budgets = {"raspberry_pi": 100}
    scores0 = ms._score_required_vs_budget(0, budgets, 0.5)
    scores1 = ms._score_required_vs_budget(40, budgets, 0.5)  # <=cap
    scores2 = ms._score_required_vs_budget(90, budgets, 0.5)  # >cap
    assert scores0["raspberry_pi"] == 1.0
    assert scores1["raspberry_pi"] == 1.0
    assert 0.0 <= scores2["raspberry_pi"] <= 1.0

    # tie preference
    s = {"raspberry_pi": 1.0, "desktop_pc": 1.0}
    assert ms._best_device(s) == "raspberry_pi"


# ------------------ metric tests ------------------


@pytest.mark.asyncio
async def test_metric_model_empty_and_hf_size() -> None:
    ctx = EvalContext(url="test://url", category="MODEL", hf_data=[])
    scores = await ms.metric(ctx)
    assert all(v == 1.0 for v in scores.values())

    # fallback to hf.size
    ctx2 = EvalContext(
        url="test://url",
        category="MODEL",
        hf_data=[{"size": 1234, "card_yaml": {}, "readme_text": ""}],
    )
    scores2 = await ms.metric(ctx2)
    assert isinstance(scores2, dict)


@pytest.mark.asyncio
async def test_metric_model_explicit_mem() -> None:
    ctx = EvalContext(
        url="test://url",
        category="MODEL",
        hf_data=[{"readme_text": "Requires 24GB VRAM", "card_yaml": {}, "files": []}],
    )
    scores = await ms.metric(ctx)
    assert isinstance(scores, dict)
    assert "raspberry_pi" in scores


@pytest.mark.asyncio
async def test_metric_dataset_disk_and_fallback() -> None:
    ctx = EvalContext(
        url="test://url",
        category="DATASET",
        hf_data=[{"readme_text": "Dataset size 50GB disk", "card_yaml": {}, "files": []}],
    )
    scores = await ms.metric(ctx)
    assert isinstance(scores, dict)

    ctx2 = EvalContext(
        url="test://url", category="DATASET", hf_data=[{"size": 999, "card_yaml": {}, "files": []}]
    )
    scores2 = await ms.metric(ctx2)
    assert isinstance(scores2, dict)


@pytest.mark.asyncio
async def test_metric_code_empty_and_paths() -> None:
    ctx = EvalContext(url="test://url", category="CODE", gh_data=[])
    scores = await ms.metric(ctx)
    assert all(v == 1.0 for v in scores.values())

    # explicit mem in readme
    gh = {"readme_text": "Needs 8GB RAM", "doc_texts": {"f": "hi"}, "files_index": []}
    ctx2 = EvalContext(url="test://url", category="CODE", gh_data=[gh])
    scores2 = await ms.metric(ctx2)
    assert isinstance(scores2, dict)

    # explicit disk in readme
    gh2 = {"readme_text": "Dataset size: 20 GB disk", "doc_texts": {}, "files_index": []}
    ctx3 = EvalContext(url="test://url", category="CODE", gh_data=[gh2])
    scores3 = await ms.metric(ctx3)
    assert isinstance(scores3, dict)

    # fallback repo size
    gh3 = {"readme_text": "", "doc_texts": {}, "files_index": [{"type": "blob", "size": 500}]}
    ctx4 = EvalContext(url="test://url", category="CODE", gh_data=[gh3])
    scores4 = await ms.metric(ctx4)
    assert isinstance(scores4, dict)


@pytest.mark.asyncio
async def test_metric_other_and_exception(monkeypatch: MonkeyPatch) -> None:
    # unknown category fallback
    ctx = EvalContext(url="test://url", category=None, hf_data=[], gh_data=[])
    scores = await ms.metric(ctx)
    assert all(isinstance(v, float) for v in scores.values())

    # exception path: break _flatten_card_yaml
    bad_ctx = EvalContext(url="test://url", category="MODEL", hf_data=[{"card_yaml": object()}])
    scores2 = await ms.metric(bad_ctx)
    assert isinstance(scores2, dict)
