import os
import sys
import io
import json
import tempfile
import pytest
from pathlib import Path

import src.metrics.size as model_size
from src.api.prep_eval_context import prepare_eval_context
from src.commands import url_file_cmd
from src.models.types import OrchestrationReport, MetricRun
from src.scoring.net_score import bundle_from_report, subscores_from_results
from src.orchestration import logging_util

# ----------------------
# Logging tests
# ----------------------

def test_logfile_missing_env(monkeypatch, tmp_path):
    log_file = tmp_path / "doesnotexist.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    # Should exit 1 if log file does not exist
    with pytest.raises(SystemExit) as e:
        url_file_cmd.run_eval("nonexistent.txt")
    assert e.value.code == 1

def test_logging_levels(monkeypatch, tmp_path):
    log_file = tmp_path / "log.txt"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "1")
    lvl = logging_util.setup_logging_util(False)
    assert lvl == 1
    assert log_file.exists()

def test_logging_level0(monkeypatch, tmp_path):
    log_file = tmp_path / "log0.txt"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "0")
    lvl = logging_util.setup_logging_util(False)
    assert lvl == 0
    assert log_file.exists()
    assert log_file.read_text() == ""

# ----------------------
# URL file parsing
# ----------------------

def test_parse_urls_ascii(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("huggingface.co/google-bert/bert-base-uncased\n")
    urls = url_file_cmd.parse_urls_from_file(str(url_file))
    assert urls[0].startswith("https://")

def test_parse_urls_missing_file():
    with pytest.raises(SystemExit):
        url_file_cmd.parse_urls_from_file("nope.txt")

def test_parse_urls_non_ascii(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("☃", encoding="utf-16")
    with pytest.raises(SystemExit):
        url_file_cmd.parse_urls_from_file(str(url_file))

# ----------------------
# Prep eval context
# ----------------------

def test_prepare_eval_context_hf(monkeypatch):
    # monkeypatch scrape_hf_url to fake profile
    from src.api import prep_eval_context
    monkeypatch.setattr(prep_eval_context, "scrape_hf_url",
                        lambda u: ({"readme_text": "hello", "github_links": []}, "model"))
    ctx = prepare_eval_context("https://huggingface.co/foo/bar")
    assert ctx.category == "MODEL"
    assert isinstance(ctx.hf_data, list)

def test_prepare_eval_context_github(monkeypatch):
    from src.api import prep_eval_context
    monkeypatch.setattr(prep_eval_context, "scrape_github_url",
                        lambda u: {"repo_id": "org/repo"})
    ctx = prepare_eval_context("https://github.com/org/repo")
    assert ctx.category == "CODE"

def test_prepare_eval_context_bad(caplog):
    caplog.set_level("DEBUG")
    ctx = prepare_eval_context("https://google.com")
    assert ctx.category is None
    assert "unsupported URL host" in caplog.text

def test_prepare_eval_context_none():
    with pytest.raises(ValueError):
        prepare_eval_context(None)

# ----------------------
# Net score
# ----------------------

def test_net_score_empty_report():
    rep = OrchestrationReport(results={}, total_latency_ms=0)
    bundle = bundle_from_report(rep, {}, clamp=True)
    assert 0.0 <= bundle.net_score <= 1.0

def test_net_score_with_size_dict():
    rep = OrchestrationReport(
        results={"size_score": MetricRun(name="size_score",
                                         value={"raspberry_pi": 0.5, "jetson_nano": 0.7},
                                         latency_ms=10)},
        total_latency_ms=10,
    )
    subs = subscores_from_results(rep.results)
    assert "size_score" in subs
    bundle = bundle_from_report(rep, {"size_score": 1.0}, clamp=True)
    assert 0.0 <= bundle.net_score <= 1.0

# ----------------------
# Metrics fallback behavior
# ----------------------

@pytest.mark.asyncio
async def test_size_metric_empty_ctx():
    from src.metrics import size
    ctx = type("Dummy", (), {"category": "MODEL", "hf_data": []})()
    scores = await size.metric(ctx)
    assert isinstance(scores, dict)

@pytest.mark.asyncio
async def test_dataset_quality_no_data():
    from src.metrics import dataset_quality
    ctx = type("Dummy", (), {"hf_data": []})()
    score = await dataset_quality.metric(ctx)
    assert score == 0.0

@pytest.mark.asyncio
async def test_performance_metric_no_keys(monkeypatch):
    from src.metrics import performance_metric
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEN_AI_STUDIO_API_KEY", raising=False)
    ctx = type("Dummy", (), {"gh_data": [{}]})()
    score = await performance_metric.metric(ctx)
    assert score == 0.0

@pytest.mark.asyncio
async def test_code_quality_metric_empty(monkeypatch):
    from src.metrics import code_quality_metric
    ctx = type("Dummy", (), {"gh_data": []})()
    score = await code_quality_metric.metric(ctx)
    assert score == 0.0

# ----------------------
# NDJSON printing
# ----------------------

def test_print_ndjson(tmp_path, capsys):
    rep = OrchestrationReport(
        results={"bus_factor": MetricRun("bus_factor", 0.5, 12)},
        total_latency_ms=12,
    )
    url = "https://huggingface.co/foo/bar"
    ctx = type("Dummy", (), {"category": "MODEL"})()
    url_file_cmd.print_ndjson([url], {url: ctx}, {url: rep})
    captured = capsys.readouterr().out
    out_json = json.loads(captured.strip())
    assert out_json["name"] == "bar"
    assert "net_score" in out_json