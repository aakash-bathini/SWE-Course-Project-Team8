# src/tests/test_all.py

import os
import sys
import io
import json
import types
import pytest
import requests
import subprocess
from pathlib import Path

import src.metrics.size as model_size
from src.api.prep_eval_context import prepare_eval_context
from src.commands import url_file_cmd
from src.models.types import OrchestrationReport, MetricRun
from src.scoring.net_score import bundle_from_report, subscores_from_results
from src.orchestration import logging_util
import src.api.huggingface as hf
import src.api.github as gh

# -------------------------------------------------------------------
# LOGGING_UTIL.PY
# -------------------------------------------------------------------

def test_logfile_missing_env(monkeypatch, tmp_path):
    log_file = tmp_path / "doesnotexist.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
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

# -------------------------------------------------------------------
# URL_FILE_CMD.PY
# -------------------------------------------------------------------

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

def test_normalize_url_variants():
    from src.commands.url_file_cmd import normalize_url
    u1 = "https://huggingface.co/org/model/tree/main"
    u2 = "https://github.com/org/repo/blob/main/file.py"
    assert normalize_url(u1) == "https://huggingface.co/org/model"
    assert normalize_url(u2) == "https://github.com/org/repo"

def test_normalize_url_passthrough():
    from src.commands.url_file_cmd import normalize_url
    url = "https://example.com/foo"
    assert normalize_url(url) == url

def test_read_lines_ascii_multiple(tmp_path):
    from src.commands.url_file_cmd import _read_lines_ascii
    f = tmp_path / "urls.txt"
    f.write_text("hf.co/a/b, github.com/c/d  \n")
    urls = list(_read_lines_ascii(f))
    assert all(u.startswith("https://") for u in urls)

def test_read_lines_ascii_blank_lines(tmp_path):
    from src.commands.url_file_cmd import _read_lines_ascii
    f = tmp_path / "urls.txt"
    f.write_text("\n\n")
    urls = list(_read_lines_ascii(f))
    assert urls == []

def test_print_ndjson(capsys):
    rep = OrchestrationReport(
        results={"bus_factor": MetricRun("bus_factor", 0.5, 12)},
        total_latency_ms=12,
    )
    url = "https://huggingface.co/foo/bar"
    ctx = types.SimpleNamespace(category="MODEL")
    url_file_cmd.print_ndjson([url], {url: ctx}, {url: rep})
    captured = capsys.readouterr().out
    out_json = json.loads(captured.strip())
    assert out_json["name"] == "bar"
    assert "net_score" in out_json

# -------------------------------------------------------------------
# PREP_EVAL_CONTEXT.PY
# -------------------------------------------------------------------

def test_prepare_eval_context_hf(monkeypatch):
    from src.api import prep_eval_context
    monkeypatch.setattr(prep_eval_context, "scrape_hf_url",
                        lambda u: ({"readme_text": "hello", "github_links": []}, "model"))
    ctx = prepare_eval_context("https://huggingface.co/foo/bar")
    assert ctx.category == "MODEL"

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

# -------------------------------------------------------------------
# NET_SCORE.PY
# -------------------------------------------------------------------

def test_net_score_empty_report():
    rep = OrchestrationReport(results={}, total_latency_ms=0)
    bundle = bundle_from_report(rep, {}, clamp=True)
    assert 0.0 <= bundle.net_score <= 1.0

def test_net_score_with_size_dict():
    rep = OrchestrationReport(
        results={"size_score": MetricRun("size_score",
                                         value={"raspberry_pi": 0.5, "jetson_nano": 0.7},
                                         latency_ms=10)},
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
async def test_size_metric_empty_ctx():
    from src.metrics import size
    ctx = types.SimpleNamespace(category="MODEL", hf_data=[])
    scores = await size.metric(ctx)
    assert isinstance(scores, dict)

@pytest.mark.asyncio
async def test_size_metric_with_readme_regex():
    from src.metrics import size
    ctx = types.SimpleNamespace(
        category="MODEL",
        hf_data=[{"readme_text": "Requires 16GB VRAM", "card_yaml": {}, "files": []}],
        gh_data=[]
    )
    scores = await size.metric(ctx)
    assert any(v <= 1.0 for v in scores.values())

def test_sum_repo_size_from_index():
    from src.metrics.size import _sum_repo_size_from_index
    files = [{"type": "blob", "size": 123}, {"type": "tree"}]
    assert _sum_repo_size_from_index(files) == 123

# -------------------------------------------------------------------
# METRICS: DATASET_QUALITY
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dataset_quality_no_data():
    from src.metrics import dataset_quality
    ctx = types.SimpleNamespace(hf_data=[])
    score = await dataset_quality.metric(ctx)
    assert score == 0.0

@pytest.mark.asyncio
async def test_dataset_quality_with_scores():
    from src.metrics import dataset_quality
    ctx = types.SimpleNamespace(hf_data=[{
        "downloads": 5000,
        "likes": 100,
        "readme_text": "dataset for testing"
    }])
    score = await dataset_quality.metric(ctx)
    assert 0.0 <= score <= 1.0

# -------------------------------------------------------------------
# METRICS: PERFORMANCE
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_performance_metric_no_keys(monkeypatch):
    from src.metrics import performance_metric
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEN_AI_STUDIO_API_KEY", raising=False)
    ctx = types.SimpleNamespace(gh_data=[{}])
    score = await performance_metric.metric(ctx)
    assert score == 0.0

@pytest.mark.asyncio
async def test_performance_metric_retry(monkeypatch):
    import src.metrics.performance_metric as pm
    ctx = types.SimpleNamespace(gh_data=[{"readme_text": "This is a model README"}])
    calls = {"n": 0}
    class FakeResp:
        def json(self):
            return {"choices": [{"message": {"content": '{"summary":{"overall_evidence_quality":1,"overall_specificity":1}}'}}]}
    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("fail")
        return FakeResp()
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "dummy")
    score = await pm.metric(ctx)
    assert 0.0 <= score <= 1.0

# -------------------------------------------------------------------
# METRICS: CODE_QUALITY
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_quality_metric_empty():
    from src.metrics import code_quality_metric
    ctx = types.SimpleNamespace(gh_data=[])
    score = await code_quality_metric.metric(ctx)
    assert score == 0.0

@pytest.mark.asyncio
async def test_code_quality_subprocess_fail(monkeypatch):
    import src.metrics.code_quality_metric as cqm
    async def fake_run_cmd(*a, **k): return "", "err", 1
    monkeypatch.setattr(cqm, "run_cmd", fake_run_cmd)
    score = await cqm.compute_linting_score(".")
    assert 0.0 <= score <= 1.0

@pytest.mark.asyncio
async def test_code_quality_tests_scoring(tmp_path):
    import src.metrics.code_quality_metric as cqm
    repo = tmp_path
    (repo / "tests").mkdir()
    (repo / "tests" / "test_example.py").write_text("def test_x(): assert True")
    ctx = types.SimpleNamespace(gh_data=[{"local_repo_path": str(repo)}])
    score = await cqm.metric(ctx)
    assert 0.0 <= score <= 1.0

# -------------------------------------------------------------------
# METRICS: LICENSE_CHECK
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_license_check_metric():
    import src.metrics.license_check as lc
    ctx = types.SimpleNamespace(hf_data=[{"license": "mit"}], gh_data=[])
    score = await lc.metric(ctx)
    assert 0.0 <= score <= 1.0

# -------------------------------------------------------------------
# METRICS: BUS_FACTOR
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bus_factor_metric():
    import src.metrics.bus_factor_metric as bf
    ctx = types.SimpleNamespace(gh_data=[{"contributors": {"a": 10, "b": 5}}], hf_data=[])
    score = await bf.metric(ctx)
    assert 0.0 <= score <= 1.0

# -------------------------------------------------------------------
# API: GITHUB
# -------------------------------------------------------------------

def test_github_rate_limit(monkeypatch):
    import src.api.github as gh
    class FakeResp:
        status_code = 403
        text = "rate limit"
        def raise_for_status(self): raise requests.exceptions.HTTPError("403")
    monkeypatch.setattr(gh.requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(requests.exceptions.HTTPError):
        gh._get_json("https://api.github.com/repos/foo/bar")

def test_github_get_json_success(monkeypatch):
    import src.api.github as gh
    class FakeResp:
        def json(self): return {"ok": True}
        def raise_for_status(self): return None
    monkeypatch.setattr(gh.requests, "get", lambda *a, **k: FakeResp())
    out = gh._get_json("https://api.github.com/repos/foo/bar")
    assert out == {"ok": True}

def test_github_cache_roundtrip(tmp_path, monkeypatch):
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

def test_parse_hf_url_variants():
    from src.api.huggingface import parse_hf_url
    assert parse_hf_url("https://huggingface.co/models/org/name")[0] == "model"
    assert parse_hf_url("https://huggingface.co/datasets/org/data")[0] == "dataset"
    assert parse_hf_url("https://huggingface.co/org/name")[0] == "model"
    with pytest.raises(ValueError):
        parse_hf_url("https://not-hf.com/foo")

def test_extract_github_links():
    from src.api.huggingface import _extract_github_links
    readme = "Check https://github.com/org/repo"
    card_yaml = {"repository": "https://github.com/org/other"}
    links = _extract_github_links(readme, card_yaml)
    assert any("github.com" in l for l in links)

def test_is_fresh_and_cache_helpers(tmp_path, monkeypatch):
    from src.api import huggingface as hf
    entry = {"fetched_at": hf._now()}
    assert hf._is_fresh(entry)
    entry["fetched_at"] = hf._now() - 999999
    assert not hf._is_fresh(entry)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    (tmp_path / ".cache").mkdir(parents=True, exist_ok=True)
    hf._save_cache({"x": {"payload": {"foo": "bar"}, "fetched_at": hf._now()}})
    cache = hf._load_cache()
    assert "x" in cache

# -------------------------------------------------------------------
# CONFIG_PARSERS_NLP: SPDX
# -------------------------------------------------------------------

def test_spdx_classify_and_normalize():
    from src.config_parsers_nlp import spdx
    spdx.LICENSE_ALIASES["apache"] = "Apache-2.0"
    assert spdx.normalize_license("apache license") == "Apache-2.0"
    spdx.LICENSE_WHITELIST.add("mit")
    score, _ = spdx.classify_license("mit")
    assert score == 1.0
    spdx.LICENSE_BLACKLIST.add("proprietary")
    score, _ = spdx.classify_license("proprietary")
    assert score == 0.0
    spdx.LICENSE_AMBIGUOUS_03.add("free")
    s, _ = spdx.classify_license("free")
    assert s == 0.3

# -------------------------------------------------------------------
# CONFIG_PARSERS_NLP: README_PARSER
# -------------------------------------------------------------------
def test_extract_license_block():
    from src.config_parsers_nlp.readme_parser import extract_license_block
    text = "Some text\n## License\nThis is licensed under MIT.\nMore text"
    block = extract_license_block(text)
    assert "MIT" in block

# -------------------------------------------------------------------
# REGISTRY.PY
# -------------------------------------------------------------------

def test_registry_returns_expected_names():
    import src.metrics.registry as reg
    items = reg.get_all_metrics()
    names = [name for name, _ in items]
    assert "ramp_up_time" in names
    assert "bus_factor" in names
    assert "performance_claims" in names
    assert "license" in names
    assert "size_score" in names
    assert "dataset_and_code_score" in names
    assert "dataset_quality" in names
    assert "code_quality" in names


def test_registry_metric_functions_are_callable():
    import src.metrics.registry as reg
    items = reg.get_all_metrics()
    for name, fn in items:
        assert callable(fn)


def test_registry_metric_wiring(monkeypatch):
    import src.metrics.registry as reg
    monkeypatch.setattr(reg.size, "metric", lambda ctx=None: "SIZE")
    monkeypatch.setattr(reg.license_check, "metric", lambda ctx=None: "LICENSE")
    monkeypatch.setattr(reg.performance_metric, "metric", lambda ctx=None: "PERF")
    monkeypatch.setattr(reg.code_quality_metric, "metric", lambda ctx=None: "CODEQ")
    monkeypatch.setattr(reg.bus_factor_metric, "metric", lambda ctx=None: "BUS")
    monkeypatch.setattr(reg.dataset_quality, "metric", lambda ctx=None: "DATAQ")
    monkeypatch.setattr(reg.ramp_up_time, "metric", lambda ctx=None: "RAMP")
    monkeypatch.setattr(reg.available_dataset_code, "metric", lambda ctx=None: "AVAIL")

    items = dict(reg.get_all_metrics())
    assert items["size_score"]() == "SIZE"
    assert items["license"]() == "LICENSE"
    assert items["performance_claims"]() == "PERF"
    assert items["code_quality"]() == "CODEQ"
    assert items["bus_factor"]() == "BUS"
    assert items["dataset_quality"]() == "DATAQ"
    assert items["ramp_up_time"]() == "RAMP"
    assert items["dataset_and_code_score"]() == "AVAIL"

    # -------------------------------------------------------------------
# WEIGHTS.PY
# -------------------------------------------------------------------

def test_weights_are_normalized_and_nonzero():
    from src.scoring import weights
    w = weights.get_weights()
    # Sum should equal 1.0
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # All weights should be > 0
    assert all(v > 0 for v in w.values())


# -------------------------------------------------------------------
# PREP_EVAL_ORCHESTRATOR.PY
# -------------------------------------------------------------------

import pytest, types

@pytest.mark.asyncio
async def test_prep_eval_orchestrator_success(monkeypatch):
    import src.orchestration.prep_eval_orchestrator as peo

    async def fake_prepare(url):
        return types.SimpleNamespace(url=url, category="MODEL")

    monkeypatch.setattr(peo, "prepare_eval_context", fake_prepare)
    urls = ["https://huggingface.co/org/model"]
    out = await peo.prep_eval_many(urls, limit=2)
    assert urls[0] in out
    assert out[urls[0]].url == urls[0]


        # -------------------------------------------------------------------
# API: GITHUB – deeper coverage
# -------------------------------------------------------------------

def test_should_fetch_variants():
    from src.api import github as gh
    assert gh._should_fetch("README.md")
    assert gh._should_fetch("docs/file.rst")
    assert not gh._should_fetch("bigfile.bin")

def test_fetch_bitesized_file_base64(monkeypatch):
    from src.api import github as gh
    content = __import__("base64").b64encode(b"hello").decode()
    monkeypatch.setattr(
        gh, "_get_json",
        lambda *a, **k: {"type": "file", "encoding": "base64", "content": content}
    )
    out = gh._fetch_bitesized_file("o", "r", "f", "main", 10)
    assert "hello" in out

def test_fetch_bitesized_file_bad(monkeypatch):
    from src.api import github as gh
    # bad base64 → should return falsy
    monkeypatch.setattr(
        gh, "_get_json",
        lambda *a, **k: {"type": "file", "encoding": "base64", "content": "@@@"}
    )
    out = gh._fetch_bitesized_file("o", "r", "f", "main", 10)
    assert not out  # covers both None and ""

# -------------------------------------------------------------------
# API: HUGGINGFACE – deeper coverage
# -------------------------------------------------------------------

def test_scrape_hf_url_cached(monkeypatch, tmp_path):
    from src.api import huggingface as hf
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    (tmp_path / ".cache").mkdir(parents=True, exist_ok=True)
    data = {"url":"x"}
    cache = {"model:foo/bar": {"payload": data, "fetched_at": hf._now()}}
    hf._save_cache(cache)
    out, kind = hf.scrape_hf_url("https://huggingface.co/foo/bar")
    assert out["url"] == "x"
    assert kind == "model"

def test_extract_github_links_nested():
    from src.api import huggingface as hf
    readme = "Text with https://github.com/org/repo."
    card_yaml = {"resources": {"x": "https://github.com/org/other"}}
    links = hf._extract_github_links(readme, card_yaml)
    assert any("github.com" in l for l in links)

# -------------------------------------------------------------------
# METRICS: SIZE – deeper coverage
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_size_metric_mem_requirements(monkeypatch):
    import src.metrics.size as size
    # README mentions GB → should parse memory requirement
    ctx = types.SimpleNamespace(
        category="MODEL",
        hf_data=[{"readme_text": "Needs 24GB GPU", "card_yaml": {}, "files": []}],
        gh_data=[]
    )
    scores = await size.metric(ctx)
    assert any(0.0 <= v <= 1.0 for v in scores.values())

@pytest.mark.asyncio
async def test_size_metric_small_large_values(monkeypatch):
    import src.metrics.size as size
    # Small requirement (MB) and huge requirement (TB) to hit edges
    ctx1 = types.SimpleNamespace(
        category="MODEL",
        hf_data=[{"readme_text": "Requires 512MB memory", "card_yaml": {}, "files": []}],
        gh_data=[]
    )
    scores1 = await size.metric(ctx1)
    assert all(0.0 <= v <= 1.0 for v in scores1.values())

    ctx2 = types.SimpleNamespace(
        category="MODEL",
        hf_data=[{"readme_text": "Needs 1000TB GPU", "card_yaml": {}, "files": []}],
        gh_data=[]
    )
    scores2 = await size.metric(ctx2)
    assert all(0.0 <= v <= 1.0 for v in scores2.values())

import types, pytest
import src.metrics.size as size

def test_to_bytes_and_human():
    assert size._to_bytes(1, "GB") == 1024**3
    assert size._to_bytes(1, "GiB") == 1024**3
    assert size._to_bytes(1, "MB") == 1024**2
    assert size._to_bytes(1, "TB") == 1024**4
    assert size._to_bytes(5, "foo") == 5
    assert size._bytes_to_human(1023) == "1023.0B"
    assert size._bytes_to_human(1024) == "1.0KB"
    assert size._bytes_to_human(5*1024**3) == "5.0GB"

def test_flatten_card_yaml_and_sum_size():
    data = {"a": {"b": [1,2,{"c":3}]}}
    flat = size._flatten_card_yaml(data)
    assert "a" in flat and "b" in flat and "3" in flat
    files = [{"type":"blob","size":100},{"type":"tree"},{"type":"blob","size":50}]
    assert size._sum_repo_size_from_index(files) == 150

def test_hf_total_size_bytes_prefers_size():
    hf = {"size": 12345, "files":[{"size":99}]}
    assert size._hf_total_size_bytes(hf) == 12345
    hf2 = {"files":[{"size":10},{"size":20}]}
    assert size._hf_total_size_bytes(hf2) == 30

@pytest.mark.asyncio
async def test_metric_dataset_disk_and_model_fallback():
    ctx = types.SimpleNamespace(
        category="DATASET",
        hf_data=[{"readme_text":"dataset size 50GB","card_yaml":{},"files":[]}],
        gh_data=[]
    )
    scores = await size.metric(ctx)
    assert isinstance(scores, dict)

    ctx2 = types.SimpleNamespace(
        category="MODEL",
        hf_data=[{"readme_text":"","card_yaml":{},"files":[],"size":1024}],
        gh_data=[]
    )
    scores2 = await size.metric(ctx2)
    assert isinstance(scores2, dict)

@pytest.mark.asyncio
async def test_metric_code_paths_and_unknown():
    ctx = types.SimpleNamespace(
        category="CODE",
        hf_data=[],
        gh_data=[{"readme_text":"Requires 2GB RAM","doc_texts":{},"files_index":[{"type":"blob","size":100}]}]
    )
    scores = await size.metric(ctx)
    assert isinstance(scores, dict)

    ctx2 = types.SimpleNamespace(category="OTHER", hf_data=[], gh_data=[])
    scores2 = await size.metric(ctx2)
    assert isinstance(scores2, dict)

# -------------------------------------------------------------------
# METRICS: LICENSE_CHECK – deeper coverage
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_license_check_fallbacks():
    import src.metrics.license_check as lc
    # HF None → fallback to GH
    ctx1 = types.SimpleNamespace(hf_data=[{"license": None}], gh_data=[{"license_spdx":"MIT"}])
    score1 = await lc.metric(ctx1)
    assert 0.0 <= score1 <= 1.0

    # Both missing → should return 0.0
    ctx2 = types.SimpleNamespace(hf_data=[{"license": None}], gh_data=[{}])
    score2 = await lc.metric(ctx2)
    assert score2 == 0.0

# -------------------------------------------------------------------
# METRICS: PERFORMANCE – deeper coverage
# -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_performance_metric_gemini(monkeypatch):
    import src.metrics.performance_metric as pm
    ctx = types.SimpleNamespace(gh_data=[{"readme_text": "test"}])
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class R:
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": '{"summary":{"overall_evidence_quality":1,"overall_specificity":1}}'
                    }
                }]
            }

    # must patch *inside the module*, not requests globally
    monkeypatch.setattr(pm, "requests", types.SimpleNamespace(post=lambda *a, **k: R()))

    score = await pm.metric(ctx)
    # assert score == 1.0

@pytest.mark.asyncio
async def test_performance_metric_bad_json(monkeypatch):
    import src.metrics.performance_metric as pm
    ctx = types.SimpleNamespace(gh_data=[{"readme_text": "test"}])
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "dummy")
    class R:
        def json(self): return {"choices":[{"message":{"content": "not-json"}}]}
    monkeypatch.setattr(pm.requests, "post", lambda *a, **k: R())
    score = await pm.metric(ctx)
    assert score == 0.0

@pytest.mark.asyncio
async def test_performance_metric_always_fail(monkeypatch):
    import src.metrics.performance_metric as pm
    ctx = types.SimpleNamespace(gh_data=[{"readme_text": "test"}])
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "dummy")
    monkeypatch.setattr(pm.requests, "post", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))
    score = await pm.metric(ctx)
    assert score == 0.0

def test_parse_github_url_valid_and_invalid():
    assert gh.parse_github_url("https://github.com/user/repo") == ("user","repo")
    with pytest.raises(ValueError):
        gh.parse_github_url("https://example.com/notgithub")
    with pytest.raises(ValueError):
        gh.parse_github_url("https://github.com/user")

def test_should_fetch_patterns():
    assert gh._should_fetch("README.md")
    assert not gh._should_fetch("binary.bin")

def test_is_fresh_and_cache(tmp_path, monkeypatch):
    entry = {"fetched_at": gh._now()}
    assert gh._is_fresh(entry)
    monkeypatch.setattr(gh, "_CACHE_TTL_S", -1)
    assert not gh._is_fresh(entry)

def test_parse_hf_url_valid_and_invalid():
    assert hf.parse_hf_url("https://huggingface.co/user/model") == ("model","user/model")
    assert hf.parse_hf_url("https://huggingface.co/datasets/user/data") == ("dataset","user/data")
    with pytest.raises(ValueError):
        hf.parse_hf_url("https://example.com/not-hf")
    with pytest.raises(ValueError):
        hf.parse_hf_url("https://huggingface.co/")

def test_extract_github_links_from_card_and_readme():
    links = hf._extract_github_links("see https://github.com/u/r", {"repository": "https://github.com/u/r2"})
    assert any("github.com" in l for l in links)

def test_parse_hf_url_valid_and_invalid():
    assert hf.parse_hf_url("https://huggingface.co/user/model") == ("model","user/model")
    assert hf.parse_hf_url("https://huggingface.co/datasets/user/data") == ("dataset","user/data")
    with pytest.raises(ValueError):
        hf.parse_hf_url("https://example.com/not-hf")
    with pytest.raises(ValueError):
        hf.parse_hf_url("https://huggingface.co/")

def test_extract_github_links_from_card_and_readme():
    links = hf._extract_github_links("see https://github.com/u/r", {"repository": "https://github.com/u/r2"})
    assert any("github.com" in l for l in links)

import src.api.prep_eval_context as prep


def test_prepare_eval_context_github(monkeypatch):
    monkeypatch.setattr("src.api.prep_eval_context.scrape_github_url", lambda u: {"repo_id": "user/repo"})
    ctx = prep.prepare_eval_context("https://github.com/user/repo")
    assert ctx.category == "CODE"
    assert ctx.gh_data

def test_prepare_eval_context_hf_with_github(monkeypatch):
    fake_hf = {"github_links": ["https://github.com/u/r"], "repo_type": "model"}
    monkeypatch.setattr("src.api.prep_eval_context.scrape_hf_url", lambda u: (fake_hf, "model"))
    monkeypatch.setattr("src.api.prep_eval_context.scrape_github_url", lambda u: {"repo_id": "u/r"})
    ctx = prep.prepare_eval_context("https://huggingface.co/u/m")
    assert ctx.category == "MODEL"
    assert ctx.gh_data

from pathlib import Path
import pytest
import src.commands.url_file_cmd as urlcmd

def test_normalize_url_variants():
    assert "huggingface.co/org/name" in urlcmd.normalize_url("https://huggingface.co/org/name/tree/main")
    assert "github.com/org/repo" in urlcmd.normalize_url("https://github.com/org/repo/blob/main/file.py")

def test_read_lines_ascii_and_parse_urls(tmp_path):
    f = tmp_path/"urls.txt"
    f.write_text("huggingface.co/u/m github.com/u/r")
    urls = list(urlcmd._read_lines_ascii(f))
    assert all(u.startswith("https://") for u in urls)

def test_parse_urls_file_not_found(tmp_path):
    with pytest.raises(SystemExit):
        urlcmd.parse_urls_from_file(str(tmp_path/"nope.txt"))

import src.config_parsers_nlp.readme_parser as rp

def test_strip_markdown_noise_removes_code_and_comments():
    md = "```python\nprint(123)\n```\n<!--comment-->\n[id]: http://link\n`inline`"
    out = rp._strip_markdown_noise(md)
    assert "```" not in out
    assert "<!--" not in out
    assert "`inline`" not in out

def test_extract_section_found_and_not_found():
    md = "# License\nMIT\n# Other\n"
    assert "MIT" in rp.extract_section(md, rp.LICENSE_HX)
    assert rp.extract_section("no headings", rp.LICENSE_HX) is None

def test_find_spdx_ids_exprs_hints():
    text = "SPDX-License-Identifier: MIT\nApache-2.0 OR GPL-3.0-only\nThis project uses the MIT license"
    ids = rp.find_spdx_ids(text)
    exprs = rp.find_spdx_expressions(text)
    hints = rp.find_license_hints(text)
    assert "MIT" in ids
    assert any("OR" in e for e in exprs)
    assert any("mit" in h for h in hints)

import src.config_parsers_nlp.spdx as spdx

def test_normalize_and_classify_license():
    assert spdx.normalize_license("MIT License") == "MIT"
    score, msg = spdx.classify_license("MIT")
    assert score == 1.0
    assert spdx.classify_license("GPL-3.0-only")[0] == 0.0
    assert spdx.classify_license("gpl")[0] == 0.3

import src.orchestration.logging_util as logutil

def test_setup_logging_util_with_env(tmp_path, monkeypatch):
    log_file = tmp_path/"log.txt"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "2")
    lvl = logutil.setup_logging_util(also_stderr=True)
    assert lvl == 2

import pytest
import src.orchestration.metric_orchestrator as mo
from src.models.types import EvalContext


@pytest.mark.asyncio
async def test_run_one_success_and_failure():
    async def good(ctx): return 0.5
    async def bad(ctx): raise RuntimeError("fail")
    ctx = EvalContext(url="u")
    r1 = await mo._run_one(("good", good), ctx)
    assert r1.value == 0.5
    r2 = await mo._run_one(("bad", bad), ctx)
    assert r2.error