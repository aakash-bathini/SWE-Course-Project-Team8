import asyncio
import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import app  # noqa: E402


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def test_search_models_permission_and_input_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "USE_S3", False, raising=False)
    monkeypatch.setattr(app, "USE_SQLITE", False, raising=False)
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    with pytest.raises(app.HTTPException) as exc:
        _run(app.search_models(query="x", user={"permissions": []}))
    assert exc.value.status_code == 401

    with pytest.raises(app.HTTPException) as exc2:
        _run(app.search_models(query="   ", user={"permissions": ["search"]}))
    assert exc2.value.status_code == 400

    with pytest.raises(app.HTTPException) as exc3:
        _run(app.search_models(query="(", user={"permissions": ["search"]}))
    assert exc3.value.status_code == 400


def test_search_models_s3_and_in_memory_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeS3:
        def list_artifacts_by_queries(self, _queries):
            return [
                {
                    "metadata": {"id": "m1", "name": "Bert-A", "type": "model"},
                    "data": {"hf_data": [{"readme_text": "no"}]},
                },
                {
                    "metadata": {"id": "m2", "name": "Other", "type": "model"},
                    "data": {"hf_data": [{"readme_text": "this mentions bert"}]},
                },
                # Non-model should be skipped
                {"metadata": {"id": "d1", "name": "ds", "type": "dataset"}, "data": {}},
            ]

    monkeypatch.setattr(app, "USE_S3", True, raising=False)
    monkeypatch.setattr(app, "USE_SQLITE", False, raising=False)
    monkeypatch.setattr(app, "s3_storage", FakeS3(), raising=False)

    # In-memory artifact that should also match
    monkeypatch.setattr(
        app,
        "artifacts_db",
        {
            "mem1": {"metadata": {"name": "bert-mem", "type": "model"}, "data": {}},
        },
        raising=False,
    )

    res = _run(app.search_models(query="bert", user={"permissions": ["search"]}))
    assert res["count"] >= 3
    ids = [r["id"] for r in res["results"]]
    assert "m1" in ids
    assert "m2" in ids
    assert "mem1" in ids


def test_search_models_by_version_s3_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeS3:
        def list_artifacts_by_queries(self, _queries):
            return [
                {
                    "metadata": {"id": "m1", "name": "Model", "type": "model"},
                    "data": {"hf_data": [{"siblings": [{"rfilename": "v1.0.0"}]}]},
                },
                {
                    "metadata": {"id": "m2", "name": "Model2", "type": "model"},
                    "data": {"hf_data": [{"siblings": [{"rfilename": "v2.0.0"}]}]},
                },
            ]

    monkeypatch.setattr(app, "USE_S3", True, raising=False)
    monkeypatch.setattr(app, "USE_SQLITE", False, raising=False)
    monkeypatch.setattr(app, "s3_storage", FakeS3(), raising=False)
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    res = _run(app.search_models_by_version(query="~1.0.0", user={"permissions": ["search"]}))
    assert res["count"] == 1
    assert res["results"][0]["id"] == "m1"
    assert res["results"][0]["version"] == "1.0.0"
