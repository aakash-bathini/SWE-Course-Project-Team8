import asyncio
import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import app  # noqa: E402


def _run(coro):
    # Keep a stable event loop for the whole test process.
    # Some existing tests still rely on asyncio.get_event_loop() semantics.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def test_artifact_by_regex_permission_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    with pytest.raises(app.HTTPException) as exc:
        _run(app.artifact_by_regex(app.ArtifactRegEx(regex="x"), user={"permissions": []}))

    assert exc.value.status_code == 401


def test_artifact_by_regex_rejects_too_long_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    long_pat = "a" * 600
    with pytest.raises(app.HTTPException) as exc:
        _run(app.artifact_by_regex(app.ArtifactRegEx(regex=long_pat), user={"permissions": ["search"]}))

    assert exc.value.status_code == 400


def test_artifact_by_regex_invalid_and_dangerous_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    with pytest.raises(app.HTTPException) as exc1:
        _run(app.artifact_by_regex(app.ArtifactRegEx(regex="("), user={"permissions": ["search"]}))
    assert exc1.value.status_code == 400

    with pytest.raises(app.HTTPException) as exc2:
        _run(app.artifact_by_regex(app.ArtifactRegEx(regex="(a+)+$"), user={"permissions": ["search"]}))
    assert exc2.value.status_code == 400


def test_artifact_by_regex_in_memory_exact_and_partial_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep search local (avoid S3/SQLite branches)
    monkeypatch.setattr(app, "USE_S3", False, raising=False)
    monkeypatch.setattr(app, "USE_SQLITE", False, raising=False)

    # Put the match early so truncation to 10000 chars doesn't remove it.
    big_readme = "BERT " + ("x" * 12050)
    artifacts = {
        "id1": {
            "metadata": {"id": "id1", "name": "ExactName", "type": "model"},
            "data": {"readme_text": "Some README"},
        },
        "id2": {
            "metadata": {"id": "id2", "name": "Other", "type": "model"},
            "data": {"readme_text": big_readme},
        },
    }
    monkeypatch.setattr(app, "artifacts_db", artifacts, raising=False)

    # Exact match should be case-sensitive
    res_exact = _run(app.artifact_by_regex(app.ArtifactRegEx(regex="^ExactName$"), user={"permissions": ["search"]}))
    assert any(getattr(r, "id", None) == "id1" for r in res_exact)

    # Partial match is case-insensitive and also checks README text (including truncation path)
    res_partial = _run(app.artifact_by_regex(app.ArtifactRegEx(regex="bert"), user={"permissions": ["search"]}))
    assert any(getattr(r, "id", None) == "id2" for r in res_partial)


def test_artifact_by_regex_s3_path_match(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the large S3 scan block (app.py ~3444-3580)
    monkeypatch.setattr(app, "USE_S3", True, raising=False)
    monkeypatch.setattr(app, "USE_SQLITE", False, raising=False)
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    class FakeS3:
        def list_artifacts_by_queries(self, _queries):
            return [
                {
                    "metadata": {"id": "s3-1", "name": "NoMatch", "type": "model"},
                    "data": {"readme_text": "nothing here"},
                },
                {
                    "metadata": {"id": "s3-2", "name": "SomeModel", "type": "model"},
                    "data": {"readme_text": "This README mentions BERT."},
                },
            ]

        def get_artifact_metadata(self, _artifact_id: str):
            return None

    monkeypatch.setattr(app, "s3_storage", FakeS3(), raising=False)

    res = _run(app.artifact_by_regex(app.ArtifactRegEx(regex="bert"), user={"permissions": ["search"]}))
    assert any(getattr(r, "id", None) == "s3-2" for r in res)


def test_artifact_by_regex_s3_error_then_404(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cover S3 exception handling path.
    monkeypatch.setattr(app, "USE_S3", True, raising=False)
    monkeypatch.setattr(app, "USE_SQLITE", False, raising=False)
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    class FakeS3Boom:
        def list_artifacts_by_queries(self, _queries):
            raise RuntimeError("boom")

        def get_artifact_metadata(self, _artifact_id: str):
            return None

    monkeypatch.setattr(app, "s3_storage", FakeS3Boom(), raising=False)

    with pytest.raises(app.HTTPException) as exc:
        _run(app.artifact_by_regex(app.ArtifactRegEx(regex="bert"), user={"permissions": ["search"]}))

    assert exc.value.status_code == 404


def test_artifact_by_regex_sqlite_partial_and_search_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the large SQLite scan block + search-history tracking block.
    monkeypatch.setattr(app, "USE_S3", False, raising=False)
    monkeypatch.setattr(app, "USE_SQLITE", True, raising=False)
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    class FakeArtifact:
        def __init__(self, id_: str, name: str, type_: str, url: str):
            self.id = id_
            self.name = name
            self.type = type_
            self.url = url

    class FakeSession:
        def __init__(self):
            self.added = []
            self.committed = False

        def add(self, obj):
            self.added.append(obj)

        def commit(self):
            self.committed = True

        def query(self, _model):
            raise AssertionError("query() should not be called for partial match path")

    class Ctx:
        def __init__(self, session: FakeSession):
            self.session = session

        def __enter__(self):
            return self.session

        def __exit__(self, *_args):
            return False

    fake_session = FakeSession()

    def fake_get_db():
        # Called twice inside artifact_by_regex (SQLite scan + search tracking)
        return iter([Ctx(fake_session)])

    monkeypatch.setattr(app, "get_db", fake_get_db, raising=False)

    def fake_list_by_regex(_db, _pattern):
        return [
            FakeArtifact("sql-1", "Other", "model", "https://example.com/a"),
            FakeArtifact("sql-2", "BertThing", "model", "https://example.com/b"),
        ]

    monkeypatch.setattr(app.db_crud, "list_by_regex", fake_list_by_regex, raising=True)

    # Force README fallback to produce a match for sql-2
    def fake_ensure(_artifact_id: str, _data_block):
        return "BERT appears here."

    monkeypatch.setattr(app, "_ensure_regex_readme_text", fake_ensure, raising=True)

    # Patch SearchHistory to avoid SQLAlchemy model requirements.
    import src.db.models as db_models

    class DummySearchHistory:
        def __init__(self, artifact_id: str, search_type: str):
            self.artifact_id = artifact_id
            self.search_type = search_type

    monkeypatch.setattr(db_models, "SearchHistory", DummySearchHistory, raising=True)

    res = _run(app.artifact_by_regex(app.ArtifactRegEx(regex="bert"), user={"permissions": ["search"]}))
    assert any(getattr(r, "id", None) == "sql-2" for r in res)
    assert fake_session.committed is True


def test_artifact_by_regex_sqlite_exact_match_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise SQLite exact-match path (name_only=True), including query().all() branch.
    monkeypatch.setattr(app, "USE_S3", False, raising=False)
    monkeypatch.setattr(app, "USE_SQLITE", True, raising=False)
    monkeypatch.setattr(app, "artifacts_db", {}, raising=False)

    class FakeArtifact:
        def __init__(self, id_: str, name: str, type_: str, url: str):
            self.id = id_
            self.name = name
            self.type = type_
            self.url = url

    class QueryResult:
        def __init__(self, items):
            self._items = items

        def all(self):
            return self._items

    class FakeSession:
        def __init__(self):
            self.added = []
            self.committed = False

        def query(self, _model):
            return QueryResult(
                [
                    FakeArtifact("sqlx-1", "ExactName", "model", "https://example.com/x"),
                    FakeArtifact("sqlx-2", "Other", "model", "https://example.com/y"),
                ]
            )

        def add(self, obj):
            self.added.append(obj)

        def commit(self):
            self.committed = True

    class Ctx:
        def __init__(self, session: FakeSession):
            self.session = session

        def __enter__(self):
            return self.session

        def __exit__(self, *_args):
            return False

    fake_session = FakeSession()

    def fake_get_db():
        return iter([Ctx(fake_session)])

    monkeypatch.setattr(app, "get_db", fake_get_db, raising=False)

    # Name-only path should NOT call list_by_regex.
    monkeypatch.setattr(
        app.db_crud,
        "list_by_regex",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("list_by_regex should not be called")),
        raising=True,
    )

    # Provide README so exact-match path also tests README search
    monkeypatch.setattr(app, "_ensure_regex_readme_text", lambda *_a, **_k: "ExactName", raising=True)

    import src.db.models as db_models

    class DummySearchHistory:
        def __init__(self, artifact_id: str, search_type: str):
            self.artifact_id = artifact_id
            self.search_type = search_type

    monkeypatch.setattr(db_models, "SearchHistory", DummySearchHistory, raising=True)

    res = _run(app.artifact_by_regex(app.ArtifactRegEx(regex="^ExactName$"), user={"permissions": ["search"]}))
    assert any(getattr(r, "id", None) == "sqlx-1" for r in res)
    assert fake_session.committed is True
