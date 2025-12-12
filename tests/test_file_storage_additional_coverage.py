import os
import sys
import zipfile

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.storage import file_storage  # noqa: E402


def test_get_artifact_directory_rejects_path_traversal(tmp_path) -> None:
    # Redirect storage root to a temp dir
    root = str(tmp_path / "uploads")
    os.makedirs(root, exist_ok=True)
    file_storage.STORAGE_ROOT = root

    with pytest.raises(ValueError):
        file_storage.get_artifact_directory("../evil")


def test_save_uploaded_file_and_checksum_and_delete(tmp_path) -> None:
    root = str(tmp_path / "uploads")
    os.makedirs(root, exist_ok=True)
    file_storage.STORAGE_ROOT = root

    meta = file_storage.save_uploaded_file("a1", b"hello", "x.bin")
    assert meta["filename"] == "x.bin"
    assert meta["size"] == 5
    assert os.path.exists(meta["path"])
    assert isinstance(meta["checksum"], str) and len(meta["checksum"]) == 64

    assert file_storage.delete_artifact_files("a1") is True
    assert file_storage.delete_artifact_files("a1") is False


def test_extract_zip_and_bad_zip(tmp_path) -> None:
    base = tmp_path / "z"
    base.mkdir()
    zip_path = base / "t.zip"
    out_dir = base / "out"
    out_dir.mkdir()

    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("README.md", "hi")
        z.writestr("sub/file.txt", "ok")

    extracted = file_storage.extract_zip(str(zip_path), str(out_dir))
    assert "README.md" in extracted
    assert "sub/file.txt" in extracted

    bad_zip = base / "bad.zip"
    bad_zip.write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        file_storage.extract_zip(str(bad_zip), str(out_dir))


def test_find_and_read_model_card(tmp_path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "README.md").write_text("root readme", encoding="utf-8")

    card_path = file_storage.find_model_card(str(root))
    assert card_path is not None and card_path.endswith("README.md")
    assert file_storage.read_model_card(card_path) == "root readme"

    # Subdirectory scan path (one level deep)
    (root / "README.md").unlink()
    sub = root / "subdir"
    sub.mkdir()
    (sub / "README.md").write_text("sub readme", encoding="utf-8")
    card_path2 = file_storage.find_model_card(str(root))
    assert card_path2 is not None and card_path2.endswith("subdir/README.md")
    assert file_storage.read_model_card(card_path2) == "sub readme"

    assert file_storage.read_model_card(str(root / "MISSING.md")) == ""


def test_filter_files_and_zip_creation_and_file_info(tmp_path) -> None:
    root = tmp_path / "artifact2"
    root.mkdir()
    (root / "a.py").write_text("print('x')", encoding="utf-8")
    (root / "b.ipynb").write_text("{}", encoding="utf-8")
    (root / "c.csv").write_text("a,b", encoding="utf-8")
    (root / "w.bin").write_bytes(b"123")

    all_files = file_storage.filter_files_by_aspect(str(root), "full")
    assert len(all_files) >= 4

    code_files = file_storage.filter_files_by_aspect(str(root), "code")
    assert any(p.endswith("a.py") for p in code_files)
    assert any(p.endswith("b.ipynb") for p in code_files)

    dataset_files = file_storage.filter_files_by_aspect(str(root), "datasets")
    assert any(p.endswith("c.csv") for p in dataset_files)

    weights_files = file_storage.filter_files_by_aspect(str(root), "weights")
    assert any(p.endswith("w.bin") for p in weights_files)

    out_zip = root / "out.zip"
    created = file_storage.create_zip_from_files(code_files, str(root), str(out_zip))
    assert created.endswith("out.zip")
    assert os.path.exists(created)

    info = file_storage.get_file_info(str(root / "a.py"))
    assert info["filename"] == "a.py"
    assert info["size"] > 0
    assert isinstance(info["checksum"], str) and len(info["checksum"]) == 64

    with pytest.raises(Exception):
        file_storage.get_file_info(str(root / "nope"))


def test_file_storage_error_branches(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = str(tmp_path / "uploads")
    os.makedirs(root, exist_ok=True)
    file_storage.STORAGE_ROOT = root

    # save_uploaded_file exception path
    def boom_ensure() -> None:
        raise RuntimeError("no storage")

    monkeypatch.setattr(file_storage, "ensure_storage_directory", boom_ensure)
    with pytest.raises(RuntimeError):
        file_storage.save_uploaded_file("a2", b"x", "f.bin")

    # extract_zip corrupted-zip path via testzip() returning a filename
    class FakeZip:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def testzip(self):
            return "badfile"

        def extractall(self, _to):
            return None

        def namelist(self):
            return ["a"]

    monkeypatch.setattr(file_storage.zipfile, "ZipFile", FakeZip)
    with pytest.raises(ValueError):
        file_storage.extract_zip("x.zip", str(tmp_path / "out"))

    # find_model_card exception path
    monkeypatch.setattr(file_storage.os, "listdir", lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert file_storage.find_model_card(str(tmp_path)) is None

    # filter_files_by_aspect exception path
    monkeypatch.setattr(file_storage.os, "walk", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert file_storage.filter_files_by_aspect(str(tmp_path), "full") == []

    # create_zip_from_files exception path
    monkeypatch.setattr(file_storage.zipfile, "ZipFile", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        file_storage.create_zip_from_files(["x"], str(tmp_path), str(tmp_path / "o.zip"))

    # delete_artifact_files exception path
    monkeypatch.setattr(file_storage, "get_artifact_directory", lambda _aid: str(tmp_path / "nope"))
    monkeypatch.setattr(file_storage.shutil, "rmtree", lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert file_storage.delete_artifact_files("a3") is False
