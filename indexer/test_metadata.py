"""
Tests for the metadata API endpoints and storage layer added in feat/metadata-api.

Run from indexer/:
    python -m pytest test_metadata.py -v
"""
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import create_engine, SQLModel, Session

import storage as storage_module
from storage import MinimaDoc, MinimaStore, MinimaDocUpdate
from singleton import Singleton
from app import router, _resolve_file_path, _find_file_by_filename, _serialize_doc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_in_memory_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_keys={"check_same_thread": False},
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def reset_store(tmp_path):
    """Replace the storage engine with an in-memory DB and reset the Singleton."""
    engine = make_in_memory_engine()
    with patch.object(storage_module, "engine", engine):
        Singleton._instances.pop(MinimaStore, None)
        yield engine
        Singleton._instances.pop(MinimaStore, None)


@pytest.fixture
def files_dir(tmp_path):
    """Temporary directory used as FILES_PATH."""
    return tmp_path


@pytest.fixture
def set_files_path(files_dir):
    with patch.dict(os.environ, {"CONTAINER_PATH": str(files_dir)}):
        with patch("app.FILES_PATH", str(files_dir)):
            yield files_dir


@pytest.fixture
def client():
    """TestClient using just the router, no lifespan."""
    test_app = FastAPI()
    test_app.include_router(router)
    return TestClient(test_app, raise_server_exceptions=True)


def write_txt(directory: Path, name: str, content: str = "hello") -> Path:
    p = directory / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# MinimaStore storage layer tests
# ---------------------------------------------------------------------------

class TestMinimaStoreUpsertMetadata:
    def test_creates_new_doc(self, reset_store):
        engine = reset_store
        doc = MinimaStore.upsert_metadata("/docs/a.pdf", "my desc", '["tag1"]', 100)
        assert doc.fpath == "/docs/a.pdf"
        assert doc.description == "my desc"
        assert doc.tags == '["tag1"]'
        assert doc.last_updated_seconds == 100
        assert doc.metadata_updated_seconds is not None

    def test_updates_existing_doc(self, reset_store):
        MinimaStore.upsert_metadata("/docs/a.pdf", "original", "[]", 100)
        doc = MinimaStore.upsert_metadata("/docs/a.pdf", "updated", '["x"]', 200)
        assert doc.description == "updated"
        assert doc.tags == '["x"]'
        assert doc.last_updated_seconds == 200

    def test_metadata_updated_seconds_set_on_create(self, reset_store):
        doc = MinimaStore.upsert_metadata("/docs/b.pdf", "", "[]", 100)
        assert doc.metadata_updated_seconds is not None
        assert doc.metadata_updated_seconds >= doc.last_updated_seconds

    def test_metadata_updated_seconds_advances_on_update(self, reset_store):
        doc1 = MinimaStore.upsert_metadata("/docs/c.pdf", "v1", "[]", 100)
        doc2 = MinimaStore.upsert_metadata("/docs/c.pdf", "v2", "[]", 100)
        assert doc2.metadata_updated_seconds >= doc1.metadata_updated_seconds


class TestMinimaStoreFindByFilename:
    def test_finds_match(self, reset_store):
        MinimaStore.upsert_metadata("/docs/sub/report.pdf", "desc", "[]", 1)
        results = MinimaStore.find_by_filename("report.pdf")
        assert "/docs/sub/report.pdf" in results

    def test_returns_empty_for_unknown(self, reset_store):
        assert MinimaStore.find_by_filename("ghost.pdf") == []

    def test_does_not_match_partial(self, reset_store):
        MinimaStore.upsert_metadata("/docs/annual_report.pdf", "", "[]", 1)
        assert MinimaStore.find_by_filename("report.pdf") == []


class TestMinimaStoreListDocs:
    def test_returns_all_docs_ordered(self, reset_store):
        MinimaStore.upsert_metadata("/b.pdf", "", "[]", 1)
        MinimaStore.upsert_metadata("/a.pdf", "", "[]", 1)
        docs = MinimaStore.list_docs()
        paths = [d.fpath for d in docs]
        assert paths == sorted(paths)

    def test_empty_when_no_docs(self, reset_store):
        assert MinimaStore.list_docs() == []


class TestSerializeDoc:
    def test_serializes_tags_from_json(self):
        doc = MinimaDoc(fpath="/a.pdf", description="d", tags='["x","y"]', last_updated_seconds=1)
        result = _serialize_doc(doc)
        assert result["tags"] == ["x", "y"]
        assert result["description"] == "d"

    def test_handles_malformed_tags(self):
        doc = MinimaDoc(fpath="/a.pdf", description="", tags="not-json", last_updated_seconds=1)
        result = _serialize_doc(doc)
        assert result["tags"] == []

    def test_handles_none_doc(self):
        result = _serialize_doc(None, fpath="/a.pdf")
        assert result["description"] == ""
        assert result["tags"] == []
        assert result["path"] == "/a.pdf"


# ---------------------------------------------------------------------------
# Path helper tests
# ---------------------------------------------------------------------------

class TestResolveFilePath:
    def test_valid_absolute_path(self, set_files_path):
        f = write_txt(set_files_path, "doc.txt")
        with patch("app._supported_file", return_value=True):
            resolved = _resolve_file_path(request_path=str(f))
        assert resolved == str(f.resolve())

    def test_valid_relative_path(self, set_files_path):
        f = write_txt(set_files_path, "doc.txt")
        with patch("app._supported_file", return_value=True):
            resolved = _resolve_file_path(request_path="doc.txt")
        assert resolved == str(f.resolve())

    def test_path_traversal_blocked(self, set_files_path):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _resolve_file_path(request_path="../../etc/passwd")
        assert exc_info.value.status_code == 400
        assert "escape" in exc_info.value.detail

    def test_missing_file_returns_404(self, set_files_path):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            with patch("app._supported_file", return_value=True):
                _resolve_file_path(request_path="missing.txt")
        assert exc_info.value.status_code == 404

    def test_unsupported_file_type_returns_400(self, set_files_path):
        from fastapi import HTTPException
        write_txt(set_files_path, "doc.xyz")
        with patch("app._supported_file", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_file_path(request_path="doc.xyz")
        assert exc_info.value.status_code == 400

    def test_no_path_or_filename_returns_400(self, set_files_path):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _resolve_file_path()
        assert exc_info.value.status_code == 400


class TestFindFileByFilename:
    def test_finds_file_on_filesystem(self, set_files_path):
        write_txt(set_files_path, "report.txt")
        with patch("app._supported_file", return_value=True):
            result = _find_file_by_filename("report.txt")
        assert result.endswith("report.txt")

    def test_raises_404_when_not_found(self, set_files_path):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _find_file_by_filename("ghost.txt")
        assert exc_info.value.status_code == 404

    def test_raises_409_on_ambiguous_match(self, set_files_path):
        from fastapi import HTTPException
        sub = set_files_path / "sub"
        sub.mkdir()
        write_txt(set_files_path, "dup.txt")
        write_txt(sub, "dup.txt")
        with patch("app._supported_file", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                _find_file_by_filename("dup.txt")
        assert exc_info.value.status_code == 409

    def test_rejects_filename_with_path_separator(self, set_files_path):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _find_file_by_filename("sub/report.txt")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

class TestGetMetadataEndpoint:
    def test_returns_metadata_for_known_file(self, client, set_files_path, reset_store):
        f = write_txt(set_files_path, "a.txt")
        MinimaStore.upsert_metadata(str(f), "my desc", '["t1"]', 1)
        with patch("app._supported_file", return_value=True):
            resp = client.get(f"/metadata?path={f}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["description"] == "my desc"
        assert data["tags"] == ["t1"]

    def test_returns_empty_metadata_for_untracked_file(self, client, set_files_path, reset_store):
        f = write_txt(set_files_path, "b.txt")
        with patch("app._supported_file", return_value=True):
            resp = client.get(f"/metadata?path={f}")
        assert resp.status_code == 200
        assert resp.json()["description"] == ""
        assert resp.json()["tags"] == []

    def test_returns_400_on_path_traversal(self, client, set_files_path):
        resp = client.get("/metadata?path=../../etc/passwd")
        assert resp.status_code == 400

    def test_returns_404_for_missing_file(self, client, set_files_path):
        with patch("app._supported_file", return_value=True):
            resp = client.get("/metadata?path=missing.txt")
        assert resp.status_code == 404


class TestGetMetadataByFilenameEndpoint:
    def test_returns_metadata_by_filename(self, client, set_files_path, reset_store):
        f = write_txt(set_files_path, "named.txt")
        MinimaStore.upsert_metadata(str(f), "desc", '["tag"]', 1)
        with patch("app._supported_file", return_value=True):
            resp = client.get("/metadata/by-filename?filename=named.txt")
        assert resp.status_code == 200
        assert resp.json()["description"] == "desc"

    def test_returns_409_on_ambiguous_filename(self, client, set_files_path):
        sub = set_files_path / "sub"
        sub.mkdir()
        write_txt(set_files_path, "dup.txt")
        write_txt(sub, "dup.txt")
        with patch("app._supported_file", return_value=True):
            resp = client.get("/metadata/by-filename?filename=dup.txt")
        assert resp.status_code == 409

    def test_returns_404_for_unknown_filename(self, client, set_files_path):
        resp = client.get("/metadata/by-filename?filename=ghost.txt")
        assert resp.status_code == 404


class TestPutMetadataEndpoint:
    def test_creates_metadata(self, client, set_files_path, reset_store):
        f = write_txt(set_files_path, "c.txt")
        with patch("app._supported_file", return_value=True):
            with patch("app.async_queue") as mock_q:
                resp = client.put("/metadata", json={
                    "path": str(f),
                    "description": "created",
                    "tags": ["a", "b"],
                })
        assert resp.status_code == 200
        data = resp.json()
        assert data["description"] == "created"
        assert data["tags"] == ["a", "b"]
        mock_q.enqueue.assert_called_once()

    def test_updates_existing_metadata(self, client, set_files_path, reset_store):
        f = write_txt(set_files_path, "d.txt")
        MinimaStore.upsert_metadata(str(f), "old", "[]", 1)
        with patch("app._supported_file", return_value=True):
            with patch("app.async_queue"):
                resp = client.put("/metadata", json={
                    "path": str(f),
                    "description": "new",
                    "tags": ["x"],
                })
        assert resp.json()["description"] == "new"

    def test_enqueues_reindex(self, client, set_files_path, reset_store):
        f = write_txt(set_files_path, "e.txt")
        with patch("app._supported_file", return_value=True):
            with patch("app.async_queue") as mock_q:
                client.put("/metadata", json={"path": str(f), "description": "", "tags": []})
        mock_q.enqueue.assert_called_once()
        queued = mock_q.enqueue.call_args[0][0]
        assert queued["path"] == str(f.resolve())
        assert queued["source"] == "metadata_update"

    def test_strips_empty_tags(self, client, set_files_path, reset_store):
        f = write_txt(set_files_path, "f.txt")
        with patch("app._supported_file", return_value=True):
            with patch("app.async_queue"):
                resp = client.put("/metadata", json={
                    "path": str(f),
                    "description": "",
                    "tags": ["", "  ", "valid"],
                })
        assert resp.json()["tags"] == ["valid"]

    def test_returns_400_on_path_traversal(self, client, set_files_path):
        resp = client.put("/metadata", json={"path": "../../etc/passwd", "description": "", "tags": []})
        assert resp.status_code == 400


class TestListMetadataEndpoint:
    def test_returns_supported_files(self, client, set_files_path, reset_store):
        write_txt(set_files_path, "a.txt")
        write_txt(set_files_path, "b.txt")
        with patch("app._supported_file", return_value=True):
            resp = client.get("/metadata/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "files" in data
        names = [f["name"] for f in data["files"]]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_excludes_unsupported_files(self, client, set_files_path, reset_store):
        write_txt(set_files_path, "good.txt")
        write_txt(set_files_path, "bad.xyz")

        def mock_supported(path):
            return path.endswith(".txt")

        with patch("app._supported_file", side_effect=mock_supported):
            resp = client.get("/metadata/list")
        names = [f["name"] for f in resp.json()["files"]]
        assert "good.txt" in names
        assert "bad.xyz" not in names

    def test_includes_metadata_for_tracked_files(self, client, set_files_path, reset_store):
        f = write_txt(set_files_path, "tracked.txt")
        MinimaStore.upsert_metadata(str(f), "my desc", '["t1"]', 1)
        with patch("app._supported_file", return_value=True):
            resp = client.get("/metadata/list")
        files = {item["name"]: item for item in resp.json()["files"]}
        assert files["tracked.txt"]["description"] == "my desc"
        assert files["tracked.txt"]["tags"] == ["t1"]

    def test_returns_empty_metadata_for_untracked_files(self, client, set_files_path, reset_store):
        write_txt(set_files_path, "untracked.txt")
        with patch("app._supported_file", return_value=True):
            resp = client.get("/metadata/list")
        files = {item["name"]: item for item in resp.json()["files"]}
        assert files["untracked.txt"]["description"] == ""
        assert files["untracked.txt"]["tags"] == []

    def test_files_sorted_by_relative_path(self, client, set_files_path, reset_store):
        sub = set_files_path / "sub"
        sub.mkdir()
        write_txt(set_files_path, "z.txt")
        write_txt(sub, "a.txt")
        with patch("app._supported_file", return_value=True):
            resp = client.get("/metadata/list")
        rel_paths = [f["relative_path"] for f in resp.json()["files"]]
        assert rel_paths == sorted(rel_paths)

    def test_includes_relative_path_and_name(self, client, set_files_path, reset_store):
        sub = set_files_path / "docs"
        sub.mkdir()
        write_txt(sub, "report.txt")
        with patch("app._supported_file", return_value=True):
            resp = client.get("/metadata/list")
        files = {f["name"]: f for f in resp.json()["files"]}
        assert files["report.txt"]["relative_path"] == "docs/report.txt"
        assert files["report.txt"]["name"] == "report.txt"
