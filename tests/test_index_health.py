from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
import rag_index


@pytest.fixture
def index(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.pdf").write_bytes(b"original")
    vector = tmp_path / "vector"
    vector.mkdir()
    metadata = {
        **rag_index.index_settings("test-model", 1000, 200),
        "rag_corpus": rag_index.corpus_fingerprint(rag_index.list_pdf_files(str(data))),
        "rag_pdf_count": 1,
    }
    collection = SimpleNamespace(metadata=metadata, count=lambda: 3)
    monkeypatch.setattr(
        rag_index.chromadb,
        "PersistentClient",
        lambda **kwargs: SimpleNamespace(
            get_collection=lambda *args, **kwargs: collection
        ),
    )
    arguments = dict(
        data_dir=str(data),
        persist_directory=str(vector),
        collection_name="test-index",
        embedding_model="test-model",
        chunk_size=1000,
        chunk_overlap=200,
        loaded_collection="test-index",
    )
    return data, collection, arguments


def test_index_status_ready(index):
    _, _, arguments = index
    assert rag_index.read_index_status(**arguments) == rag_index.IndexHealth(
        "ready", pdf_count=1, chunk_count=3
    )


@pytest.mark.parametrize("change", ["edit", "add", "remove", "rename"])
def test_document_changes_make_index_stale(index, change):
    data, _, arguments = index
    document = data / "sample.pdf"
    if change == "edit":
        document.write_bytes(b"modified")
    elif change == "add":
        (data / "new.pdf").write_bytes(b"new")
    elif change == "remove":
        document.unlink()
    else:
        document.rename(data / "renamed.pdf")
    assert rag_index.read_index_status(**arguments).reason == "documents_changed"


@pytest.mark.parametrize(
    "setting,value",
    [("embedding_model", "another-model"), ("chunk_size", 900), ("chunk_overlap", 100)],
)
def test_index_settings_changes_require_rebuild(index, setting, value):
    _, _, arguments = index
    arguments[setting] = value
    assert rag_index.read_index_status(**arguments).reason == "settings_changed"


def test_legacy_and_empty_indexes_are_not_ready(index):
    _, collection, arguments = index
    collection.metadata = None
    assert rag_index.read_index_status(**arguments).reason == "metadata_missing"
    collection.count = lambda: 0
    assert rag_index.read_index_status(**arguments).state == "empty"


def test_missing_directory_is_not_created(index, tmp_path):
    _, _, arguments = index
    missing = tmp_path / "missing"
    arguments["persist_directory"] = str(missing)
    assert rag_index.read_index_status(**arguments).state == "empty"
    assert not missing.exists()


def test_external_index_activation_requires_reload(index):
    _, _, arguments = index
    arguments["loaded_collection"] = "old-index"
    assert rag_index.read_index_status(**arguments).state == "needs_reload"


@pytest.mark.parametrize(
    "state", ["empty", "needs_rebuild", "needs_reload", "building", "unavailable"]
)
def test_unready_index_blocks_cached_answers_and_model_calls(monkeypatch, state):
    main.rate_limit_events.clear()
    cache_get, retrieve, generate = Mock(), Mock(), Mock()
    monkeypatch.setattr(main, "retriever", SimpleNamespace(retrieve=retrieve))
    monkeypatch.setattr(
        main, "generation_provider", SimpleNamespace(name="fake", generate=generate)
    )
    monkeypatch.setattr(main, "_index_health", lambda: rag_index.IndexHealth(state))
    monkeypatch.setattr(main.answer_cache, "get", cache_get)
    monkeypatch.setattr(main, "RAG_API_TOKEN", None)
    client = TestClient(main.app)
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_configured"] is True
    assert health["rag_ready"] is False
    assert health["index_state"] == state
    assert client.post("/ask", json={"question": "Test?"}).status_code == 503
    cache_get.assert_not_called()
    retrieve.assert_not_called()
    generate.assert_not_called()


def test_health_read_failure_does_not_leak_paths(monkeypatch):
    monkeypatch.setattr(main, "retriever", object())
    monkeypatch.setattr(main, "generation_provider", SimpleNamespace(name="fake"))
    monkeypatch.setattr(
        main, "read_index_status", Mock(side_effect=RuntimeError("private/path"))
    )
    response = TestClient(main.app).get("/health")
    assert response.json()["index_state"] == "unavailable"
    assert "private/path" not in response.text


def test_health_detects_active_rebuild_without_reading_index(monkeypatch):
    monkeypatch.setattr(main, "retriever", object())
    monkeypatch.setattr(main, "generation_provider", object())
    read = Mock()
    monkeypatch.setattr(main, "read_index_status", read)
    with main.index_rebuild_lock:
        assert main._index_health().state == "building"
    read.assert_not_called()
