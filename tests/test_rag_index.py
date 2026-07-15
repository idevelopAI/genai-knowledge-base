import pytest
from langchain_core.documents import Document

import rag_index


class StubPdfLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load(self):
        return [
            Document(
                page_content="Test content",
                metadata={"source": f"{self.data_dir}/sample.pdf", "page": 0},
            )
        ]


def configure_index_stubs(monkeypatch, from_documents):
    monkeypatch.setattr(rag_index, "PyPDFDirectoryLoader", StubPdfLoader)
    monkeypatch.setattr(
        rag_index,
        "GoogleGenerativeAIEmbeddings",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        rag_index.Chroma,
        "from_documents",
        staticmethod(from_documents),
    )


def test_failed_index_build_keeps_previous_collection(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    vector_dir = tmp_path / "chroma"
    data_dir.mkdir()
    (data_dir / "sample.pdf").write_bytes(b"placeholder")
    rag_index.set_active_collection_name(str(vector_dir), "existing-collection")

    def fail_build(**kwargs):
        raise RuntimeError("embedding failed")

    configure_index_stubs(monkeypatch, fail_build)

    with pytest.raises(RuntimeError, match="embedding failed"):
        rag_index.build_vector_db(
            data_dir=str(data_dir),
            persist_directory=str(vector_dir),
            embedding_model="test-model",
            google_api_key="test-key",
            collection_name="documents",
        )

    assert (
        rag_index.get_active_collection_name(str(vector_dir), "fallback")
        == "existing-collection"
    )


def test_successful_index_build_activates_versioned_collection(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    vector_dir = tmp_path / "chroma"
    data_dir.mkdir()
    (data_dir / "sample.pdf").write_bytes(b"placeholder")
    captured = {}

    def record_build(**kwargs):
        captured.update(kwargs)
        return object()

    configure_index_stubs(monkeypatch, record_build)

    stats = rag_index.build_vector_db(
        data_dir=str(data_dir),
        persist_directory=str(vector_dir),
        embedding_model="test-model",
        google_api_key="test-key",
        collection_name="documents",
    )

    assert stats.collection_name.startswith("documents-")
    assert captured["collection_name"] == stats.collection_name
    assert (
        rag_index.get_active_collection_name(str(vector_dir), "fallback")
        == stats.collection_name
    )
    assert list(vector_dir.glob(".active_collection.*.tmp")) == []
