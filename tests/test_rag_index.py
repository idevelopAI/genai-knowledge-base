from pathlib import Path

import pytest
from langchain_core.documents import Document

import rag_index


def test_digest_detects_same_size_edits_even_with_unchanged_stat(tmp_path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"original")
    original_stat = pdf.stat()
    before = rag_index.pdf_digest(pdf)
    pdf.write_bytes(b"modified")
    original_method = Path.stat
    monkeypatch.setattr(
        Path,
        "stat",
        lambda path, **kwargs: (
            original_stat if path == pdf else original_method(path, **kwargs)
        ),
    )
    assert rag_index.pdf_digest(pdf) != before


def configure_index_stubs(monkeypatch, from_documents):
    monkeypatch.setattr(
        rag_index,
        "load_pdf_documents",
        lambda pdf_files: [
            Document(
                page_content="Test content",
                metadata={"source": str(pdf_files[0]), "page": 0},
            )
        ],
    )
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
    assert captured["collection_configuration"] == {"hnsw": {"space": "cosine"}}
    assert captured["collection_metadata"] == {
        **rag_index.index_settings("test-model", 1000, 200),
        "rag_corpus": rag_index.corpus_fingerprint(
            rag_index.list_pdf_files(str(data_dir))
        ),
        "rag_pdf_count": 1,
    }
    assert (
        rag_index.get_active_collection_name(str(vector_dir), "fallback")
        == stats.collection_name
    )
    assert list(vector_dir.glob(".active_collection.*.tmp")) == []


def test_direct_pdf_loader_preserves_source_and_page_metadata():
    project_root = Path(__file__).resolve().parents[1]
    pdf_path = project_root / "data" / "sample_document.pdf"

    documents = rag_index.load_pdf_documents([pdf_path])

    assert documents
    assert documents[0].metadata == {
        "source": str(pdf_path),
        "source_name": "sample_document.pdf",
        "page": 0,
        "source_sha256": rag_index.pdf_digest(pdf_path),
    }
    assert "Retrieval-Augmented Generation" in documents[0].page_content


def test_list_pdf_files_is_case_insensitive_and_excludes_symlinks(
    tmp_path,
):
    lowercase = tmp_path / "a.pdf"
    uppercase = tmp_path / "b.PDF"
    lowercase.write_bytes(b"pdf")
    uppercase.write_bytes(b"pdf")
    (tmp_path / "notes.txt").write_text("text")
    (tmp_path / "linked.pdf").symlink_to(lowercase)

    files = rag_index.list_pdf_files(str(tmp_path))

    assert files == [lowercase, uppercase]


def test_changed_corpus_during_build_keeps_previous_index(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pdf = data / "sample.pdf"
    pdf.write_bytes(b"original")
    vector = tmp_path / "chroma"
    rag_index.set_active_collection_name(str(vector), "previous")
    deleted = []
    configure_index_stubs(monkeypatch, lambda **kwargs: pdf.write_bytes(b"modified"))
    monkeypatch.setattr(
        rag_index, "delete_collection", lambda *args: deleted.append(args)
    )
    with pytest.raises(ValueError, match="changed during indexing"):
        rag_index.build_vector_db(
            data_dir=str(data),
            persist_directory=str(vector),
            embedding_model="test",
            google_api_key="test",
        )
    assert rag_index.get_active_collection_name(str(vector), "fallback") == "previous"
    assert len(deleted) == 1


def test_real_chroma_build_persists_health_metadata(tmp_path, monkeypatch):
    from langchain_core.embeddings import Embeddings

    class FixedEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    data = tmp_path / "data"
    data.mkdir()
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_document.pdf"
    (data / "sample.pdf").write_bytes(sample.read_bytes())
    monkeypatch.setattr(
        rag_index, "GoogleGenerativeAIEmbeddings", lambda **kwargs: FixedEmbeddings()
    )
    arguments = dict(
        data_dir=str(data),
        persist_directory=str(tmp_path / "chroma"),
        embedding_model="test-model",
        collection_name="test-index",
        chunk_size=1000,
        chunk_overlap=200,
    )
    stats = rag_index.build_vector_db(**arguments, google_api_key="test")
    health = rag_index.read_index_status(
        **arguments, loaded_collection=stats.collection_name
    )
    assert health.state == "ready"
    assert health.pdf_count == 1
    assert health.chunk_count == stats.chunks_created


def test_failed_index_build_deletes_partial_collection(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample.pdf").write_bytes(b"placeholder")
    created = []
    deleted = []

    def fail_partway(**kwargs):
        created.append(kwargs["collection_name"])
        raise RuntimeError("quota exhausted")

    configure_index_stubs(monkeypatch, fail_partway)
    monkeypatch.setattr(
        rag_index,
        "delete_collection",
        lambda persist_directory, name: deleted.append(name),
    )

    with pytest.raises(RuntimeError, match="quota exhausted"):
        rag_index.build_vector_db(
            data_dir=str(data_dir),
            persist_directory=str(tmp_path / "chroma"),
            embedding_model="test-model",
            google_api_key="test-key",
            collection_name="documents",
        )

    assert deleted == created


def test_failed_partial_cleanup_keeps_original_error(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample.pdf").write_bytes(b"placeholder")

    def fail_build(**kwargs):
        raise RuntimeError("embedding failed")

    def fail_cleanup(persist_directory, name):
        raise OSError("disk unavailable")

    configure_index_stubs(monkeypatch, fail_build)
    monkeypatch.setattr(rag_index, "delete_collection", fail_cleanup)

    with pytest.raises(RuntimeError, match="embedding failed"):
        rag_index.build_vector_db(
            data_dir=str(data_dir),
            persist_directory=str(tmp_path / "chroma"),
            embedding_model="test-model",
            google_api_key="test-key",
            collection_name="documents",
        )


def test_recovery_restores_interrupted_deletion(tmp_path):
    backup = tmp_path / f".guide.pdf.{'a' * 32}.deleting"
    backup.write_bytes(b"pdf")

    report = rag_index.recover_interrupted_changes(str(tmp_path))

    assert report.restored == ["guide.pdf"]
    assert (tmp_path / "guide.pdf").read_bytes() == b"pdf"
    assert not backup.exists()


def test_recovery_never_overwrites_existing_document(tmp_path):
    (tmp_path / "guide.pdf").write_bytes(b"current")
    backup = tmp_path / f".guide.pdf.{'b' * 32}.deleting"
    backup.write_bytes(b"old")

    report = rag_index.recover_interrupted_changes(str(tmp_path))

    assert report.restored == []
    assert report.conflicting_backups == [backup.name]
    assert (tmp_path / "guide.pdf").read_bytes() == b"current"
    assert backup.exists()


def test_recovery_removes_upload_staging_and_ignores_other_files(tmp_path):
    staging = tmp_path / ".upload-abc123"
    staging.mkdir()
    (staging / "new.pdf").write_bytes(b"pdf")
    (tmp_path / "keep.pdf").write_bytes(b"pdf")
    (tmp_path / ".notes.deleting").write_bytes(b"not a backup")

    report = rag_index.recover_interrupted_changes(str(tmp_path))

    assert report.removed_staging_dirs == 1
    assert not staging.exists()
    assert (tmp_path / "keep.pdf").exists()
    assert (tmp_path / ".notes.deleting").exists()


def test_recovery_handles_missing_data_directory(tmp_path):
    report = rag_index.recover_interrupted_changes(str(tmp_path / "missing"))

    assert report == rag_index.RecoveryReport([], [], 0)
