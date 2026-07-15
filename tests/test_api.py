import os
import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from index_jobs import IndexJobStore
from langchain_core.documents import Document
from pypdf import PdfWriter
from rag_index import (
    IndexStats,
    get_active_collection_name,
    sanitize_filename,
    save_pdf_bytes,
    set_active_collection_name,
)


os.environ.pop("GOOGLE_API_KEY", None)

import main  # noqa: E402


client = TestClient(main.app)


def make_pdf_bytes(page_count: int = 1) -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    writer.write(output)
    return output.getvalue()


def setup_function():
    main.rate_limit_events.clear()
    main.index_job_store.clear()


def wait_for_terminal_job(job_id: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = main.index_job_store.get(job_id)
        if job and job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not finish in time")


def test_health_is_available_without_api_key():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["rag_ready"] is False
    assert payload["min_relevance_score"] == main.RAG_MIN_RELEVANCE_SCORE
    assert "pdf_count" in payload


def test_health_returns_generated_request_id():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_request_id_header_is_preserved():
    response = client.get("/health", headers={"X-Request-ID": "demo-request-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "demo-request-123"


def test_invalid_request_id_is_replaced():
    response = client.get("/health", headers={"X-Request-ID": "invalid\nvalue"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "invalid\nvalue"


def test_ask_returns_503_when_rag_is_not_configured(monkeypatch):
    monkeypatch.setattr(main, "retriever", None)
    monkeypatch.setattr(main, "llm", None)

    response = client.post("/ask", json={"question": "What is in the document?"})

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_ask_requires_bearer_token_when_configured(monkeypatch):
    monkeypatch.setattr(main, "RAG_API_TOKEN", "test-token")

    response = client.post("/ask", json={"question": "What is in the document?"})

    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()


def test_ask_accepts_bearer_token_when_configured(monkeypatch):
    monkeypatch.setattr(main, "RAG_API_TOKEN", "test-token")
    monkeypatch.setattr(main, "retriever", None)
    monkeypatch.setattr(main, "llm", None)

    response = client.post(
        "/ask",
        json={"question": "What is in the document?"},
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_ask_rejects_blank_question():
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_rejects_question_over_length_limit():
    response = client.post(
        "/ask",
        json={"question": "x" * (main.MAX_QUESTION_LENGTH + 1)},
    )

    assert response.status_code == 422


def test_ask_returns_without_llm_call_when_retrieval_has_no_context(monkeypatch):
    class EmptyRetriever:
        def retrieve(self, question):
            return main.RetrievalResult(
                documents=[],
                candidate_count=1,
                best_candidate_score=0.12,
            )

    monkeypatch.setattr(main, "retriever", EmptyRetriever())
    monkeypatch.setattr(main, "llm", object())

    response = client.post("/ask", json={"question": "Unknown topic"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": main.NO_CONTEXT_ANSWER,
        "sources": [],
    }


def test_ask_maps_transient_provider_failure_to_503(monkeypatch):
    class ProviderUnavailable(Exception):
        status_code = 503

    class FailingRetriever:
        def retrieve(self, question):
            raise ProviderUnavailable("temporary upstream failure")

    monkeypatch.setattr(main, "retriever", FailingRetriever())
    monkeypatch.setattr(main, "llm", object())

    response = client.post("/ask", json={"question": "Known topic"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json()["detail"] == (
        "The model provider is temporarily unavailable."
    )


def test_relevance_filter_keeps_only_scored_usable_documents():
    relevant = Document(
        page_content="Relevant context",
        metadata={"source": "guide.pdf", "page": 0},
    )
    irrelevant = Document(
        page_content="Irrelevant context",
        metadata={"source": "other.pdf", "page": 0},
    )

    class ScoredVectorStore:
        def similarity_search_with_relevance_scores(self, question, k):
            assert question == "test question"
            assert k == 3
            return [
                (relevant, 0.81),
                (irrelevant, 0.44),
                (Document(page_content="  ", metadata={}), 0.9),
            ]

    retriever = main.RelevanceFilteringRetriever(
        ScoredVectorStore(),
        top_k=3,
        min_score=0.45,
    )

    result = retriever.retrieve("test question")

    assert result.documents == [relevant]
    assert result.candidate_count == 3
    assert result.best_candidate_score == 0.9


def test_relevance_filter_rejects_non_finite_scores():
    document = Document(page_content="Context", metadata={})

    class InvalidScoreVectorStore:
        def similarity_search_with_relevance_scores(self, question, k):
            return [(document, float("nan")), (document, 1.2)]

    retriever = main.RelevanceFilteringRetriever(
        InvalidScoreVectorStore(),
        top_k=1,
        min_score=0,
    )

    result = retriever.retrieve("test")

    assert result.documents == []
    assert result.best_candidate_score is None


def test_list_documents_returns_pdf_names(tmp_path, monkeypatch):
    (tmp_path / "sample.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "notes.txt").write_text("not a pdf")
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": ["sample.pdf"]}


def test_list_documents_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(main, "RAG_API_TOKEN", "test-token")

    response = client.get("/documents")

    assert response.status_code == 401


def test_rate_limit_blocks_expensive_endpoints(monkeypatch):
    monkeypatch.setattr(main, "RAG_API_TOKEN", None)
    monkeypatch.setattr(main, "RAG_RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(main, "RAG_RATE_LIMIT_WINDOW_SECONDS", 60)
    main.rate_limit_events.clear()

    first_response = client.get("/documents")
    second_response = client.get("/documents")

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert "Retry-After" in second_response.headers


def test_rate_limit_can_be_disabled(monkeypatch):
    monkeypatch.setattr(main, "RAG_API_TOKEN", None)
    monkeypatch.setattr(main, "RAG_RATE_LIMIT_REQUESTS", 0)
    main.rate_limit_events.clear()

    first_response = client.get("/documents")
    second_response = client.get("/documents")

    assert first_response.status_code == 200
    assert second_response.status_code == 200


def test_rate_limit_does_not_store_or_log_bearer_token(monkeypatch, caplog):
    token = "private-test-token"
    monkeypatch.setattr(main, "RAG_API_TOKEN", token)
    monkeypatch.setattr(main, "RAG_RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(main, "RAG_RATE_LIMIT_WINDOW_SECONDS", 60)
    main.rate_limit_events.clear()
    headers = {"Authorization": f"Bearer {token}"}

    first_response = client.get("/documents", headers=headers)
    second_response = client.get("/documents", headers=headers)

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert token not in " ".join(main.rate_limit_events)
    assert token not in caplog.text


def test_upload_rejects_non_pdf_files():
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_pdf_requires_configured_key(monkeypatch):
    monkeypatch.setattr(main, "GOOGLE_API_KEY", None)

    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", make_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_upload_rejects_invalid_pdf_contents():
    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", b"not-a-pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "signature" in response.json()["detail"]


def test_upload_rejects_files_over_size_limit(monkeypatch):
    monkeypatch.setattr(main, "RAG_MAX_UPLOAD_BYTES", 8)

    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", make_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 413


def test_upload_rejects_pdf_over_page_limit(monkeypatch):
    monkeypatch.setattr(main, "RAG_MAX_PDF_PAGES", 1)

    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", make_pdf_bytes(2), "application/pdf")},
    )

    assert response.status_code == 400
    assert "page limit" in response.json()["detail"]


def test_upload_removes_saved_file_when_indexing_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "GOOGLE_API_KEY", "configured-for-test")
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))

    def fail_rebuild(**kwargs):
        raise main.HTTPException(status_code=500, detail="index failed")

    monkeypatch.setattr(main, "_rebuild_vector_store", fail_rebuild)

    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", make_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 202
    job = wait_for_terminal_job(response.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"] == "index failed"
    assert list(tmp_path.glob("*.pdf")) == []


def test_rebuild_rejects_concurrent_indexing(monkeypatch):
    monkeypatch.setattr(main, "GOOGLE_API_KEY", "configured-for-test")
    main.index_rebuild_lock.acquire()
    try:
        with pytest.raises(main.HTTPException) as exc_info:
            main._rebuild_vector_store()
    finally:
        main.index_rebuild_lock.release()

    assert exc_info.value.status_code == 409


def test_reindex_returns_job_and_status_endpoint(monkeypatch):
    monkeypatch.setattr(main, "GOOGLE_API_KEY", "configured-for-test")
    monkeypatch.setattr(
        main,
        "_rebuild_vector_store",
        lambda **kwargs: IndexStats(
            pdf_count=2,
            pages_loaded=3,
            chunks_created=4,
            data_dir="data",
            vector_store="chroma",
            collection_name="documents-test",
        ),
    )

    response = client.post("/reindex", headers={"X-Request-ID": "job-request"})

    assert response.status_code == 202
    payload = response.json()
    assert response.headers["Location"] == f"/jobs/{payload['job_id']}"
    job = wait_for_terminal_job(payload["job_id"])
    assert job["status"] == "completed"
    assert job["request_id"] == "job-request"
    assert job["result"]["chunks_created"] == 4

    status_response = client.get(response.headers["Location"])
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"


def test_unknown_job_returns_404():
    response = client.get("/jobs/missing-job")

    assert response.status_code == 404


def test_job_status_requires_token_when_configured(monkeypatch):
    job = main.index_job_store.create(
        operation="reindex",
        request_id="request-1",
    )
    monkeypatch.setattr(main, "RAG_API_TOKEN", "test-token")

    unauthorized = client.get(f"/jobs/{job['job_id']}")
    authorized = client.get(
        f"/jobs/{job['job_id']}",
        headers={"Authorization": "Bearer test-token"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_full_job_queue_rolls_back_pending_upload(tmp_path, monkeypatch):
    store = IndexJobStore(max_jobs=1)
    store.create(operation="reindex", request_id="existing-request")
    monkeypatch.setattr(main, "index_job_store", store)
    uploaded_path = tmp_path / "pending.pdf"
    uploaded_path.write_bytes(make_pdf_bytes())

    with pytest.raises(main.HTTPException) as exc_info:
        main._submit_index_job(
            operation="upload",
            request_id="upload-request",
            uploaded_path=uploaded_path,
        )

    assert exc_info.value.status_code == 503
    assert uploaded_path.exists() is False


def test_rebuild_restores_previous_collection_when_reload_fails(
    tmp_path,
    monkeypatch,
):
    vector_dir = tmp_path / "chroma"
    set_active_collection_name(str(vector_dir), "existing-collection")
    monkeypatch.setattr(main, "GOOGLE_API_KEY", "configured-for-test")
    monkeypatch.setattr(main, "CHROMA_DB_DIR", str(vector_dir))
    monkeypatch.setattr(main, "CHROMA_COLLECTION_NAME", "documents")

    def build_new_collection(**kwargs):
        set_active_collection_name(str(vector_dir), "new-collection")
        return IndexStats(
            pdf_count=1,
            pages_loaded=1,
            chunks_created=1,
            data_dir="data",
            vector_store=str(vector_dir),
            collection_name="new-collection",
        )

    monkeypatch.setattr(main, "build_vector_db", build_new_collection)

    def fail_reload():
        raise RuntimeError("reload failed")

    monkeypatch.setattr(main, "_reload_rag_components", fail_reload)

    with pytest.raises(main.HTTPException) as exc_info:
        main._rebuild_vector_store()

    assert exc_info.value.status_code == 500
    assert (
        get_active_collection_name(str(vector_dir), "fallback")
        == "existing-collection"
    )


def test_build_sources_uses_filename_and_one_based_page_number():
    docs = [
        Document(
            page_content="This is a longer source excerpt from the document.",
            metadata={"source": "data/sample_document.pdf", "page": 0},
        )
    ]

    sources = main.build_sources(docs)

    assert len(sources) == 1
    assert sources[0].source == "sample_document.pdf"
    assert sources[0].page == 1
    assert sources[0].excerpt == "This is a longer source excerpt from the document."


def test_format_docs_adds_source_labels():
    docs = [
        Document(
            page_content="Document context",
            metadata={"source": "data/sample_document.pdf", "page": 1},
        )
    ]

    formatted = main.format_docs(docs)

    assert "[Source 1: sample_document.pdf, page 2]" in formatted
    assert "Document context" in formatted


def test_prepare_retrieved_documents_removes_empty_and_exact_duplicates():
    duplicate = Document(
        page_content="Document context",
        metadata={"source": "data/sample_document.pdf", "page": 1},
    )
    unique = Document(
        page_content="Other context",
        metadata={"source": "data/sample_document.pdf", "page": 2},
    )

    prepared = main.prepare_retrieved_documents(
        [duplicate, duplicate, Document(page_content=" ", metadata={}), unique]
    )

    assert prepared == [duplicate, unique]


def test_system_prompt_treats_document_context_as_untrusted_and_requires_sources():
    assert "nicht vertrauenswuerdigen Dokumentdaten" in main.system_prompt
    assert "[Source N]" in main.system_prompt


def test_ensure_source_markers_appends_available_sources_when_missing():
    answer = main.ensure_source_markers("Grounded answer.", source_count=2)

    assert answer == "Grounded answer.\n\nQuellen: [Source 1] [Source 2]"


def test_ensure_source_markers_removes_unavailable_source_numbers():
    answer = main.ensure_source_markers(
        "First claim [Source 1]. Invalid [Source 9].",
        source_count=2,
    )

    assert "[Source 1]" in answer
    assert "[Source 9]" not in answer


def test_sanitize_filename_removes_path_and_unsafe_characters():
    assert sanitize_filename("../../My File (Final).pdf") == "My_File_Final_.pdf"


def test_save_pdf_bytes_avoids_overwriting_existing_files(tmp_path):
    first = save_pdf_bytes(b"one", "report.pdf", str(tmp_path))
    second = save_pdf_bytes(b"two", "report.pdf", str(tmp_path))

    assert first.name == "report.pdf"
    assert second.name == "report-1.pdf"
    assert first.read_bytes() == b"one"
    assert second.read_bytes() == b"two"
