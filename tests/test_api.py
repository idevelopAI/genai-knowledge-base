import os

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from rag_index import sanitize_filename, save_pdf_bytes


os.environ.pop("GOOGLE_API_KEY", None)

import main  # noqa: E402


client = TestClient(main.app)


def setup_function():
    main.rate_limit_events.clear()


def test_health_is_available_without_api_key():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["rag_ready"] is False
    assert "pdf_count" in payload


def test_health_returns_generated_request_id():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_request_id_header_is_preserved():
    response = client.get("/health", headers={"X-Request-ID": "demo-request-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "demo-request-123"


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
        files={"file": ("sample.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


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


def test_sanitize_filename_removes_path_and_unsafe_characters():
    assert sanitize_filename("../../My File (Final).pdf") == "My_File_Final_.pdf"


def test_save_pdf_bytes_avoids_overwriting_existing_files(tmp_path):
    first = save_pdf_bytes(b"one", "report.pdf", str(tmp_path))
    second = save_pdf_bytes(b"two", "report.pdf", str(tmp_path))

    assert first.name == "report.pdf"
    assert second.name == "report-1.pdf"
    assert first.read_bytes() == b"one"
    assert second.read_bytes() == b"two"
