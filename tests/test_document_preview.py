import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from PIL import Image, ImageStat
from pypdf import PdfWriter

import main
import pdf_preview


SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_document.pdf"


@pytest.fixture
def preview(tmp_path, monkeypatch):
    contents = SAMPLE.read_bytes()
    (tmp_path / "sample.pdf").write_bytes(contents)
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "RAG_API_TOKEN", "preview-test-token")
    main.rate_limit_events.clear()
    return (
        TestClient(main.app),
        {"Authorization": "Bearer preview-test-token"},
        contents,
    )


@pytest.mark.parametrize("suffix", ["", "/pages/1", "/file"])
def test_preview_requires_auth(preview, suffix):
    client, _, _ = preview
    assert client.get(f"/documents/sample.pdf{suffix}").status_code == 401


def test_preview_metadata_download_and_rendered_pixels(preview):
    client, headers, contents = preview
    metadata = client.get("/documents/sample.pdf", headers=headers)
    assert metadata.status_code == 200
    assert metadata.headers["Cache-Control"] == "no-store"
    revision = metadata.json()["revision"]
    assert revision == hashlib.sha256(contents).hexdigest()
    assert metadata.json()["page_count"] >= 1
    image = client.get(
        "/documents/sample.pdf/pages/1", params={"revision": revision}, headers=headers
    )
    assert image.status_code == 200
    assert image.headers["Content-Type"] == "image/png"
    assert image.headers["Cache-Control"] == "no-store"
    with Image.open(BytesIO(image.content)) as page:
        assert max(page.size) <= pdf_preview.MAX_PREVIEW_EDGE
        assert max(ImageStat.Stat(page).stddev) > 5
    download = client.get("/documents/sample.pdf/file", headers=headers)
    assert download.content == contents
    assert (
        download.headers["Content-Disposition"] == 'attachment; filename="sample.pdf"'
    )
    assert download.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("suffix", ["", "/pages/1", "/file"])
def test_replaced_source_is_rejected(preview, suffix):
    client, headers, _ = preview
    response = client.get(
        f"/documents/sample.pdf{suffix}", params={"revision": "0" * 64}, headers=headers
    )
    assert response.status_code == 409


@pytest.mark.parametrize(
    "filename", ["missing.pdf", "linked.pdf", ".env", "..%2Fsample.pdf"]
)
def test_preview_rejects_missing_symlink_and_unsafe_paths(preview, tmp_path, filename):
    client, headers, _ = preview
    (tmp_path / "linked.pdf").symlink_to(tmp_path / "sample.pdf")
    assert client.get(f"/documents/{filename}", headers=headers).status_code == 404


@pytest.mark.parametrize("page", [0, -1, 999, "bad"])
def test_preview_rejects_invalid_page(preview, page):
    client, headers, _ = preview
    assert (
        client.get(f"/documents/sample.pdf/pages/{page}", headers=headers).status_code
        == 422
    )


def test_preview_rejects_size_page_and_invalid_pdf_limits(
    preview, monkeypatch, tmp_path
):
    client, headers, _ = preview
    monkeypatch.setattr(main, "RAG_MAX_UPLOAD_BYTES", 5)
    assert client.get("/documents/sample.pdf", headers=headers).status_code == 413
    monkeypatch.setattr(main, "RAG_MAX_UPLOAD_BYTES", 100000)
    (tmp_path / "sample.pdf").write_bytes(b"not a pdf")
    assert client.get("/documents/sample.pdf/file", headers=headers).status_code == 422
    writer, output = PdfWriter(), BytesIO()
    writer.add_blank_page(width=600, height=800)
    writer.add_blank_page(width=600, height=800)
    writer.write(output)
    (tmp_path / "sample.pdf").write_bytes(output.getvalue())
    monkeypatch.setattr(main, "RAG_MAX_PDF_PAGES", 1)
    assert (
        client.get("/documents/sample.pdf/pages/1", headers=headers).status_code == 422
    )


def test_preview_busy_and_renderer_failures_are_safe(preview, monkeypatch):
    client, headers, _ = preview
    monkeypatch.setattr(
        main, "render_pdf_page", Mock(side_effect=pdf_preview.PreviewBusyError())
    )
    busy = client.get("/documents/sample.pdf/pages/1", headers=headers)
    assert busy.status_code == 503
    assert busy.headers["Retry-After"] == "5"
    monkeypatch.setattr(
        main, "render_pdf_page", Mock(side_effect=ValueError("private/path"))
    )
    failed = client.get("/documents/sample.pdf/pages/1", headers=headers)
    assert failed.status_code == 422
    assert "private/path" not in failed.text


def test_preview_is_rate_limited(preview, monkeypatch):
    client, headers, _ = preview
    monkeypatch.setattr(main, "RAG_RATE_LIMIT_REQUESTS", 1)
    assert client.get("/documents/sample.pdf", headers=headers).status_code == 200
    assert (
        client.get("/documents/sample.pdf/pages/1", headers=headers).status_code == 429
    )


def test_sources_include_document_revision():
    source = main.build_sources(
        [
            Document(
                page_content="Text",
                metadata={
                    "source": "/private/path/sample.pdf",
                    "page": 2,
                    "source_sha256": "a" * 64,
                },
            )
        ]
    )[0]
    assert source.source == "sample.pdf"
    assert source.page == 3
    assert source.revision == "a" * 64


def test_renderer_releases_lock_after_invalid_pdf():
    with pytest.raises(ValueError):
        pdf_preview.render_pdf_page(b"invalid", 1)
    assert not pdf_preview._render_lock.locked()
    assert pdf_preview.render_pdf_page(SAMPLE.read_bytes(), 1).startswith(b"\x89PNG")
