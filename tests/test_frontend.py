import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from streamlit.testing.v1 import AppTest


FRONTEND = Path(__file__).resolve().parents[1] / "frontend.py"


def response(payload, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode()
    result.headers.update(headers or {})
    return result


@pytest.fixture
def backend(monkeypatch):
    state = {
        "documents": ["first.pdf", "second.pdf"],
        "health": {"rag_ready": True, "answer_timeout_seconds": 210},
        "offline": False,
        "job": {"job_id": "test-job", "status": "running"},
    }

    def get(url, **kwargs):
        if state["offline"]:
            raise requests.ConnectionError("private upstream detail")
        if url.endswith("/health"):
            return response(state["health"])
        if "/jobs/" in url:
            return response(state["job"])
        return response({"documents": state["documents"]})

    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    monkeypatch.delenv("RAG_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(requests, "get", get)
    post = Mock(
        return_value=response(
            {
                "answer": "Eine belegte Antwort. [Source 1]",
                "sources": [
                    {"source": "first.pdf", "page": 1, "excerpt": "Beleg aus der PDF."}
                ],
            }
        )
    )
    monkeypatch.setattr(requests, "post", post)
    state["post"] = post
    return state


@pytest.mark.parametrize(
    "state,label",
    [
        ("empty", "Kein durchsuchbarer Index"),
        ("needs_rebuild", "Index muss neu erstellt werden"),
        ("needs_reload", "Backend-Neustart erforderlich"),
        ("unavailable", "Index nicht verfügbar"),
    ],
)
def test_index_state_is_distinct_from_api_connectivity(backend, state, label):
    backend["health"].update(rag_ready=False, model_configured=True, index_state=state)
    app = AppTest.from_file(str(FRONTEND)).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("API verbunden" in caption.value for caption in app.caption)
    assert any(label in caption.value for caption in app.caption)
    backend["post"].assert_not_called()


def test_source_preview_starts_at_cited_page_and_keeps_token_server_side(
    backend, monkeypatch
):
    from io import BytesIO
    from PIL import Image

    original_get = requests.get
    image = BytesIO()
    Image.new("RGB", (20, 30), "white").save(image, format="PNG")
    requests_seen = []

    def get(url, **kwargs):
        if "/documents/" not in url:
            return original_get(url, **kwargs)
        requests_seen.append((url, kwargs))
        if url.endswith("/first.pdf"):
            return response(
                {"filename": "first.pdf", "revision": "a" * 64, "page_count": 4}
            )
        result = response({})
        result._content = b"%PDF-test" if url.endswith("/file") else image.getvalue()
        return result

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setenv("RAG_API_TOKEN", "preview-test-token")
    app = AppTest.from_file(str(FRONTEND)).run()
    app.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Answer",
            "sources": [
                {
                    "source": "first.pdf",
                    "page": 3,
                    "excerpt": "Text",
                    "revision": "a" * 64,
                }
            ],
        }
    ]
    app.run()
    app.button(key="source_0_1").click().run()
    assert not app.exception
    assert app.number_input(key="preview_page").value == 3
    assert any(url.endswith("/pages/3") for url, _ in requests_seen)
    assert all("preview-test-token" not in url for url, _ in requests_seen)
    assert all(
        kwargs["headers"] == {"Authorization": "Bearer preview-test-token"}
        for _, kwargs in requests_seen
    )
    assert all(
        kwargs["params"] == {"revision": "a" * 64} for _, kwargs in requests_seen
    )
    assert len(app.get("download_button")) == 2


@pytest.mark.parametrize(
    "status,text", [(404, "nicht mehr verfügbar"), (409, "geändert")]
)
def test_missing_or_changed_source_has_clear_preview_error(
    backend, monkeypatch, status, text
):
    original_get = requests.get
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **kwargs: (
            response({"detail": "private detail"}, status=status)
            if "/documents/" in url
            else original_get(url, **kwargs)
        ),
    )
    app = AppTest.from_file(str(FRONTEND)).run()
    app.chat_input[0].set_value("Question").run()
    app.button(key="source_1_1").click().run()
    assert not app.exception
    assert any(text in warning.value for warning in app.warning)
    assert "private detail" not in str(app.warning)


def test_clear_selection_stays_empty_and_disables_chat(backend):
    app = AppTest.from_file(str(FRONTEND)).run()
    assert not app.exception
    assert app.multiselect[0].value == backend["documents"]

    app.multiselect[0].set_value([]).run()
    app.run()

    assert not app.exception
    assert app.multiselect[0].value == []
    assert app.chat_input[0].disabled
    backend["post"].assert_not_called()


def test_selection_survives_outage_and_drops_removed_documents(backend):
    app = AppTest.from_file(str(FRONTEND)).run()
    app.multiselect[0].set_value(["second.pdf"]).run()
    backend["offline"] = True
    app.run()
    assert app.chat_input[0].disabled
    assert "private upstream detail" not in str(app.error)
    backend["offline"] = False
    app.run()
    assert app.multiselect[0].value == ["second.pdf"]
    backend["documents"] = ["first.pdf"]
    app.run()
    assert app.multiselect[0].value == []
    assert app.chat_input[0].disabled


def test_question_uses_scope_and_timeout_then_renders_sources(backend):
    app = AppTest.from_file(str(FRONTEND)).run()
    app.multiselect[0].set_value(["first.pdf"]).run()
    app.chat_input[0].set_value("Was steht im Dokument?").run()

    assert not app.exception
    kwargs = backend["post"].call_args.kwargs
    assert kwargs["json"] == {
        "question": "Was steht im Dokument?",
        "documents": ["first.pdf"],
    }
    assert kwargs["timeout"] == (10, 210)
    assert len(app.chat_message) == 2
    source_panels = [*app.expander, *app.status]
    assert any(element.label == "1 Quelle" for element in source_panels)
    assert any("Beleg aus der PDF." in element.value for element in app.text)
    app.button(key="new_chat").click().run()
    assert len(app.chat_message) == 0


def test_timeout_override_is_used(backend, monkeypatch):
    monkeypatch.setenv("RAG_REQUEST_TIMEOUT_SECONDS", "320")
    app = AppTest.from_file(str(FRONTEND)).run()
    app.chat_input[0].set_value("Frage").run()
    assert backend["post"].call_args.kwargs["timeout"] == (10, 320)


def test_error_shows_retry_timing_and_request_id_without_raw_exception(backend):
    backend["post"].return_value = response(
        {"detail": "private upstream detail"},
        status=503,
        headers={"Retry-After": "30", "X-Request-ID": "test-request"},
    )
    app = AppTest.from_file(str(FRONTEND)).run()
    app.chat_input[0].set_value("Frage").run()
    assert not app.exception
    assert "30 Sekunden" in app.error[0].value
    assert "private upstream detail" not in app.error[0].value
    assert any("test-request" in element.value for element in app.caption)
    backend["post"].return_value = response({"answer": "Recovered", "sources": []})
    retry = next(button for button in app.button if button.label == "Erneut versuchen")
    retry.click().run()
    assert not app.exception
    assert not app.error
    assert len(app.chat_message) == 2
    assert len(app.session_state["messages"]) == 2


def test_pending_job_finishes_and_selects_new_document(backend):
    app = AppTest.from_file(str(FRONTEND)).run()
    app.session_state["pending_job"] = backend["job"]
    app.run()
    assert app.chat_input[0].disabled
    backend["documents"].append("new.pdf")
    backend["job"] = {
        "job_id": "test-job",
        "status": "completed",
        "operation": "upload",
        "filename": "new.pdf",
    }
    app.run()
    assert not app.exception
    assert "new.pdf" in app.multiselect[0].value
    assert not app.chat_input[0].disabled
