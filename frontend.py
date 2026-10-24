import math
import os
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st


st.set_page_config(
    page_title="Knowledge Base",
    page_icon=":material/menu_book:",
    layout="wide",
)
st.html(Path(__file__).with_name("frontend.css"))

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000/ask")
API_BASE_URL = API_URL.removesuffix("/ask")
DOCUMENTS_URL = os.environ.get("RAG_DOCUMENTS_URL", f"{API_BASE_URL}/documents")
REINDEX_URL = os.environ.get("RAG_REINDEX_URL", f"{API_BASE_URL}/reindex")
JOBS_URL = os.environ.get("RAG_JOBS_URL", f"{API_BASE_URL}/jobs")
RAG_API_TOKEN = os.environ.get("RAG_API_TOKEN")


def auth_headers():
    return {"Authorization": f"Bearer {RAG_API_TOKEN}"} if RAG_API_TOKEN else {}


def request_error(error):
    response = getattr(error, "response", None)
    if response is not None:
        status = response.status_code
        if status == 401:
            message = (
                "Zugriff verweigert. Der API-Token fehlt oder stimmt nicht überein."
            )
        elif status == 429:
            message = "Zu viele Anfragen. Bitte kurz warten."
        elif status == 503:
            message = "Der Dienst ist vorübergehend nicht verfügbar."
        elif status == 413:
            message = "Die PDF-Datei überschreitet das Upload-Limit."
        elif status == 409:
            message = "Die Aktion ist aktuell nicht möglich. Bitte die Dokumentliste aktualisieren."
        elif status in {400, 422}:
            message = (
                "Ungültige Anfrage. Bitte Datei, Frage und Dokumentauswahl prüfen."
            )
        else:
            message = "Die Anfrage konnte nicht abgeschlossen werden."
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            message += f" Erneut versuchen in {retry_after} Sekunden."
        return {"message": message, "request_id": response.headers.get("X-Request-ID")}
    if isinstance(error, requests.Timeout):
        return {
            "message": "Die Anfrage dauert zu lange. Bitte später erneut versuchen."
        }
    return {"message": "Keine Verbindung zur API. Bitte den Backend-Status prüfen."}


def show_error(error):
    st.error(error["message"], icon=":material/error_outline:")
    if error.get("request_id"):
        st.caption(f"Anfrage: {error['request_id']}")


def load_backend_state():
    try:
        response = requests.get(DOCUMENTS_URL, headers=auth_headers(), timeout=10)
        response.raise_for_status()
        documents = response.json()["documents"]
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return documents, response.json(), None
    except requests.RequestException as error:
        return [], {}, request_error(error)
    except (ValueError, KeyError):
        return [], {}, {"message": "Die API hat eine ungültige Antwort geliefert."}


def answer_timeout(health):
    configured = os.environ.get("RAG_REQUEST_TIMEOUT_SECONDS", "").strip()
    try:
        return max(1, int(configured or health.get("answer_timeout_seconds", 240)))
    except (ValueError, TypeError):
        return 240


@st.dialog("Dokumentvorschau", width="large", icon=":material/description:")
def preview_source(source):
    filename = source["source"]
    url = f"{DOCUMENTS_URL}/{quote(filename, safe='')}"
    st.text(filename)
    try:
        response = requests.get(
            url,
            params={"revision": source.get("revision")},
            headers=auth_headers(),
            timeout=15,
        )
        response.raise_for_status()
        details = response.json()
        params = {"revision": details["revision"]}
        page_count = int(details["page_count"])
        if not 1 <= st.session_state.preview_page <= page_count:
            st.warning("Die angegebene Seite ist nicht verfügbar.")
            return
        with st.container(horizontal=True, vertical_alignment="bottom"):
            page = st.number_input(
                "Seite",
                min_value=1,
                max_value=page_count,
                step=1,
                key="preview_page",
                width=160,
            )
            scale = st.segmented_control(
                "Ansicht",
                ["Ganze Seite", "Seitenbreite"],
                default="Ganze Seite",
                required=True,
                key="preview_scale",
                label_visibility="collapsed",
            )
        st.caption(f"Seite {page} von {page_count}")
        response = requests.get(
            f"{url}/pages/{page}", params=params, headers=auth_headers(), timeout=30
        )
        response.raise_for_status()
        with st.container(
            key="pdf_preview_width" if scale == "Seitenbreite" else "pdf_preview_fit"
        ):
            st.image(response.content, width="stretch", output_format="PNG")
        cached = st.session_state.get("preview_download", {})
        if cached.get("revision") != details["revision"]:
            response = requests.get(
                f"{url}/file", params=params, headers=auth_headers(), timeout=30
            )
            response.raise_for_status()
            cached = {"revision": details["revision"], "contents": response.content}
            st.session_state.preview_download = cached
        st.download_button(
            "PDF herunterladen",
            cached["contents"],
            filename,
            "application/pdf",
            icon=":material/download:",
            on_click="ignore",
        )
    except requests.RequestException as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status == 404:
            st.warning("Dieses Dokument ist nicht mehr verfügbar.")
        elif status == 409:
            st.warning("Das Dokument wurde seit dieser Antwort geändert.")
        else:
            show_error(request_error(error))
    except (ValueError, KeyError, TypeError):
        st.error("Die Dokumentvorschau konnte nicht geladen werden.")


def render_sources(sources, message_index):
    if not sources:
        return
    label = "1 Quelle" if len(sources) == 1 else f"{len(sources)} Quellen"
    with st.expander(label, icon=":material/description:"):
        for index, source in enumerate(sources, start=1):
            if index > 1:
                st.divider()
            page = source.get("page")
            page_label = f" · Seite {page}" if page else ""
            st.text(f"[Source {index}] {source.get('source', 'Dokument')}{page_label}")
            st.text(source.get("excerpt", ""))
            if st.button(
                "Quelle öffnen",
                icon=":material/open_in_new:",
                key=f"source_{message_index}_{index}",
            ):
                st.session_state.preview_page = page or 1
                preview_source(source)


def index_status_label(health):
    state = health.get("index_state")
    return {
        "unconfigured": "Modell-Konfiguration unvollständig",
        "empty": "Kein durchsuchbarer Index",
        "needs_rebuild": "Index muss neu erstellt werden",
        "needs_reload": "Backend-Neustart erforderlich",
        "building": "Index wird aktualisiert",
        "unavailable": "Index nicht verfügbar",
        "ready": "Index aktuell",
    }.get(
        state,
        "Index aktuell" if health.get("rag_ready") else "Konfiguration erforderlich",
    )


def conversation_markdown():
    parts = ["# Knowledge Base\n"]
    for message in st.session_state.messages:
        title = "Frage" if message["role"] == "user" else "Antwort"
        content = message.get("content", message.get("error", {}).get("message", ""))
        parts.append(f"## {title}\n\n{content}\n")
        for index, source in enumerate(message.get("sources", []), start=1):
            parts.append(
                f"[Source {index}] {source.get('source', 'Dokument')}, "
                f"Seite {source.get('page', '?')}\n\n{source.get('excerpt', '')}\n"
            )
    return "\n".join(parts)


def queue_question(question=None):
    question = question or st.session_state.question_input
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.pending_question = question


def retry_question():
    st.session_state.messages.pop()
    st.session_state.pending_question = st.session_state.messages[-1]["content"]


def clear_chat():
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)


def start_index_job(method, url, **kwargs):
    try:
        response = method(url, headers=auth_headers(), timeout=30, **kwargs)
        response.raise_for_status()
        job = response.json()
        if "job_id" not in job:
            raise ValueError("Missing job ID")
        st.session_state.pending_job = job
        st.rerun()
    except requests.RequestException as error:
        show_error(request_error(error))
    except ValueError:
        st.error("Die API hat keinen gültigen Auftrag zurückgegeben.")


@st.fragment(run_every=2)
def render_index_job():
    job = st.session_state.get("pending_job")
    if not job:
        return
    try:
        response = requests.get(
            f"{JOBS_URL}/{job['job_id']}", headers=auth_headers(), timeout=10
        )
        response.raise_for_status()
        current = response.json()
    except requests.RequestException as error:
        show_error(request_error(error))
        if st.button("Status schließen", key="dismiss_job", icon=":material/close:"):
            st.session_state.pop("pending_job", None)
            st.rerun()
        return
    except ValueError:
        st.warning("Der Auftragsstatus konnte nicht gelesen werden.")
        return

    if current.get("status") in {"queued", "running"}:
        label = (
            "Wartet auf Indexierung"
            if current["status"] == "queued"
            else "Index wird aktualisiert"
        )
        with st.status(label, expanded=False):
            if current.get("filename"):
                st.text(current["filename"])
        return

    st.session_state.pop("pending_job", None)
    if current.get("status") == "completed":
        st.session_state.messages = []
        st.session_state.upload_version += 1
        st.session_state.document_notice = "Index aktualisiert."
        if current.get("operation") == "upload" and current.get("filename"):
            st.session_state.select_after_refresh = current["filename"]
    else:
        st.session_state.document_error = {
            "message": "Die Indexierung ist fehlgeschlagen. Bitte den Backend-Status prüfen.",
            "request_id": current.get("request_id"),
        }
    st.rerun()


@st.dialog("Dokument löschen", icon=":material/delete_outline:")
def confirm_deletion(filename):
    st.text(filename)
    st.warning("Die PDF-Datei wird entfernt und der Index neu erstellt.")
    cancel, confirm = st.columns(2)
    if cancel.button("Abbrechen", width="stretch"):
        st.rerun()
    if confirm.button(
        "Löschen", icon=":material/delete:", type="primary", width="stretch"
    ):
        start_index_job(requests.delete, f"{DOCUMENTS_URL}/{quote(filename, safe='')}")


st.session_state.setdefault("messages", [])
st.session_state.setdefault("upload_version", 0)
documents, health, backend_error = load_backend_state()
busy = bool(st.session_state.get("pending_job"))

# An explicit empty selection must stay empty, including across API outages.
if backend_error is None:
    st.session_state.available_documents = documents
    current_selection = st.session_state.get("selected_documents", documents)
    st.session_state.selected_documents = [
        name for name in current_selection if name in documents
    ]
    new_document = st.session_state.pop("select_after_refresh", None)
    if (
        new_document in documents
        and new_document not in st.session_state.selected_documents
    ):
        st.session_state.selected_documents.append(new_document)

with st.sidebar:
    with st.container(key="brand"):
        st.markdown("**Knowledge Base**")
        st.caption("Dokumente & Wissen")
    st.divider()

    title, refresh = st.columns([5, 1], vertical_alignment="center")
    title.subheader("Bibliothek")
    refresh.button(
        "", icon=":material/refresh:", help="Dokumentliste aktualisieren", key="refresh"
    )
    if notice := st.session_state.pop("document_notice", None):
        st.success(notice)
    if error := st.session_state.pop("document_error", None):
        show_error(error)
    if backend_error:
        show_error(backend_error)

    library_tab, upload_tab = st.tabs(["Dokumente", "Hochladen"])
    with library_tab:
        st.caption(
            f"{len(documents)} PDF-Dokumente"
            if not backend_error
            else "Nicht verfügbar"
        )
        selected_documents = st.multiselect(
            "Suchbereich",
            options=st.session_state.get("available_documents", []),
            key="selected_documents",
            placeholder="Dokumente auswählen",
            disabled=not documents or busy or bool(backend_error),
        )
        if documents and not selected_documents and not backend_error:
            st.caption("Kein Dokument ausgewählt.")
        elif not documents and not backend_error:
            st.caption("Noch keine Dokumente vorhanden.")

        with st.expander("Verwalten", icon=":material/tune:"):
            document_to_delete = st.selectbox(
                "Dokument",
                options=documents,
                index=None,
                placeholder="PDF auswählen",
                disabled=len(documents) <= 1 or busy,
            )
            if st.button(
                "Dokument löschen",
                icon=":material/delete_outline:",
                disabled=document_to_delete is None or len(documents) <= 1 or busy,
                width="stretch",
            ):
                confirm_deletion(document_to_delete)
            if len(documents) == 1:
                st.caption("Das letzte Dokument bleibt erhalten.")
            st.divider()
            if st.button(
                "Index neu erstellen",
                icon=":material/sync:",
                disabled=not documents or busy or bool(backend_error),
                width="stretch",
            ):
                start_index_job(requests.post, REINDEX_URL)

    with upload_tab:
        uploaded_file = st.file_uploader(
            "PDF-Datei",
            type=["pdf"],
            disabled=busy or bool(backend_error),
            key=f"upload_{st.session_state.upload_version}",
            max_upload_size=max(
                1,
                math.ceil(
                    health.get("max_upload_bytes", 10 * 1024 * 1024) / (1024 * 1024)
                ),
            ),
            help=f"Maximal {health.get('max_pdf_pages', 200)} Seiten pro PDF.",
        )
        if st.button(
            "Hochladen & indexieren",
            icon=":material/upload_file:",
            type="primary",
            disabled=uploaded_file is None or busy or bool(backend_error),
            width="stretch",
        ):
            start_index_job(
                requests.post,
                DOCUMENTS_URL,
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                },
            )

    render_index_job()
    st.divider()
    with st.container(key="connection"):
        if backend_error:
            st.caption(":material/cloud_off: API nicht erreichbar")
        else:
            st.caption(":material/check_circle: API verbunden")
            icon = "check_circle" if health.get("rag_ready") else "info"
            st.caption(f":material/{icon}: {index_status_label(health)}")
            if health.get("rag_ready") and "indexed_chunk_count" in health:
                st.caption(f"{health['indexed_chunk_count']} Textabschnitte indexiert")
            if health.get("provider_circuit_state") == "open":
                st.caption(":material/hourglass_top: Gemini pausiert")
        if health.get("ollama_fallback_enabled"):
            st.caption("Ollama-Fallback aktiviert")

chat_disabled = (
    not selected_documents or busy or bool(backend_error) or not health.get("rag_ready")
)
with st.container(key="chat_header"):
    heading, actions = st.columns([4, 1], vertical_alignment="center")
    with heading:
        st.caption("WISSENSDATENBANK")
        st.title("Dokument-Chat")
        scope = (
            "1 Dokument"
            if len(selected_documents) == 1
            else f"{len(selected_documents)} Dokumente"
        )
        st.caption(f"{scope} im Suchbereich")
st.divider()

if not st.session_state.messages:
    with st.container(key="empty_chat"):
        st.subheader("Neuer Chat")
        if backend_error:
            st.caption("Warten auf die API-Verbindung.")
        elif not health.get("rag_ready"):
            st.caption(index_status_label(health))
        elif not selected_documents:
            st.caption("Kein Dokument im Suchbereich.")
        else:
            st.caption("Ausgewählte Dokumente")
            for name in selected_documents[:3]:
                st.text(name)
            if len(selected_documents) > 3:
                st.caption(f"+ {len(selected_documents) - 3} weitere")
            with st.container(key="suggestions"):
                for question, icon in (
                    ("Was sind die Kernaussagen?", ":material/short_text:"),
                    ("Welche Schritte werden beschrieben?", ":material/checklist:"),
                ):
                    st.button(
                        question,
                        icon=icon,
                        disabled=chat_disabled,
                        width="stretch",
                        on_click=queue_question,
                        args=(question,),
                    )

for index, message in enumerate(st.session_state.messages):
    avatar = (
        ":material/person:" if message["role"] == "user" else ":material/menu_book:"
    )
    with st.chat_message(message["role"], avatar=avatar):
        st.caption("Du" if message["role"] == "user" else "Knowledge Base")
        if message.get("error"):
            show_error(message["error"])
            if index == len(st.session_state.messages) - 1:
                st.button(
                    "Erneut versuchen",
                    icon=":material/refresh:",
                    disabled=chat_disabled,
                    on_click=retry_question,
                )
        else:
            st.markdown(message["content"])
            render_sources(message.get("sources", []), index)

st.chat_input(
    "Frage zu deinen Dokumenten",
    max_chars=2000,
    disabled=chat_disabled,
    key="question_input",
    on_submit=queue_question,
)
question = st.session_state.pop("pending_question", None)
if question and not chat_disabled:
    with st.chat_message("assistant", avatar=":material/menu_book:"):
        st.caption("Knowledge Base")
        with st.spinner("Dokumente werden durchsucht …"):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question, "documents": selected_documents},
                    headers=auth_headers(),
                    timeout=(10, answer_timeout(health)),
                )
                response.raise_for_status()
                payload = response.json()
                message = {
                    "role": "assistant",
                    "content": payload["answer"],
                    "sources": payload.get("sources", []),
                }
            except requests.RequestException as error:
                message = {"role": "assistant", "error": request_error(error)}
            except (ValueError, KeyError):
                message = {
                    "role": "assistant",
                    "error": {"message": "Die Antwort konnte nicht gelesen werden."},
                }
        st.session_state.messages.append(message)
        if message.get("error"):
            show_error(message["error"])
            st.button(
                "Erneut versuchen", icon=":material/refresh:", on_click=retry_question
            )
        else:
            st.markdown(message["content"])
            render_sources(
                message.get("sources", []), len(st.session_state.messages) - 1
            )
elif question:
    error = backend_error or {
        "message": "Die Frage konnte nicht gesendet werden. Bitte den Suchbereich prüfen."
    }
    st.session_state.messages.append({"role": "assistant", "error": error})
    show_error(error)

with actions:
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.download_button(
            "",
            conversation_markdown(),
            "knowledge-base-chat.md",
            "text/markdown",
            icon=":material/download:",
            help="Chat herunterladen",
            disabled=not st.session_state.messages,
        )
        st.button(
            "",
            icon=":material/edit_square:",
            help="Neuer Chat",
            key="new_chat",
            disabled=not st.session_state.messages,
            on_click=clear_chat,
        )
