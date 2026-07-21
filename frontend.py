import os
import time
from urllib.parse import quote

import requests
import streamlit as st


st.set_page_config(
    page_title="GenAI Wissensdatenbank",
    page_icon="🤖",
    layout="wide",
)
st.markdown(
    """
    <style>
    h1 {
        font-size: 40px !important;
        line-height: 1.2 !important;
        letter-spacing: 0 !important;
    }
    @media (min-width: 769px) {
        section[data-testid="stSidebar"] {
            width: 360px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 360px !important;
        }
    }
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            box-sizing: border-box !important;
            max-width: 100% !important;
            padding-left: 16px !important;
            padding-right: 16px !important;
        }
        h1 {
            font-size: 24px !important;
            overflow-wrap: anywhere !important;
        }
        [data-testid="stMainBlockContainer"] p {
            overflow-wrap: anywhere !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Interaktive Wissensdatenbank")
st.markdown(
    "Stelle Fragen an deine internen Dokumente. Die KI antwortet basierend "
    "auf dem bereitgestellten Kontext."
)

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000/ask")
API_BASE_URL = API_URL.removesuffix("/ask")
DOCUMENTS_URL = os.environ.get("RAG_DOCUMENTS_URL", f"{API_BASE_URL}/documents")
REINDEX_URL = os.environ.get("RAG_REINDEX_URL", f"{API_BASE_URL}/reindex")
JOBS_URL = os.environ.get("RAG_JOBS_URL", f"{API_BASE_URL}/jobs")
RAG_API_TOKEN = os.environ.get("RAG_API_TOKEN")


def auth_headers():
    if not RAG_API_TOKEN:
        return {}
    return {"Authorization": f"Bearer {RAG_API_TOKEN}"}


def render_sources(sources):
    if not sources:
        return

    with st.expander("Quellen"):
        for index, source in enumerate(sources, start=1):
            page = source.get("page")
            page_text = f", Seite {page}" if page else ""
            st.markdown(f"**{index}. {source.get('source', 'unknown')}{page_text}**")
            st.caption(source.get("excerpt", ""))


def refresh_document_list():
    try:
        response = requests.get(DOCUMENTS_URL, headers=auth_headers(), timeout=10)
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.exceptions.RequestException:
        return []


def wait_for_index_job(job, timeout=180):
    job_id = job["job_id"]
    deadline = time.monotonic() + timeout
    current = job

    while current.get("status") in {"queued", "running"}:
        if time.monotonic() >= deadline:
            raise RuntimeError("Zeitueberschreitung bei der Indexierung.")
        time.sleep(1)
        response = requests.get(
            f"{JOBS_URL}/{job_id}",
            headers=auth_headers(),
            timeout=10,
        )
        response.raise_for_status()
        current = response.json()

    if current.get("status") == "failed":
        raise RuntimeError(current.get("error") or "Indexierung fehlgeschlagen.")
    if current.get("status") != "completed" or not current.get("result"):
        raise RuntimeError("Ungueltiger Status der Indexierung.")
    return current["result"]


with st.sidebar:
    st.header("Dokumente")
    if operation_message := st.session_state.pop("document_operation_message", None):
        st.success(operation_message)

    documents = refresh_document_list()
    if documents:
        for document in documents:
            st.caption(document)
    else:
        st.caption("Keine Dokumente gefunden oder Backend nicht erreichbar.")

    current_selection = st.session_state.get("selected_documents", documents)
    st.session_state.selected_documents = [
        document for document in current_selection if document in documents
    ]
    if documents and not st.session_state.selected_documents:
        st.session_state.selected_documents = documents

    selected_documents = st.multiselect(
        "Dokumentauswahl",
        options=documents,
        key="selected_documents",
        disabled=not documents,
    )

    document_to_delete = st.selectbox(
        "Dokument loeschen",
        options=documents,
        index=None,
        placeholder="PDF auswaehlen",
        disabled=len(documents) <= 1,
    )
    if st.button(
        "Ausgewaehltes Dokument loeschen",
        disabled=document_to_delete is None or len(documents) <= 1,
    ):
        with st.spinner("Dokument wird geloescht und der Index aktualisiert..."):
            try:
                response = requests.delete(
                    f"{DOCUMENTS_URL}/{quote(document_to_delete, safe='')}",
                    headers=auth_headers(),
                    timeout=30,
                )
                response.raise_for_status()
                payload = wait_for_index_job(response.json())
                st.session_state.document_operation_message = (
                    "Dokument geloescht: "
                    f"{payload['pdf_count']} PDFs, {payload['chunks_created']} Chunks."
                )
                st.session_state.messages = []
                st.rerun()
            except (requests.exceptions.RequestException, RuntimeError) as e:
                st.error(f"Loeschen fehlgeschlagen: {str(e)}")

    uploaded_file = st.file_uploader("PDF hochladen", type=["pdf"])
    if st.button("Hochladen und indexieren", disabled=uploaded_file is None):
        with st.spinner("Dokument wird indexiert..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                response = requests.post(
                    DOCUMENTS_URL,
                    files=files,
                    headers=auth_headers(),
                    timeout=30,
                )
                response.raise_for_status()
                payload = wait_for_index_job(response.json())
                st.session_state.document_operation_message = (
                    "Index aktualisiert: "
                    f"{payload['pdf_count']} PDFs, {payload['chunks_created']} Chunks."
                )
                st.session_state.messages = []
                st.rerun()
            except (requests.exceptions.RequestException, RuntimeError) as e:
                st.error(f"Upload fehlgeschlagen: {str(e)}")

    if st.button("Index neu erstellen"):
        with st.spinner("Index wird neu erstellt..."):
            try:
                response = requests.post(
                    REINDEX_URL,
                    headers=auth_headers(),
                    timeout=30,
                )
                response.raise_for_status()
                payload = wait_for_index_job(response.json())
                st.session_state.document_operation_message = (
                    "Index aktualisiert: "
                    f"{payload['pdf_count']} PDFs, {payload['chunks_created']} Chunks."
                )
                st.session_state.messages = []
                st.rerun()
            except (requests.exceptions.RequestException, RuntimeError) as e:
                st.error(f"Reindex fehlgeschlagen: {str(e)}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

chat_disabled = not documents or not selected_documents
if prompt := st.chat_input(
    "Deine Frage an die ausgewaehlten Dokumente...",
    disabled=chat_disabled,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(KI ueberlegt...)*")

        try:
            response = requests.post(
                API_URL,
                json={"question": prompt, "documents": selected_documents},
                headers=auth_headers(),
                timeout=60,
            )
            response.raise_for_status()

            payload = response.json()
            answer = payload["answer"]
            sources = payload.get("sources", [])

            message_placeholder.markdown(answer)
            render_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except requests.exceptions.RequestException as e:
            error_msg = (
                "Fehler: Das Backend ist nicht erreichbar oder konnte nicht antworten. "
                f"Laeuft der API-Server? ({str(e)})"
            )
            message_placeholder.error(error_msg)
