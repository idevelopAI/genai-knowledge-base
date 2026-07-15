import os
import time

import requests
import streamlit as st


st.set_page_config(
    page_title="GenAI Wissensdatenbank",
    page_icon="🤖",
    layout="wide",
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

    documents = refresh_document_list()
    if documents:
        for document in documents:
            st.caption(document)
    else:
        st.caption("Keine Dokumente gefunden oder Backend nicht erreichbar.")

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
                st.success(
                    "Index aktualisiert: "
                    f"{payload['pdf_count']} PDFs, {payload['chunks_created']} Chunks."
                )
                st.session_state.messages = []
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
                st.success(
                    "Index aktualisiert: "
                    f"{payload['pdf_count']} PDFs, {payload['chunks_created']} Chunks."
                )
                st.session_state.messages = []
            except (requests.exceptions.RequestException, RuntimeError) as e:
                st.error(f"Reindex fehlgeschlagen: {str(e)}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if prompt := st.chat_input("Deine Frage an das Dokument..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(KI ueberlegt...)*")

        try:
            response = requests.post(
                API_URL,
                json={"question": prompt},
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
