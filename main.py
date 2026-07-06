import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

from rag_index import IndexStats, build_vector_db, list_pdf_files, save_pdf_bytes


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw_value, default)
        return default

    return max(1, value)


GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
RAG_API_TOKEN = os.environ.get("RAG_API_TOKEN")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "langchain")
RAG_TOP_K = _positive_int_from_env("RAG_TOP_K", 3)
RAG_CHUNK_SIZE = _positive_int_from_env("RAG_CHUNK_SIZE", 1000)
RAG_CHUNK_OVERLAP = _positive_int_from_env("RAG_CHUNK_OVERLAP", 200)
GEMINI_EMBEDDING_MODEL = os.environ.get(
    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
)
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.5-flash")

app = FastAPI(title="GenAI Knowledge Base API")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceDocument(BaseModel):
    source: str
    page: int | None = None
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: list[str]


class IndexResponse(BaseModel):
    status: str
    pdf_count: int
    pages_loaded: int
    chunks_created: int
    data_dir: str
    vector_store: str


class UploadResponse(IndexResponse):
    filename: str


def _elapsed_ms(start_time: float) -> int:
    return round((time.perf_counter() - start_time) * 1000)


def _require_api_token(authorization: str | None = Header(default=None)) -> None:
    if not RAG_API_TOKEN:
        return

    expected = f"Bearer {RAG_API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")


system_prompt = (
    "Du bist ein hilfsbereiter KI-Assistent fuer eine interne Wissensdatenbank. "
    "Nutze ausschliesslich den folgenden bereitgestellten Kontext, um die Frage zu beantworten. "
    "Wenn du die Antwort im Kontext nicht findest, sage, dass du es nicht weisst.\n\n"
    "Kontext: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)


def _build_rag_components() -> tuple[Any | None, Any | None]:
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is not configured; /ask will return 503.")
        return None, None

    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K})
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    return retriever, llm


retriever, llm = _build_rag_components()


def _require_google_api_key() -> str:
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not configured. Set GOOGLE_API_KEY first.",
        )
    return GOOGLE_API_KEY


def _index_response(stats: IndexStats, status: str = "indexed") -> IndexResponse:
    return IndexResponse(
        status=status,
        pdf_count=stats.pdf_count,
        pages_loaded=stats.pages_loaded,
        chunks_created=stats.chunks_created,
        data_dir=stats.data_dir,
        vector_store=stats.vector_store,
    )


def _reload_rag_components() -> None:
    global retriever, llm
    retriever, llm = _build_rag_components()


def _rebuild_vector_store() -> IndexStats:
    google_api_key = _require_google_api_key()
    start_time = time.perf_counter()

    try:
        stats = build_vector_db(
            data_dir=DATA_DIR,
            persist_directory=CHROMA_DB_DIR,
            embedding_model=GEMINI_EMBEDDING_MODEL,
            google_api_key=google_api_key,
            collection_name=CHROMA_COLLECTION_NAME,
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
        )
        _reload_rag_components()
        logger.info(
            "rag_index_rebuilt status=success pdf_count=%s pages_loaded=%s "
            "chunks_created=%s elapsed_ms=%s",
            stats.pdf_count,
            stats.pages_loaded,
            stats.chunks_created,
            _elapsed_ms(start_time),
        )
        return stats
    except ValueError as exc:
        logger.warning(
            "rag_index_rebuilt status=invalid_request elapsed_ms=%s error=%s",
            _elapsed_ms(start_time),
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception(
            "rag_index_rebuilt status=error elapsed_ms=%s",
            _elapsed_ms(start_time),
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to rebuild the vector store right now.",
        )


def _source_label(metadata: dict[str, Any]) -> str:
    source = metadata.get("source") or "unknown"
    return Path(str(source)).name


def _page_number(metadata: dict[str, Any]) -> int | None:
    page = metadata.get("page")
    if isinstance(page, int):
        return page + 1
    return None


def _clean_excerpt(text: str, max_length: int = 320) -> str:
    excerpt = " ".join(text.split())
    if len(excerpt) <= max_length:
        return excerpt
    return f"{excerpt[: max_length - 3]}..."


def format_docs(docs: list[Document]) -> str:
    formatted_docs = []
    for index, doc in enumerate(docs, start=1):
        source = _source_label(doc.metadata)
        page = _page_number(doc.metadata)
        page_text = f", page {page}" if page is not None else ""
        formatted_docs.append(
            f"[Source {index}: {source}{page_text}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted_docs)


def build_sources(docs: list[Document]) -> list[SourceDocument]:
    sources = []
    for doc in docs:
        sources.append(
            SourceDocument(
                source=_source_label(doc.metadata),
                page=_page_number(doc.metadata),
                excerpt=_clean_excerpt(doc.page_content),
            )
        )
    return sources


@app.get("/health")
async def health() -> dict[str, bool | str | int]:
    return {
        "status": "ok",
        "rag_ready": retriever is not None and llm is not None,
        "top_k": RAG_TOP_K,
        "data_dir": DATA_DIR,
        "vector_store": CHROMA_DB_DIR,
        "pdf_count": len(list_pdf_files(DATA_DIR)),
    }


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[Depends(_require_api_token)],
)
async def list_documents() -> DocumentListResponse:
    return DocumentListResponse(
        documents=[path.name for path in list_pdf_files(DATA_DIR)],
    )


@app.post(
    "/reindex",
    response_model=IndexResponse,
    dependencies=[Depends(_require_api_token)],
)
async def reindex_documents() -> IndexResponse:
    stats = _rebuild_vector_store()
    return _index_response(stats, status="reindexed")


@app.post(
    "/documents",
    response_model=UploadResponse,
    dependencies=[Depends(_require_api_token)],
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    start_time = time.perf_counter()
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    _require_google_api_key()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    saved_path = save_pdf_bytes(contents, filename, DATA_DIR)
    stats = _rebuild_vector_store()
    logger.info(
        "rag_document_uploaded filename=%s bytes=%s pdf_count=%s chunks_created=%s "
        "elapsed_ms=%s",
        saved_path.name,
        len(contents),
        stats.pdf_count,
        stats.chunks_created,
        _elapsed_ms(start_time),
    )
    return UploadResponse(
        filename=saved_path.name,
        status="uploaded_and_indexed",
        pdf_count=stats.pdf_count,
        pages_loaded=stats.pages_loaded,
        chunks_created=stats.chunks_created,
        data_dir=stats.data_dir,
        vector_store=stats.vector_store,
    )


@app.post(
    "/ask",
    response_model=QueryResponse,
    dependencies=[Depends(_require_api_token)],
)
async def ask_question(request: QueryRequest) -> QueryResponse:
    request_start = time.perf_counter()
    if retriever is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not configured. Set GOOGLE_API_KEY before using /ask.",
        )

    try:
        retrieval_start = time.perf_counter()
        docs = retriever.invoke(request.question)
        retrieval_ms = _elapsed_ms(retrieval_start)

        llm_start = time.perf_counter()
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {
                "context": format_docs(docs),
                "question": request.question,
            }
        )
        llm_ms = _elapsed_ms(llm_start)
        total_ms = _elapsed_ms(request_start)
        logger.info(
            "rag_question_answered status=success question_chars=%s sources=%s "
            "retrieval_ms=%s llm_ms=%s total_ms=%s",
            len(request.question),
            len(docs),
            retrieval_ms,
            llm_ms,
            total_ms,
        )
        return QueryResponse(answer=answer, sources=build_sources(docs))
    except Exception:
        logger.exception(
            "rag_question_answered status=error question_chars=%s total_ms=%s",
            len(request.question),
            _elapsed_ms(request_start),
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer right now.",
        )
