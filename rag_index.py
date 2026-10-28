import hashlib
import io
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError


ACTIVE_COLLECTION_FILENAME = ".active_collection"
DELETION_BACKUP_PATTERN = re.compile(r"^\.(?P<name>.+)\.[0-9a-f]{32}\.deleting$")
UPLOAD_STAGING_PREFIX = ".upload-"
INDEX_METADATA_VERSION = 1


@dataclass(frozen=True)
class IndexHealth:
    state: str
    reason: str = ""
    pdf_count: int = 0
    chunk_count: int = 0


@dataclass(frozen=True)
class IndexStats:
    pdf_count: int
    pages_loaded: int
    chunks_created: int
    data_dir: str
    vector_store: str
    collection_name: str


def list_pdf_files(data_dir: str) -> list[Path]:
    directory = Path(data_dir)
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name,
    )


def pdf_digest(path: Path) -> str:
    with path.open("rb") as document:
        return hashlib.file_digest(document, "sha256").hexdigest()


def corpus_fingerprint(pdf_files: list[Path]) -> str:
    manifest = sorted((path.name, pdf_digest(path)) for path in pdf_files)
    return hashlib.sha256(json.dumps(manifest).encode("utf-8")).hexdigest()


def index_settings(embedding_model: str, chunk_size: int, chunk_overlap: int) -> dict:
    return {
        "rag_schema": INDEX_METADATA_VERSION,
        "rag_embedding_model": embedding_model,
        "rag_chunk_size": chunk_size,
        "rag_chunk_overlap": chunk_overlap,
    }


def read_index_status(
    *,
    data_dir: str,
    persist_directory: str,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    loaded_collection: str,
) -> IndexHealth:
    if not Path(persist_directory).is_dir():
        return IndexHealth("empty", "no_index")
    active = get_active_collection_name(persist_directory, collection_name)
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(active, embedding_function=None)
    except NotFoundError:
        return IndexHealth("empty", "no_index")
    count = collection.count()
    if not count:
        return IndexHealth("empty", "no_chunks")
    metadata = collection.metadata or {}
    pdf_count = int(metadata.get("rag_pdf_count", 0))
    if metadata.get("rag_schema") != INDEX_METADATA_VERSION or not metadata.get(
        "rag_corpus"
    ):
        return IndexHealth("needs_rebuild", "metadata_missing", pdf_count, count)
    if any(
        metadata.get(key) != value
        for key, value in index_settings(
            embedding_model, chunk_size, chunk_overlap
        ).items()
    ):
        return IndexHealth("needs_rebuild", "settings_changed", pdf_count, count)
    if corpus_fingerprint(list_pdf_files(data_dir)) != metadata["rag_corpus"]:
        return IndexHealth("needs_rebuild", "documents_changed", pdf_count, count)
    if loaded_collection != active:
        return IndexHealth("needs_reload", "reload_required", pdf_count, count)
    return IndexHealth("ready", pdf_count=pdf_count, chunk_count=count)


@dataclass(frozen=True)
class RecoveryReport:
    restored: list[str]
    conflicting_backups: list[str]
    removed_staging_dirs: int


def recover_interrupted_changes(data_dir: str) -> RecoveryReport:
    """Undo file changes left behind when an indexing job was interrupted."""
    directory = Path(data_dir)
    restored: list[str] = []
    conflicting: list[str] = []
    removed_staging_dirs = 0
    if not directory.is_dir():
        return RecoveryReport(restored, conflicting, removed_staging_dirs)

    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            continue
        match = DELETION_BACKUP_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            original = directory / match["name"]
            if original.exists():
                # Never overwrite a file that reappeared; leave it for manual review.
                conflicting.append(path.name)
                continue
            path.replace(original)
            restored.append(original.name)
        elif path.name.startswith(UPLOAD_STAGING_PREFIX) and path.is_dir():
            shutil.rmtree(path)
            removed_staging_dirs += 1
    return RecoveryReport(restored, conflicting, removed_staging_dirs)


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        return "document.pdf"
    if not name.lower().endswith(".pdf"):
        return f"{name}.pdf"
    return name


def next_available_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_pdf_bytes(data: bytes, filename: str, data_dir: str) -> Path:
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = next_available_path(directory, sanitize_filename(filename))
    destination.write_bytes(data)
    return destination


def validate_pdf_bytes(data: bytes, max_pages: int) -> int:
    if not data.startswith(b"%PDF-"):
        raise ValueError("Uploaded file does not have a valid PDF signature.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, OSError, ValueError, EOFError) as exc:
        raise ValueError("Uploaded file is not a readable PDF.") from exc

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    try:
        page_count = len(reader.pages)
    except (PdfReadError, OSError, ValueError, EOFError) as exc:
        raise ValueError("Uploaded file is not a readable PDF.") from exc

    if page_count == 0:
        raise ValueError("Uploaded PDF does not contain any pages.")
    if page_count > max_pages:
        raise ValueError(f"Uploaded PDF exceeds the {max_pages}-page limit.")
    return page_count


def load_pdf_documents(pdf_files: list[Path]) -> list[Document]:
    documents = []
    for pdf_path in pdf_files:
        revision = pdf_digest(pdf_path)
        try:
            reader = PdfReader(pdf_path)
        except (PdfReadError, OSError, ValueError, EOFError) as exc:
            raise ValueError(f"Unable to read PDF {pdf_path.name!r}.") from exc

        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDF {pdf_path.name!r} is not supported.")

        try:
            for page_number, page in enumerate(reader.pages):
                documents.append(
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={
                            "source": str(pdf_path),
                            "source_name": pdf_path.name,
                            "page": page_number,
                            "source_sha256": revision,
                        },
                    )
                )
        except (PdfReadError, OSError, ValueError, EOFError) as exc:
            raise ValueError(
                f"Unable to extract text from PDF {pdf_path.name!r}."
            ) from exc
    return documents


def get_active_collection_name(
    persist_directory: str,
    default_collection_name: str,
) -> str:
    pointer = Path(persist_directory) / ACTIVE_COLLECTION_FILENAME
    try:
        collection_name = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default_collection_name
    return collection_name or default_collection_name


def set_active_collection_name(persist_directory: str, collection_name: str) -> None:
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / ACTIVE_COLLECTION_FILENAME
    temporary_pointer = directory / (
        f"{ACTIVE_COLLECTION_FILENAME}.{uuid.uuid4().hex}.tmp"
    )
    temporary_pointer.write_text(collection_name, encoding="utf-8")
    temporary_pointer.replace(pointer)


def delete_collection(persist_directory: str, collection_name: str) -> None:
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(name=collection_name)
    except NotFoundError:
        return


def build_vector_db(
    *,
    data_dir: str,
    persist_directory: str,
    embedding_model: str,
    google_api_key: str,
    collection_name: str = "langchain",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IndexStats:
    pdf_files = list_pdf_files(data_dir)
    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")
    fingerprint = corpus_fingerprint(pdf_files)

    docs = load_pdf_documents(pdf_files)
    if not docs:
        raise ValueError(f"No document pages could be loaded from {data_dir}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(docs)
    if not splits:
        raise ValueError(f"No extractable text found in PDFs from {data_dir}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=google_api_key,
    )

    versioned_collection_name = f"{collection_name}-{uuid.uuid4().hex}"
    try:
        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=versioned_collection_name,
            collection_configuration={"hnsw": {"space": "cosine"}},
            collection_metadata={
                **index_settings(embedding_model, chunk_size, chunk_overlap),
                "rag_corpus": fingerprint,
                "rag_pdf_count": len(pdf_files),
            },
        )
    except BaseException:
        # Embedding can fail partway (for example on quota); drop the partial build.
        try:
            delete_collection(persist_directory, versioned_collection_name)
        except Exception:
            pass
        raise
    if fingerprint != corpus_fingerprint(list_pdf_files(data_dir)):
        delete_collection(persist_directory, versioned_collection_name)
        raise ValueError("PDF files changed during indexing. Rebuild the index again.")
    set_active_collection_name(persist_directory, versioned_collection_name)

    return IndexStats(
        pdf_count=len(pdf_files),
        pages_loaded=len(docs),
        chunks_created=len(splits),
        data_dir=data_dir,
        vector_store=persist_directory,
        collection_name=versioned_collection_name,
    )
